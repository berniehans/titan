//! Real Qwen3 layer-0 post-attention diagnostic (GPU vs independent CPU).
//! Requires the real fixture, CUDA and NVRTC; intentionally ignored.
use cudarc::driver::CudaDevice;
use engine_core::forward_cpu::{Tensor, TensorType, embed_lookup, matmul, rms_norm, sdpa_decode};
use engine_cuda::{
    BatchedGEMM, CudaError, CudaStream, DeviceBuffer, FlashAttention2, GemvFormat, KvDataType,
    MODE_NORM, NormRope, PagedKvGpu, PagedKvLayout,
};
use engine_io::{GgmlType, GgufReader, LoadedPinned, ModelConfig, load_to_pinned};
use std::{path::PathBuf, sync::Arc};

fn fixture() -> Option<PathBuf> {
    let m = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    [
        std::env::var_os("ENGINE_TESTDATA").map(PathBuf::from),
        Some(m.join("../../testdata/Qwen3-0.6B-Q4_K_M.gguf")),
        Some(m.join("../testdata/Qwen3-0.6B-Q4_K_M.gguf")),
        Some(PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf")),
    ]
    .into_iter()
    .flatten()
    .find(|p| p.exists())
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn vec32(p: &LoadedPinned, n: &str) -> Vec<f32> {
    f32s(p.tensor(n).unwrap())
}
fn metric(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut n, mut d, mut dot, mut aa, mut bb) = (0., 0., 0., 0., 0.);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        n += (x - y) * (x - y);
        d += y * y;
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    ((n / d).sqrt(), dot / (aa * bb).sqrt())
}
fn tensor<'a>(r: &GgufReader, p: &'a LoadedPinned, n: &str) -> Tensor<'a> {
    let i = r.get_tensor(n).unwrap();
    Tensor {
        ty: match i.ggml_type {
            GgmlType::F32 => TensorType::F32,
            GgmlType::Q4_K => TensorType::Q4K,
            GgmlType::Q6_K => TensorType::Q6K,
            x => panic!("unsupported {x:?}"),
        },
        data: p.tensor(n).unwrap(),
        ne0: i.dims[0] as usize,
        ne1: i.dims[1] as usize,
        n_rot: 0,
    }
}
fn gemm_fmt(t: GgmlType) -> GemvFormat {
    match t {
        GgmlType::Q4_K => GemvFormat::Q4K,
        GgmlType::Q6_K => GemvFormat::Q6K,
        x => panic!("unsupported GEMM format {x:?}"),
    }
}
fn upload(s: &CudaStream, d: &Arc<CudaDevice>, v: &[f32]) -> Result<DeviceBuffer, CudaError> {
    let b = DeviceBuffer::alloc(d.clone(), v.len() * 4)?;
    b.copy_from_host(s, &bytes(v))?;
    Ok(b)
}
fn download(s: &CudaStream, b: &DeviceBuffer, n: usize) -> Result<Vec<f32>, CudaError> {
    let mut x = vec![0; n * 4];
    b.copy_to_host(s, &mut x)?;
    Ok(f32s(&x))
}
fn report(stage: &str, fmt: &str, shape: &str, batch: usize, gpu: &[f32], cpu: &[f32]) {
    let (r, c) = metric(gpu, cpu);
    println!("stage={stage} format={fmt} shape={shape} batch={batch} rel_l2={r:.6e} cosine={c:.9}");
    assert!(r < 2e-4 && c > 0.999999, "{stage} parity failed")
}

#[test]
#[ignore]
fn real_layer0_post_attention_parity() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    engine_cuda::ensure_cuda_dll_paths();
    let Some(path) = fixture() else {
        eprintln!("SKIP: Qwen3 fixture not present");
        return Ok(());
    };
    let r = GgufReader::open(&path)?;
    let p = load_to_pinned(&r, &path)?;
    let c = ModelConfig::from_reader(&r)?;
    let (h, hd, nh, nkv) = (
        c.hidden_size as usize,
        c.head_dim as usize,
        c.n_head as usize,
        c.n_head_kv as usize,
    );
    let ids = [9707usize, 1128, 13, 198];
    let n = ids.len();
    let eps = c.rms_norm_eps;
    let emb = tensor(&r, &p, "token_embd.weight");
    let an = vec32(&p, "blk.0.attn_norm.weight");
    let qn = vec32(&p, "blk.0.attn_q_norm.weight");
    let kn = vec32(&p, "blk.0.attn_k_norm.weight");
    let qw = tensor(&r, &p, "blk.0.attn_q.weight");
    let kw = tensor(&r, &p, "blk.0.attn_k.weight");
    let vw = tensor(&r, &p, "blk.0.attn_v.weight");
    let wo = tensor(&r, &p, "blk.0.attn_output.weight");
    let fnorm = vec32(&p, "blk.0.ffn_norm.weight");

    let mut x = Vec::new();
    for &id in &ids {
        x.extend(embed_lookup(&emb, id));
    }
    let mut q = Vec::new();
    let mut k = Vec::new();
    let mut v = Vec::new();
    for (pos, row) in x.chunks_exact(h).enumerate() {
        let z = rms_norm(row, &an, eps);
        let mut a = vec![0.; nh * hd];
        matmul(&mut a, &qw, &z);
        for j in 0..nh {
            q.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(&a[j * hd..(j + 1) * hd], &qn, eps),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        let mut a = vec![0.; nkv * hd];
        matmul(&mut a, &kw, &z);
        for j in 0..nkv {
            k.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(&a[j * hd..(j + 1) * hd], &kn, eps),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        matmul(&mut a, &vw, &z);
        v.extend(a);
    }
    let d = CudaDevice::new(0)?;
    let s = CudaStream::new(d.clone())?;
    let kv = PagedKvGpu::new(d.clone())?;
    let fa = FlashAttention2::new(d.clone())?;
    let g = BatchedGEMM::new(d.clone())?;
    let layout = PagedKvLayout {
        n_blocks: 1,
        block_tokens: 64,
        row_len: nkv * hd,
        data_type: KvDataType::F32,
    };
    let pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let bt = upload(&s, &d, &[0.])?;
    let kd = upload(&s, &d, &k)?;
    let vd = upload(&s, &d, &v)?;
    kv.append_kv(&s, &layout, &pool, &kd, &vd, &bt, 0, n)?;
    let qd = upload(&s, &d, &q)?;
    let att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(&s, &qd, &pool, &bt, &att, nh, nkv, hd, 64, n, 0)?;
    let att_gpu = download(&s, &att, n * nh * hd)?;
    let mut att_cpu = Vec::new();
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let z = t * 2 * nkv * hd;
            ph[z..z + nkv * hd].copy_from_slice(&k[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[z + nkv * hd..z + 2 * nkv * hd]
                .copy_from_slice(&v[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &q[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &att_gpu,
        &att_cpu,
    );
    let wdev = DeviceBuffer::alloc(d.clone(), wo.data.len())?;
    wdev.copy_from_host(&s, wo.data)?;
    let h1 = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &wdev,
        &att,
        &h1,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.0.attn_output.weight").unwrap().ggml_type),
        Some(&upload(&s, &d, &x)?),
    )?;
    let h1g = download(&s, &h1, n * h)?;
    let mut h1c = Vec::new();
    for i in 0..n {
        let mut z = vec![0.; h];
        matmul(&mut z, &wo, &att_cpu[i * nh * hd..(i + 1) * nh * hd]);
        h1c.extend(x[i * h..(i + 1) * h].iter().zip(z).map(|(&a, b)| a + b));
    }
    report(
        "output_projection_residual",
        &format!(
            "{:?}",
            r.get_tensor("blk.0.attn_output.weight").unwrap().ggml_type
        ),
        &format!("[{n},{h}]"),
        n,
        &h1g,
        &h1c,
    );
    let nr = NormRope::new(d.clone())?;
    let fd = upload(&s, &d, &vec![0.; n * h])?;
    let residual = upload(&s, &d, &vec![0.; n * h])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &h1,
        &residual,
        &upload(&s, &d, &fnorm)?,
        &fd,
        &fd,
        eps,
        h,
        0,
        c.rope_freq_base,
        0,
        MODE_NORM,
        None,
        n,
        1,
    )?;
    let fg = download(&s, &fd, n * h)?;
    let fc = h1c
        .chunks_exact(h)
        .flat_map(|z| rms_norm(z, &fnorm, eps))
        .collect::<Vec<_>>();
    report(
        "ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &fg,
        &fc,
    );
    println!(
        "API limitation: stopped before gate/up SwiGLU and down projection; this diagnostic uses only public BatchedGEMM::gemm_with_residual and NormRope stages"
    );
    Ok(())
}
