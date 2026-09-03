//! Minimal real Qwen3 two-layer transition diagnostic (GPU vs independent CPU).
//! Requires the real fixture, CUDA and NVRTC; intentionally ignored.
use cudarc::driver::CudaDevice;
use engine_core::forward_cpu::{Tensor, TensorType, embed_lookup, matmul, rms_norm, sdpa_decode};
use engine_cuda::{
    BatchedGEMM, CudaError, CudaStream, DeviceBuffer, FlashAttention2, GemvFormat, KvDataType,
    MODE_NORM, MODE_ROPE, MODE_SWIGLU, NormRope, PagedKvGpu, PagedKvLayout,
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
fn upload_raw(s: &CudaStream, d: &Arc<CudaDevice>, v: &[u8]) -> Result<DeviceBuffer, CudaError> {
    let b = DeviceBuffer::alloc(d.clone(), v.len())?;
    b.copy_from_host(s, v)?;
    Ok(b)
}
fn download(s: &CudaStream, b: &DeviceBuffer, n: usize) -> Result<Vec<f32>, CudaError> {
    s.sync()?;
    let mut x = vec![0; n * 4];
    b.copy_to_host(s, &mut x)?;
    Ok(f32s(&x))
}
fn report_q_metadata(
    r: &GgufReader,
    p: &LoadedPinned,
    name: &str,
    effective_ne0: usize,
    effective_ne1: usize,
) {
    let info = r.get_tensor(name).unwrap();
    let pinned_len = p.tensor(name).unwrap().len();
    println!(
        "Q metadata: name={name} dims={:?} ggml_type={:?} size_bytes={} pinned_len={} effective_ne0={} effective_ne1={}",
        info.dims, info.ggml_type, info.size_bytes, pinned_len, effective_ne0, effective_ne1
    );
    assert_eq!(
        info.size_bytes as usize, pinned_len,
        "pinned length mismatch for {name}"
    );
    assert_eq!(
        info.dims[0] as usize, effective_ne0,
        "ne0 mismatch for {name}"
    );
    assert_eq!(
        info.dims[1] as usize, effective_ne1,
        "ne1 mismatch for {name}"
    );
}
fn report(stage: &str, fmt: &str, shape: &str, batch: usize, gpu: &[f32], cpu: &[f32]) {
    let (r, c) = metric(gpu, cpu);
    println!("stage={stage} format={fmt} shape={shape} batch={batch} rel_l2={r:.6e} cosine={c:.9}");
    assert!(r < 2e-4 && c > 0.999999, "{stage} parity failed")
}

fn report_raw_q_cross(
    weight: &str,
    input: &str,
    shape: &str,
    batch: usize,
    gpu: &[f32],
    cpu: &[f32],
) {
    let (rel_l2, cosine) = metric(gpu, cpu);
    println!(
        "raw_q_cross weight={weight} input={input} output_shape={shape} batch={batch} rel_l2={rel_l2:.6e} cosine={cosine:.9}"
    );
}

fn report_q_norm_rope_tokens(
    batch: usize,
    n_heads: usize,
    head_dim: usize,
    gpu: &[f32],
    cpu: &[f32],
) {
    if batch < 2 {
        return;
    }
    let mut worst = (0usize, 0usize, f64::NEG_INFINITY, 0.0f64);
    for token in 0..batch {
        for head in 0..n_heads {
            let start = (token * n_heads + head) * head_dim;
            let end = start + head_dim;
            let (rel_l2, cosine) = metric(&gpu[start..end], &cpu[start..end]);
            if rel_l2 > worst.2 {
                worst = (token, head, rel_l2, cosine);
            }
        }
    }
    let (token, head, rel_l2, cosine) = worst;
    let start = (token * n_heads + head) * head_dim;
    let preview = head_dim.min(4);
    println!(
        "layer1_q_norm_rope_worst token={token} head={head} rel_l2={rel_l2:.6e} cosine={cosine:.9} gpu={:?} cpu={:?}",
        &gpu[start..start + preview],
        &cpu[start..start + preview]
    );
}

fn run_transition(chunk: usize) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
    let n = chunk;
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
    let intermediate = r.get_tensor("blk.0.ffn_gate.weight").unwrap().dims[1] as usize;
    let gate_info = r.get_tensor("blk.0.ffn_gate.weight").unwrap();
    let up_info = r.get_tensor("blk.0.ffn_up.weight").unwrap();
    let down_info = r.get_tensor("blk.0.ffn_down.weight").unwrap();
    assert_eq!(
        (gate_info.ggml_type, up_info.ggml_type),
        (GgmlType::Q4_K, GgmlType::Q4_K)
    );
    assert_eq!(down_info.ggml_type, GgmlType::Q6_K);
    println!(
        "FFN tensors: gate={:?} {:?}, up={:?} {:?}, down={:?} {:?}",
        gate_info.ggml_type,
        gate_info.dims,
        up_info.ggml_type,
        up_info.dims,
        down_info.ggml_type,
        down_info.dims
    );

    let gate = tensor(&r, &p, "blk.0.ffn_gate.weight");
    let up = tensor(&r, &p, "blk.0.ffn_up.weight");
    let down = tensor(&r, &p, "blk.0.ffn_down.weight");
    let norm_dev = upload(&s, &d, &fc)?;
    let gate_dev = DeviceBuffer::alloc(d.clone(), gate.data.len())?;
    gate_dev.copy_from_host(&s, gate.data)?;
    let up_dev = DeviceBuffer::alloc(d.clone(), up.data.len())?;
    up_dev.copy_from_host(&s, up.data)?;
    let gate_out = DeviceBuffer::alloc(d.clone(), n * intermediate * 4)?;
    let up_out = DeviceBuffer::alloc(d.clone(), n * intermediate * 4)?;
    let mut gate_cpu = Vec::new();
    let mut up_cpu = Vec::new();
    for (name, weight, info, out) in [
        ("gate", &gate, gate_info, &gate_out),
        ("up", &up, up_info, &up_out),
    ] {
        g.gemm(
            &s,
            if name == "gate" { &gate_dev } else { &up_dev },
            &norm_dev,
            out,
            h,
            intermediate,
            n,
            gemm_fmt(info.ggml_type),
        )?;
        let gpu = download(&s, out, n * intermediate)?;
        let cpu = norm_cpu_for_ffn(&fc, weight, intermediate, h);
        if name == "gate" {
            gate_cpu = cpu.clone();
        } else {
            up_cpu = cpu.clone();
        }
        report(
            name,
            &format!("{:?}", info.ggml_type),
            &format!("[{n},{intermediate}]"),
            n,
            &gpu,
            &cpu,
        );
    }

    let mut sw_cpu = Vec::with_capacity(n * intermediate);
    for row in 0..n {
        sw_cpu.extend(engine_core::forward_cpu::swiglu(
            &gate_cpu[row * intermediate..(row + 1) * intermediate],
            &up_cpu[row * intermediate..(row + 1) * intermediate],
        ));
    }
    let sw_out = DeviceBuffer::alloc(d.clone(), n * intermediate * 4)?;
    let zero = upload(&s, &d, &vec![0.; n * intermediate])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &gate_out,
        &zero,
        &zero,
        &up_out,
        &sw_out,
        eps,
        intermediate,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let sw_gpu = download(&s, &sw_out, n * intermediate)?;
    report(
        "swiglu",
        "F32",
        &format!("[{n},{intermediate}]"),
        n,
        &sw_gpu,
        &sw_cpu,
    );

    let down_dev = DeviceBuffer::alloc(d.clone(), down.data.len())?;
    down_dev.copy_from_host(&s, down.data)?;
    let residual = upload(&s, &d, &h1c)?;
    let down_out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &down_dev,
        &sw_out,
        &down_out,
        intermediate,
        h,
        n,
        gemm_fmt(down_info.ggml_type),
        Some(&residual),
    )?;
    let mut down_cpu = Vec::with_capacity(n * h);
    for row in 0..n {
        let mut z = vec![0.; h];
        matmul(
            &mut z,
            &down,
            &sw_cpu[row * intermediate..(row + 1) * intermediate],
        );
        down_cpu.extend(
            h1c[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b),
        );
    }
    let down_gpu = download(&s, &down_out, n * h)?;
    report(
        "hidden0",
        &format!("{:?}", down_info.ggml_type),
        &format!("[{n},{h}]"),
        n,
        &down_gpu,
        &down_cpu,
    );
    report(
        "down_projection_residual",
        &format!("{:?}", down_info.ggml_type),
        &format!("[{n},{h}]"),
        n,
        &down_gpu,
        &down_cpu,
    );

    // This is the real layer-0 post-down residual, not a synthetic input.
    // Keep the layer-1 transition exactly on ForwardDriver's NormRope contract.
    let l1w = vec32(&p, "blk.1.attn_norm.weight");
    assert_eq!(l1w.len(), h);
    let l1_in = upload(&s, &d, &down_gpu)?;
    let zero = upload(&s, &d, &vec![0.0; n * h])?;
    let l1w_dev = upload(&s, &d, &l1w)?;
    let l1_out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l1_in,
        &zero,
        &l1w_dev,
        &l1_out,
        &l1_out,
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
    let layer1_gpu = download(&s, &l1_out, n * h)?;
    let layer1_cpu = down_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l1w, eps))
        .collect::<Vec<_>>();
    report(
        "layer1_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &layer1_gpu,
        &layer1_cpu,
    );

    // Continue the transition through the real second block.  Keep every CPU
    // value independent of the GPU buffer so a failure identifies the first
    // mismatched stage rather than merely comparing two GPU stages.
    let l1qn = vec32(&p, "blk.1.attn_q_norm.weight");
    let l1kn = vec32(&p, "blk.1.attn_k_norm.weight");
    let l1qw = tensor(&r, &p, "blk.1.attn_q.weight");
    let l1kw = tensor(&r, &p, "blk.1.attn_k.weight");
    let l1vw = tensor(&r, &p, "blk.1.attn_v.weight");
    let l1wo = tensor(&r, &p, "blk.1.attn_output.weight");
    let l1fn = vec32(&p, "blk.1.ffn_norm.weight");
    report_q_metadata(&r, &p, "blk.0.attn_q.weight", h, nh * hd);
    let l0_attn_cpu = x
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &an, eps))
        .collect::<Vec<_>>();
    let l0_attn_gpu = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &upload(&s, &d, &x)?,
        &upload(&s, &d, &vec![0.; n * h])?,
        &upload(&s, &d, &an)?,
        &l0_attn_gpu,
        &l0_attn_gpu,
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
    let l0_q_raw = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l0_q_info = r.get_tensor("blk.0.attn_q.weight").unwrap();
    g.gemm(
        &s,
        &upload_raw(&s, &d, qw.data)?,
        &l0_attn_gpu,
        &l0_q_raw,
        h,
        nh * hd,
        n,
        gemm_fmt(l0_q_info.ggml_type),
    )?;
    let l0_q_raw_gpu = download(&s, &l0_q_raw, n * nh * hd)?;
    let l0_q_raw_cpu = l0_attn_cpu
        .chunks_exact(h)
        .flat_map(|row| {
            let mut z = vec![0.; nh * hd];
            matmul(&mut z, &qw, row);
            z
        })
        .collect::<Vec<_>>();
    report(
        "layer0_q_raw_gemm",
        &format!("{:?}", l0_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l0_q_raw_gpu,
        &l0_q_raw_cpu,
    );
    report_q_metadata(&r, &p, "blk.1.attn_q.weight", h, nh * hd);

    // Diagnostic-only cross-factor matrix. Each case owns its output buffer.
    let l0_q_weight = upload_raw(&s, &d, qw.data)?;
    let l1_q_weight = upload_raw(&s, &d, l1qw.data)?;
    let x0_dev = upload(&s, &d, &l0_attn_cpu)?;
    let x1_dev = upload(&s, &d, &layer1_cpu)?;
    let l0_q_fmt = gemm_fmt(l0_q_info.ggml_type);
    let l1_q_info = r.get_tensor("blk.1.attn_q.weight").unwrap();
    let l1_q_fmt = gemm_fmt(l1_q_info.ggml_type);
    for (weight_name, input_name, weight_dev, input_dev, weight, input_cpu, fmt) in [
        (
            "W0",
            "X0",
            &l0_q_weight,
            &x0_dev,
            &qw,
            &l0_attn_cpu,
            l0_q_fmt,
        ),
        (
            "W0",
            "X1(normalized CPU reference)",
            &l0_q_weight,
            &x1_dev,
            &qw,
            &layer1_cpu,
            l0_q_fmt,
        ),
        (
            "W1",
            "X0",
            &l1_q_weight,
            &x0_dev,
            &l1qw,
            &l0_attn_cpu,
            l1_q_fmt,
        ),
        (
            "W1",
            "X1(normalized CPU reference)",
            &l1_q_weight,
            &x1_dev,
            &l1qw,
            &layer1_cpu,
            l1_q_fmt,
        ),
    ] {
        let output = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
        g.gemm(&s, weight_dev, input_dev, &output, h, nh * hd, n, fmt)?;
        let gpu = download(&s, &output, n * nh * hd)?;
        let cpu = input_cpu
            .chunks_exact(h)
            .flat_map(|row| {
                let mut z = vec![0.; nh * hd];
                matmul(&mut z, weight, row);
                z
            })
            .collect::<Vec<_>>();
        report_raw_q_cross(
            weight_name,
            input_name,
            &format!("[{n},{nh},{hd}]"),
            n,
            &gpu,
            &cpu,
        );
    }
    let l1q_raw = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l1k_raw = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l1v = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let qwd = upload_raw(&s, &d, l1qw.data)?;
    let kwd = upload_raw(&s, &d, l1kw.data)?;
    let vwd = upload_raw(&s, &d, l1vw.data)?;
    let qfmt = gemm_fmt(r.get_tensor("blk.1.attn_q.weight").unwrap().ggml_type);
    let kfmt = gemm_fmt(r.get_tensor("blk.1.attn_k.weight").unwrap().ggml_type);
    let vfmt = gemm_fmt(r.get_tensor("blk.1.attn_v.weight").unwrap().ggml_type);
    g.gemm(&s, &qwd, &l1_out, &l1q_raw, h, nh * hd, n, qfmt)?;
    g.gemm(&s, &kwd, &l1_out, &l1k_raw, h, nkv * hd, n, kfmt)?;
    g.gemm(&s, &vwd, &l1_out, &l1v, h, nkv * hd, n, vfmt)?;

    let q_raw_cpu = layer1_cpu
        .chunks_exact(h)
        .flat_map(|row| {
            let mut z = vec![0.; nh * hd];
            matmul(&mut z, &l1qw, row);
            z
        })
        .collect::<Vec<_>>();
    let k_raw_cpu = layer1_cpu
        .chunks_exact(h)
        .flat_map(|row| {
            let mut z = vec![0.; nkv * hd];
            matmul(&mut z, &l1kw, row);
            z
        })
        .collect::<Vec<_>>();
    let v_raw_cpu = layer1_cpu
        .chunks_exact(h)
        .flat_map(|row| {
            let mut z = vec![0.; nkv * hd];
            matmul(&mut z, &l1vw, row);
            z
        })
        .collect::<Vec<_>>();
    let q_raw_gpu = download(&s, &l1q_raw, n * nh * hd)?;
    let k_raw_gpu = download(&s, &l1k_raw, n * nkv * hd)?;
    let v_raw_gpu = download(&s, &l1v, n * nkv * hd)?;
    report(
        "layer1_q_raw_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.1.attn_q.weight").unwrap().ggml_type
        ),
        &format!("[{n},{nh},{hd}]"),
        n,
        &q_raw_gpu,
        &q_raw_cpu,
    );
    report(
        "layer1_k_raw_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.1.attn_k.weight").unwrap().ggml_type
        ),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &k_raw_gpu,
        &k_raw_cpu,
    );
    report(
        "layer1_v_raw_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.1.attn_v.weight").unwrap().ggml_type
        ),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &v_raw_gpu,
        &v_raw_cpu,
    );

    let qd1 = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let kd1 = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let qzero = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let kzero = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l1q_raw,
        &qzero,
        &upload(&s, &d, &l1qn)?,
        &qd1,
        &qd1,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        None,
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l1k_raw,
        &kzero,
        &upload(&s, &d, &l1kn)?,
        &kd1,
        &kd1,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        None,
        n * nkv,
        nkv,
    )?;
    let qg = download(&s, &qd1, n * nh * hd)?;
    let kg = download(&s, &kd1, n * nkv * hd)?;
    let vg = download(&s, &l1v, n * nkv * hd)?;
    let mut qc = Vec::new();
    let mut kc = Vec::new();
    let mut vc = Vec::new();
    for (pos, row) in layer1_cpu.chunks_exact(h).enumerate() {
        let mut zq = vec![0.; nh * hd];
        matmul(&mut zq, &l1qw, row);
        for j in 0..nh {
            qc.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(&zq[j * hd..(j + 1) * hd], &l1qn, eps),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        let mut zk = vec![0.; nkv * hd];
        matmul(&mut zk, &l1kw, row);
        for j in 0..nkv {
            kc.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(&zk[j * hd..(j + 1) * hd], &l1kn, eps),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        let mut zv = vec![0.; nkv * hd];
        matmul(&mut zv, &l1vw, row);
        vc.extend(zv);
    }
    report_q_norm_rope_tokens(n, nh, hd, &qg, &qc);

    report(
        "layer1_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &qg,
        &qc,
    );
    report(
        "layer1_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &kg,
        &kc,
    );
    report(
        "layer1_v_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.1.attn_v.weight").unwrap().ggml_type
        ),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &vg,
        &vc,
    );
    let bt1 = upload(&s, &d, &[0.])?;
    let pool1 = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    kv.append_kv(&s, &layout, &pool1, &kd1, &l1v, &bt1, 0, n)?;
    let att1 = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(&s, &qd1, &pool1, &bt1, &att1, nh, nkv, hd, 64, n, 0)?;
    let att1g = download(&s, &att1, n * nh * hd)?;
    let mut att1c = Vec::new();
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let z = t * 2 * nkv * hd;
            ph[z..z + nkv * hd].copy_from_slice(&kc[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[z + nkv * hd..z + 2 * nkv * hd]
                .copy_from_slice(&vc[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        att1c.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &qc[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer1_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &att1g,
        &att1c,
    );
    let l1od = upload_raw(&s, &d, l1wo.data)?;
    let l1h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &l1od,
        &att1,
        &l1h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.1.attn_output.weight").unwrap().ggml_type),
        Some(&l1_in),
    )?;
    let l1hc = down_cpu
        .chunks_exact(h)
        .zip(att1c.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l1wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let l1hg = download(&s, &l1h, n * h)?;
    report(
        "layer1_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l1hg,
        &l1hc,
    );
    let l1fd = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l1h,
        &zero,
        &upload(&s, &d, &l1fn)?,
        &l1fd,
        &l1fd,
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
    let l1fc = l1hc
        .chunks_exact(h)
        .flat_map(|z| rms_norm(z, &l1fn, eps))
        .collect::<Vec<_>>();
    let l1fg = download(&s, &l1fd, n * h)?;
    report(
        "layer1_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l1fg,
        &l1fc,
    );
    let i1 = r.get_tensor("blk.1.ffn_gate.weight").unwrap().dims[1] as usize;
    let g1 = tensor(&r, &p, "blk.1.ffn_gate.weight");
    let u1 = tensor(&r, &p, "blk.1.ffn_up.weight");
    let d1 = tensor(&r, &p, "blk.1.ffn_down.weight");
    let gd = upload_raw(&s, &d, g1.data)?;
    let ud = upload_raw(&s, &d, u1.data)?;
    let go = DeviceBuffer::alloc(d.clone(), n * i1 * 4)?;
    let uo = DeviceBuffer::alloc(d.clone(), n * i1 * 4)?;
    g.gemm(
        &s,
        &gd,
        &l1fd,
        &go,
        h,
        i1,
        n,
        gemm_fmt(r.get_tensor("blk.1.ffn_gate.weight").unwrap().ggml_type),
    )?;
    g.gemm(
        &s,
        &ud,
        &l1fd,
        &uo,
        h,
        i1,
        n,
        gemm_fmt(r.get_tensor("blk.1.ffn_up.weight").unwrap().ggml_type),
    )?;
    let gg = download(&s, &go, n * i1)?;
    let ug = download(&s, &uo, n * i1)?;
    let gc = norm_cpu_for_ffn(&l1fc, &g1, i1, h);
    let uc = norm_cpu_for_ffn(&l1fc, &u1, i1, h);
    report(
        "layer1_gate_gemm",
        "F32",
        &format!("[{n},{i1}]"),
        n,
        &gg,
        &gc,
    );
    report("layer1_up_gemm", "F32", &format!("[{n},{i1}]"), n, &ug, &uc);
    let swc = (0..n)
        .flat_map(|r| {
            engine_core::forward_cpu::swiglu(&gc[r * i1..(r + 1) * i1], &uc[r * i1..(r + 1) * i1])
        })
        .collect::<Vec<_>>();
    let swo = DeviceBuffer::alloc(d.clone(), n * i1 * 4)?;
    let zero_i1 = upload(&s, &d, &vec![0.; n * i1])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &go,
        &zero_i1,
        &zero_i1,
        &uo,
        &swo,
        eps,
        i1,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let swg = download(&s, &swo, n * i1)?;
    report(
        "layer1_swiglu",
        "F32",
        &format!("[{n},{i1}]"),
        n,
        &swg,
        &swc,
    );
    let dd = upload_raw(&s, &d, d1.data)?;
    let out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &dd,
        &swo,
        &out,
        i1,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.1.ffn_down.weight").unwrap().ggml_type),
        Some(&l1h),
    )?;
    let mut outc = Vec::new();
    for r in 0..n {
        let mut z = vec![0.; h];
        matmul(&mut z, &d1, &swc[r * i1..(r + 1) * i1]);
        outc.extend(l1hc[r * h..(r + 1) * h].iter().zip(z).map(|(&a, b)| a + b));
    }
    let outg = download(&s, &out, n * h)?;
    report(
        "layer1_down_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &outg,
        &outc,
    );

    // Continue from the real layer-1 post-down residual into block 2.
    // Keep this deliberately narrow: it isolates the next transition before
    // adding another attention/FFN composition on top of it.
    let l2w = vec32(&p, "blk.2.attn_norm.weight");
    let l2qw = tensor(&r, &p, "blk.2.attn_q.weight");
    let l2kw = tensor(&r, &p, "blk.2.attn_k.weight");
    let l2vw = tensor(&r, &p, "blk.2.attn_v.weight");
    assert_eq!(l2w.len(), h);
    report_q_metadata(&r, &p, "blk.2.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.2.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.2.attn_v.weight", h, nkv * hd);

    let l2_in = upload(&s, &d, &outg)?;
    let l2_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l2w_dev = upload(&s, &d, &l2w)?;
    let l2_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l2_in,
        &l2_zero,
        &l2w_dev,
        &l2_norm,
        &l2_norm,
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
    let l2_norm_gpu = download(&s, &l2_norm, n * h)?;
    let l2_norm_cpu = outc
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l2w, eps))
        .collect::<Vec<_>>();
    report(
        "layer2_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l2_norm_gpu,
        &l2_norm_cpu,
    );

    let l2_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l2_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l2_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l2_qd = upload_raw(&s, &d, l2qw.data)?;
    let l2_kd = upload_raw(&s, &d, l2kw.data)?;
    let l2_vd = upload_raw(&s, &d, l2vw.data)?;
    let l2_q_info = r.get_tensor("blk.2.attn_q.weight").unwrap();
    let l2_k_info = r.get_tensor("blk.2.attn_k.weight").unwrap();
    let l2_v_info = r.get_tensor("blk.2.attn_v.weight").unwrap();
    g.gemm(
        &s,
        &l2_qd,
        &l2_norm,
        &l2_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l2_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l2_kd,
        &l2_norm,
        &l2_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l2_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l2_vd,
        &l2_norm,
        &l2_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l2_v_info.ggml_type),
    )?;
    let l2_q_gpu = download(&s, &l2_q, n * nh * hd)?;
    let l2_k_gpu = download(&s, &l2_k, n * nkv * hd)?;
    let l2_v_gpu = download(&s, &l2_v, n * nkv * hd)?;
    let l2_q_cpu = norm_cpu_for_ffn(&l2_norm_cpu, &l2qw, nh * hd, h);
    let l2_k_cpu = norm_cpu_for_ffn(&l2_norm_cpu, &l2kw, nkv * hd, h);
    let l2_v_cpu = norm_cpu_for_ffn(&l2_norm_cpu, &l2vw, nkv * hd, h);
    report(
        "layer2_q_raw_gemm",
        &format!("{:?}", l2_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l2_q_gpu,
        &l2_q_cpu,
    );
    report(
        "layer2_k_raw_gemm",
        &format!("{:?}", l2_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l2_k_gpu,
        &l2_k_cpu,
    );
    report(
        "layer2_v_raw_gemm",
        &format!("{:?}", l2_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l2_v_gpu,
        &l2_v_cpu,
    );

    // Finish block 2 from the raw Q/K/V products.  All references below are
    // computed from the independent CPU residual, never from a downloaded GPU
    // intermediate.
    let l2_qn = vec32(&p, "blk.2.attn_q_norm.weight");
    let l2_kn = vec32(&p, "blk.2.attn_k_norm.weight");
    let l2_fn = vec32(&p, "blk.2.ffn_norm.weight");
    let l2_wo = tensor(&r, &p, "blk.2.attn_output.weight");
    let l2_gw = tensor(&r, &p, "blk.2.ffn_gate.weight");
    let l2_uw = tensor(&r, &p, "blk.2.ffn_up.weight");
    let l2_dw = tensor(&r, &p, "blk.2.ffn_down.weight");
    let l2_qn_g = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l2_kn_g = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l2_q,
        &zq,
        &upload(&s, &d, &l2_qn)?,
        &l2_qn_g,
        &l2_qn_g,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        None,
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l2_k,
        &zk,
        &upload(&s, &d, &l2_kn)?,
        &l2_kn_g,
        &l2_kn_g,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        None,
        n * nkv,
        nkv,
    )?;
    let q2g = download(&s, &l2_qn_g, n * nh * hd)?;
    let k2g = download(&s, &l2_kn_g, n * nkv * hd)?;
    let mut q2c = Vec::with_capacity(n * nh * hd);
    let mut k2c = Vec::with_capacity(n * nkv * hd);
    for (pos, row) in l2_norm_cpu.chunks_exact(h).enumerate() {
        let mut q = vec![0.; nh * hd];
        matmul(&mut q, &l2qw, row);
        for j in 0..nh {
            q2c.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(&q[j * hd..(j + 1) * hd], &l2_qn, eps),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        let mut k = vec![0.; nkv * hd];
        matmul(&mut k, &l2kw, row);
        for j in 0..nkv {
            k2c.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(&k[j * hd..(j + 1) * hd], &l2_kn, eps),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer2_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &q2g,
        &q2c,
    );
    report(
        "layer2_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &k2g,
        &k2c,
    );
    report(
        "layer2_v_gemm",
        &format!("{:?}", l2_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l2_v_gpu,
        &l2_v_cpu,
    );

    let bt2 = upload(&s, &d, &[0.])?;
    let pool2 = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let kd2 = upload(&s, &d, &k2g)?;
    let vd2 = upload(&s, &d, &l2_v_gpu)?;
    kv.append_kv(&s, &layout, &pool2, &kd2, &vd2, &bt2, 0, n)?;
    let qd2 = upload(&s, &d, &q2g)?;
    let att2 = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(&s, &qd2, &pool2, &bt2, &att2, nh, nkv, hd, 64, n, 0)?;
    let att2g = download(&s, &att2, n * nh * hd)?;
    let mut att2c = Vec::new();
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd].copy_from_slice(&k2c[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l2_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        att2c.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &q2c[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer2_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &att2g,
        &att2c,
    );
    let o2d = upload_raw(&s, &d, l2_wo.data)?;
    let h2 = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &o2d,
        &att2,
        &h2,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.2.attn_output.weight").unwrap().ggml_type),
        Some(&l2_in),
    )?;
    let h2g = download(&s, &h2, n * h)?;
    let h2c = outc
        .chunks_exact(h)
        .zip(att2c.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l2_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer2_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &h2g,
        &h2c,
    );
    let f2g = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &h2,
        &l2_zero,
        &upload(&s, &d, &l2_fn)?,
        &f2g,
        &f2g,
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
    let f2gpu = download(&s, &f2g, n * h)?;
    let f2cpu = h2c
        .chunks_exact(h)
        .flat_map(|x| rms_norm(x, &l2_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer2_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &f2gpu,
        &f2cpu,
    );
    let i2 = r.get_tensor("blk.2.ffn_gate.weight").unwrap().dims[1] as usize;
    let gd = upload_raw(&s, &d, l2_gw.data)?;
    let ud = upload_raw(&s, &d, l2_uw.data)?;
    let go = DeviceBuffer::alloc(d.clone(), n * i2 * 4)?;
    let uo = DeviceBuffer::alloc(d.clone(), n * i2 * 4)?;
    let nd = upload(&s, &d, &f2cpu)?;
    g.gemm(
        &s,
        &gd,
        &nd,
        &go,
        h,
        i2,
        n,
        gemm_fmt(r.get_tensor("blk.2.ffn_gate.weight").unwrap().ggml_type),
    )?;
    g.gemm(
        &s,
        &ud,
        &nd,
        &uo,
        h,
        i2,
        n,
        gemm_fmt(r.get_tensor("blk.2.ffn_up.weight").unwrap().ggml_type),
    )?;
    let gog = download(&s, &go, n * i2)?;
    let uog = download(&s, &uo, n * i2)?;
    let goc = norm_cpu_for_ffn(&f2cpu, &l2_gw, i2, h);
    let uoc = norm_cpu_for_ffn(&f2cpu, &l2_uw, i2, h);
    report(
        "layer2_gate_gemm",
        "F32",
        &format!("[{n},{i2}]"),
        n,
        &gog,
        &goc,
    );
    report(
        "layer2_up_gemm",
        "F32",
        &format!("[{n},{i2}]"),
        n,
        &uog,
        &uoc,
    );
    let swc = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &goc[row * i2..(row + 1) * i2],
                &uoc[row * i2..(row + 1) * i2],
            )
        })
        .collect::<Vec<_>>();
    let swo = DeviceBuffer::alloc(d.clone(), n * i2 * 4)?;
    let zi = upload(&s, &d, &vec![0.; n * i2])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &go,
        &zi,
        &zi,
        &uo,
        &swo,
        eps,
        i2,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let swg = download(&s, &swo, n * i2)?;
    report(
        "layer2_swiglu",
        "F32",
        &format!("[{n},{i2}]"),
        n,
        &swg,
        &swc,
    );
    let dd = upload_raw(&s, &d, l2_dw.data)?;
    let finalg = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &dd,
        &swo,
        &finalg,
        i2,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.2.ffn_down.weight").unwrap().ggml_type),
        Some(&h2),
    )?;
    let final_gpu = download(&s, &finalg, n * h)?;
    let mut final_cpu = Vec::new();
    for row in 0..n {
        let mut z = vec![0.; h];
        matmul(&mut z, &l2_dw, &swc[row * i2..(row + 1) * i2]);
        final_cpu.extend(
            h2c[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b),
        );
    }
    report(
        "layer2_down_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &final_gpu,
        &final_cpu,
    );

    // Continue from the real layer-2 post-down residual into block 3.  Keep
    // the CPU path independent so the first failing layer-3 stage is useful.
    let l3w = vec32(&p, "blk.3.attn_norm.weight");
    let l3qw = tensor(&r, &p, "blk.3.attn_q.weight");
    let l3kw = tensor(&r, &p, "blk.3.attn_k.weight");
    let l3vw = tensor(&r, &p, "blk.3.attn_v.weight");
    assert_eq!(l3w.len(), h);
    report_q_metadata(&r, &p, "blk.3.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.3.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.3.attn_v.weight", h, nkv * hd);

    let l3_in = upload(&s, &d, &final_gpu)?;
    let l3_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l3_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l3_in,
        &l3_zero,
        &upload(&s, &d, &l3w)?,
        &l3_norm,
        &l3_norm,
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
    let l3_norm_gpu = download(&s, &l3_norm, n * h)?;
    let l3_norm_cpu = final_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l3w, eps))
        .collect::<Vec<_>>();
    report(
        "layer3_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l3_norm_gpu,
        &l3_norm_cpu,
    );

    let l3_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l3_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l3_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l3_q_info = r.get_tensor("blk.3.attn_q.weight").unwrap();
    let l3_k_info = r.get_tensor("blk.3.attn_k.weight").unwrap();
    let l3_v_info = r.get_tensor("blk.3.attn_v.weight").unwrap();
    g.gemm(
        &s,
        &upload_raw(&s, &d, l3qw.data)?,
        &l3_norm,
        &l3_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l3_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l3kw.data)?,
        &l3_norm,
        &l3_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l3_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l3vw.data)?,
        &l3_norm,
        &l3_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l3_v_info.ggml_type),
    )?;
    let l3_q_gpu = download(&s, &l3_q, n * nh * hd)?;
    let l3_k_gpu = download(&s, &l3_k, n * nkv * hd)?;
    let l3_v_gpu = download(&s, &l3_v, n * nkv * hd)?;
    let l3_q_cpu = norm_cpu_for_ffn(&l3_norm_cpu, &l3qw, nh * hd, h);
    let l3_k_cpu = norm_cpu_for_ffn(&l3_norm_cpu, &l3kw, nkv * hd, h);
    let l3_v_cpu = norm_cpu_for_ffn(&l3_norm_cpu, &l3vw, nkv * hd, h);
    report(
        "layer3_q_raw_gemm",
        &format!("{:?}", l3_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l3_q_gpu,
        &l3_q_cpu,
    );
    report(
        "layer3_k_raw_gemm",
        &format!("{:?}", l3_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l3_k_gpu,
        &l3_k_cpu,
    );
    report(
        "layer3_v_raw_gemm",
        &format!("{:?}", l3_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l3_v_gpu,
        &l3_v_cpu,
    );

    let l3_qn = vec32(&p, "blk.3.attn_q_norm.weight");
    let l3_kn = vec32(&p, "blk.3.attn_k_norm.weight");
    let l3_qn_g = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l3_kn_g = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l3_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l3_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l3_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l3_q,
        &l3_zq,
        &upload(&s, &d, &l3_qn)?,
        &l3_qn_g,
        &l3_qn_g,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l3_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l3_k,
        &l3_zk,
        &upload(&s, &d, &l3_kn)?,
        &l3_kn_g,
        &l3_kn_g,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l3_pos),
        n * nkv,
        nkv,
    )?;
    let l3_q_norm_rope_gpu = download(&s, &l3_qn_g, n * nh * hd)?;
    let l3_k_norm_rope_gpu = download(&s, &l3_kn_g, n * nkv * hd)?;
    let mut l3_q_norm_rope_cpu = Vec::with_capacity(n * nh * hd);
    let mut l3_k_norm_rope_cpu = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            l3_q_norm_rope_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l3_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l3_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            l3_k_norm_rope_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l3_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l3_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer3_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l3_q_norm_rope_gpu,
        &l3_q_norm_rope_cpu,
    );
    report(
        "layer3_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l3_k_norm_rope_gpu,
        &l3_k_norm_rope_cpu,
    );

    report(
        "layer3_v_gemm",
        &format!("{:?}", l3_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l3_v_gpu,
        &l3_v_cpu,
    );
    let l3_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l3_bt = upload(&s, &d, &[0.])?;
    let l3_kd = upload(&s, &d, &l3_k_norm_rope_gpu)?;
    let l3_vd = upload(&s, &d, &l3_v_gpu)?;
    kv.append_kv(&s, &layout, &l3_pool, &l3_kd, &l3_vd, &l3_bt, 0, n)?;
    let l3_qd = upload(&s, &d, &l3_q_norm_rope_gpu)?;
    let l3_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(&s, &l3_qd, &l3_pool, &l3_bt, &l3_att, nh, nkv, hd, 64, n, 0)?;
    let l3_att_gpu = download(&s, &l3_att, n * nh * hd)?;
    let mut l3_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd]
                .copy_from_slice(&l3_k_norm_rope_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l3_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l3_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &l3_q_norm_rope_cpu[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer3_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l3_att_gpu,
        &l3_att_cpu,
    );

    let l3_wo = tensor(&r, &p, "blk.3.attn_output.weight");
    let l3_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l3_wo.data)?,
        &l3_att,
        &l3_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.3.attn_output.weight").unwrap().ggml_type),
        Some(&l3_in),
    )?;
    let l3_h_gpu = download(&s, &l3_h, n * h)?;
    let l3_h_cpu = final_cpu
        .chunks_exact(h)
        .zip(l3_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l3_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer3_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l3_h_gpu,
        &l3_h_cpu,
    );

    let l3_fn = vec32(&p, "blk.3.ffn_norm.weight");
    let l3_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l3_h,
        &l3_zero,
        &upload(&s, &d, &l3_fn)?,
        &l3_f,
        &l3_f,
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
    let l3_f_gpu = download(&s, &l3_f, n * h)?;
    let l3_f_cpu = l3_h_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l3_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer3_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l3_f_gpu,
        &l3_f_cpu,
    );

    let i3 = r.get_tensor("blk.3.ffn_gate.weight").unwrap().dims[1] as usize;
    let l3_gw = tensor(&r, &p, "blk.3.ffn_gate.weight");
    let l3_uw = tensor(&r, &p, "blk.3.ffn_up.weight");
    let l3_dw = tensor(&r, &p, "blk.3.ffn_down.weight");
    let gd = upload_raw(&s, &d, l3_gw.data)?;
    let ud = upload_raw(&s, &d, l3_uw.data)?;
    let go = DeviceBuffer::alloc(d.clone(), n * i3 * 4)?;
    let uo = DeviceBuffer::alloc(d.clone(), n * i3 * 4)?;
    for (w, o, name) in [(&gd, &go, "layer3_gate_gemm"), (&ud, &uo, "layer3_up_gemm")] {
        g.gemm(
            &s,
            w,
            &l3_f,
            o,
            h,
            i3,
            n,
            gemm_fmt(
                r.get_tensor(if name.ends_with("gate_gemm") {
                    "blk.3.ffn_gate.weight"
                } else {
                    "blk.3.ffn_up.weight"
                })
                .unwrap()
                .ggml_type,
            ),
        )?;
        let gpu = download(&s, o, n * i3)?;
        let cpu = norm_cpu_for_ffn(
            &l3_f_cpu,
            if name.ends_with("gate_gemm") {
                &l3_gw
            } else {
                &l3_uw
            },
            i3,
            h,
        );
        report(name, "Q4_K", &format!("[{n},{i3}]"), n, &gpu, &cpu);
    }
    let gate_cpu = norm_cpu_for_ffn(&l3_f_cpu, &l3_gw, i3, h);
    let up_cpu = norm_cpu_for_ffn(&l3_f_cpu, &l3_uw, i3, h);
    let sw_cpu = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &gate_cpu[row * i3..(row + 1) * i3],
                &up_cpu[row * i3..(row + 1) * i3],
            )
        })
        .collect::<Vec<_>>();
    let sw = DeviceBuffer::alloc(d.clone(), n * i3 * 4)?;
    let zi = upload(&s, &d, &vec![0.; n * i3])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &go,
        &zi,
        &zi,
        &uo,
        &sw,
        eps,
        i3,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let sw_gpu = download(&s, &sw, n * i3)?;
    report(
        "layer3_swiglu",
        "F32",
        &format!("[{n},{i3}]"),
        n,
        &sw_gpu,
        &sw_cpu,
    );

    let out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l3_dw.data)?,
        &sw,
        &out,
        i3,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.3.ffn_down.weight").unwrap().ggml_type),
        Some(&l3_h),
    )?;
    let out_gpu = download(&s, &out, n * h)?;
    let mut out_cpu = Vec::with_capacity(n * h);
    for row in 0..n {
        let mut z = vec![0.; h];
        matmul(&mut z, &l3_dw, &sw_cpu[row * i3..(row + 1) * i3]);
        out_cpu.extend(
            l3_h_cpu[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b),
        );
    }
    report(
        "layer3_down_projection_residual",
        "Q6_K",
        &format!("[{n},{h}]"),
        n,
        &out_gpu,
        &out_cpu,
    );
    // Continue from the real layer-3 post-down residual into block 4.  Keep
    // the CPU path independent so the first failing layer-3 stage is useful.
    let l4w = vec32(&p, "blk.4.attn_norm.weight");
    let l4qw = tensor(&r, &p, "blk.4.attn_q.weight");
    let l4kw = tensor(&r, &p, "blk.4.attn_k.weight");
    let l4vw = tensor(&r, &p, "blk.4.attn_v.weight");
    assert_eq!(l4w.len(), h);
    report_q_metadata(&r, &p, "blk.4.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.4.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.4.attn_v.weight", h, nkv * hd);

    let l4_in = upload(&s, &d, &out_gpu)?;
    let l4_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l4_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l4_in,
        &l4_zero,
        &upload(&s, &d, &l4w)?,
        &l4_norm,
        &l4_norm,
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
    let l4_norm_gpu = download(&s, &l4_norm, n * h)?;
    let l4_norm_cpu = out_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l4w, eps))
        .collect::<Vec<_>>();
    report(
        "layer4_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l4_norm_gpu,
        &l4_norm_cpu,
    );

    let l4_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l4_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l4_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l4_q_info = r.get_tensor("blk.4.attn_q.weight").unwrap();
    let l4_k_info = r.get_tensor("blk.4.attn_k.weight").unwrap();
    let l4_v_info = r.get_tensor("blk.4.attn_v.weight").unwrap();
    g.gemm(
        &s,
        &upload_raw(&s, &d, l4qw.data)?,
        &l4_norm,
        &l4_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l4_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l4kw.data)?,
        &l4_norm,
        &l4_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l4_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l4vw.data)?,
        &l4_norm,
        &l4_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l4_v_info.ggml_type),
    )?;
    let l4_q_gpu = download(&s, &l4_q, n * nh * hd)?;
    let l4_k_gpu = download(&s, &l4_k, n * nkv * hd)?;
    let l4_v_gpu = download(&s, &l4_v, n * nkv * hd)?;
    let l4_q_cpu = norm_cpu_for_ffn(&l4_norm_cpu, &l4qw, nh * hd, h);
    let l4_k_cpu = norm_cpu_for_ffn(&l4_norm_cpu, &l4kw, nkv * hd, h);
    let l4_v_cpu = norm_cpu_for_ffn(&l4_norm_cpu, &l4vw, nkv * hd, h);
    report(
        "layer4_q_raw_gemm",
        &format!("{:?}", l4_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l4_q_gpu,
        &l4_q_cpu,
    );
    report(
        "layer4_k_raw_gemm",
        &format!("{:?}", l4_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l4_k_gpu,
        &l4_k_cpu,
    );
    report(
        "layer4_v_raw_gemm",
        &format!("{:?}", l4_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l4_v_gpu,
        &l4_v_cpu,
    );

    let l4_qn = vec32(&p, "blk.4.attn_q_norm.weight");
    let l4_kn = vec32(&p, "blk.4.attn_k_norm.weight");
    let l4_qn_g = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l4_kn_g = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l4_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l4_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l4_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l4_q,
        &l4_zq,
        &upload(&s, &d, &l4_qn)?,
        &l4_qn_g,
        &l4_qn_g,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l4_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l4_k,
        &l4_zk,
        &upload(&s, &d, &l4_kn)?,
        &l4_kn_g,
        &l4_kn_g,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l4_pos),
        n * nkv,
        nkv,
    )?;
    let l4_q_norm_rope_gpu = download(&s, &l4_qn_g, n * nh * hd)?;
    let l4_k_norm_rope_gpu = download(&s, &l4_kn_g, n * nkv * hd)?;
    let mut l4_q_norm_rope_cpu = Vec::with_capacity(n * nh * hd);
    let mut l4_k_norm_rope_cpu = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            l4_q_norm_rope_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l4_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l4_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            l4_k_norm_rope_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l4_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l4_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer4_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l4_q_norm_rope_gpu,
        &l4_q_norm_rope_cpu,
    );
    report(
        "layer4_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l4_k_norm_rope_gpu,
        &l4_k_norm_rope_cpu,
    );

    report(
        "layer4_v_gemm",
        &format!("{:?}", l4_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l4_v_gpu,
        &l4_v_cpu,
    );
    let l4_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l4_bt = upload(&s, &d, &[0.])?;
    let l4_kd = upload(&s, &d, &l4_k_norm_rope_gpu)?;
    let l4_vd = upload(&s, &d, &l4_v_gpu)?;
    kv.append_kv(&s, &layout, &l4_pool, &l4_kd, &l4_vd, &l4_bt, 0, n)?;
    let l4_qd = upload(&s, &d, &l4_q_norm_rope_gpu)?;
    let l4_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(&s, &l4_qd, &l4_pool, &l4_bt, &l4_att, nh, nkv, hd, 64, n, 0)?;
    let l4_att_gpu = download(&s, &l4_att, n * nh * hd)?;
    let mut l4_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd]
                .copy_from_slice(&l4_k_norm_rope_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l4_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l4_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &l4_q_norm_rope_cpu[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer4_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l4_att_gpu,
        &l4_att_cpu,
    );

    let l4_wo = tensor(&r, &p, "blk.4.attn_output.weight");
    let l4_wo_info = r.get_tensor("blk.4.attn_output.weight").unwrap();
    let l4_wo_dev = upload_raw(&s, &d, l4_wo.data)?;

    // Minimal A/B decomposition of the layer-4 output projection residual.
    // Keep every GPU result in an independent buffer and download after the
    // stream is synchronized so each report isolates one possible mismatch.
    let l4_projection = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm(
        &s,
        &l4_wo_dev,
        &l4_att,
        &l4_projection,
        nh * hd,
        h,
        n,
        gemm_fmt(l4_wo_info.ggml_type),
    )?;
    let l4_projection_gpu = download(&s, &l4_projection, n * h)?;
    let l4_projection_cpu = l4_att_cpu
        .chunks_exact(nh * hd)
        .flat_map(|a| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l4_wo, a);
            z
        })
        .collect::<Vec<_>>();
    report(
        "layer4_output_projection",
        &format!("{:?}", l4_wo_info.ggml_type),
        &format!("[{n},{h}]"),
        n,
        &l4_projection_gpu,
        &l4_projection_cpu,
    );

    let l4_residual_gpu = download(&s, &l4_in, n * h)?;
    report(
        "layer4_residual_input",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l4_residual_gpu,
        &out_cpu,
    );

    let l4_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &l4_wo_dev,
        &l4_att,
        &l4_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.4.attn_output.weight").unwrap().ggml_type),
        Some(&l4_in),
    )?;
    let l4_h_gpu = download(&s, &l4_h, n * h)?;
    let l4_projection_plus_residual_gpu = l4_projection_gpu
        .iter()
        .zip(&l4_residual_gpu)
        .map(|(&projection, &residual)| projection + residual)
        .collect::<Vec<_>>();
    report(
        "layer4_output_projection_residual_ab",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l4_h_gpu,
        &l4_projection_plus_residual_gpu,
    );
    let l4_h_cpu = out_cpu
        .chunks_exact(h)
        .zip(l4_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l4_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer4_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l4_h_gpu,
        &l4_h_cpu,
    );

    let l4_fn = vec32(&p, "blk.4.ffn_norm.weight");
    let l4_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l4_h,
        &l4_zero,
        &upload(&s, &d, &l4_fn)?,
        &l4_f,
        &l4_f,
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
    let l4_f_gpu = download(&s, &l4_f, n * h)?;
    let l4_f_cpu = l4_h_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l4_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer4_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l4_f_gpu,
        &l4_f_cpu,
    );

    let i3 = r.get_tensor("blk.4.ffn_gate.weight").unwrap().dims[1] as usize;
    let l4_gw = tensor(&r, &p, "blk.4.ffn_gate.weight");
    let l4_uw = tensor(&r, &p, "blk.4.ffn_up.weight");
    let l4_dw = tensor(&r, &p, "blk.4.ffn_down.weight");
    let gd = upload_raw(&s, &d, l4_gw.data)?;
    let ud = upload_raw(&s, &d, l4_uw.data)?;
    let go = DeviceBuffer::alloc(d.clone(), n * i3 * 4)?;
    let uo = DeviceBuffer::alloc(d.clone(), n * i3 * 4)?;
    for (w, o, name) in [(&gd, &go, "layer4_gate_gemm"), (&ud, &uo, "layer4_up_gemm")] {
        g.gemm(
            &s,
            w,
            &l4_f,
            o,
            h,
            i3,
            n,
            gemm_fmt(
                r.get_tensor(if name.ends_with("gate_gemm") {
                    "blk.4.ffn_gate.weight"
                } else {
                    "blk.4.ffn_up.weight"
                })
                .unwrap()
                .ggml_type,
            ),
        )?;
        let gpu = download(&s, o, n * i3)?;
        let cpu = norm_cpu_for_ffn(
            &l4_f_cpu,
            if name.ends_with("gate_gemm") {
                &l4_gw
            } else {
                &l4_uw
            },
            i3,
            h,
        );
        report(name, "Q4_K", &format!("[{n},{i3}]"), n, &gpu, &cpu);
    }
    let gate_cpu = norm_cpu_for_ffn(&l4_f_cpu, &l4_gw, i3, h);
    let up_cpu = norm_cpu_for_ffn(&l4_f_cpu, &l4_uw, i3, h);
    let sw_cpu = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &gate_cpu[row * i3..(row + 1) * i3],
                &up_cpu[row * i3..(row + 1) * i3],
            )
        })
        .collect::<Vec<_>>();
    let sw = DeviceBuffer::alloc(d.clone(), n * i3 * 4)?;
    let zi = upload(&s, &d, &vec![0.; n * i3])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &go,
        &zi,
        &zi,
        &uo,
        &sw,
        eps,
        i3,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let sw_gpu = download(&s, &sw, n * i3)?;
    report(
        "layer4_swiglu",
        "F32",
        &format!("[{n},{i3}]"),
        n,
        &sw_gpu,
        &sw_cpu,
    );

    let out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l4_dw.data)?,
        &sw,
        &out,
        i3,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.4.ffn_down.weight").unwrap().ggml_type),
        Some(&l4_h),
    )?;
    let out_gpu = download(&s, &out, n * h)?;
    let mut out_cpu = Vec::with_capacity(n * h);
    for row in 0..n {
        let mut z = vec![0.; h];
        matmul(&mut z, &l4_dw, &sw_cpu[row * i3..(row + 1) * i3]);
        out_cpu.extend(
            l4_h_cpu[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b),
        );
    }
    report(
        "layer4_down_projection_residual",
        "Q6_K",
        &format!("[{n},{h}]"),
        n,
        &out_gpu,
        &out_cpu,
    );
    // Continue from the real layer-4 post-down residual into block 5.  Keep
    // the CPU path independent so the first failing layer-3 stage is useful.
    let l5w = vec32(&p, "blk.5.attn_norm.weight");
    let l5qw = tensor(&r, &p, "blk.5.attn_q.weight");
    let l5kw = tensor(&r, &p, "blk.5.attn_k.weight");
    let l5vw = tensor(&r, &p, "blk.5.attn_v.weight");
    assert_eq!(l5w.len(), h);
    report_q_metadata(&r, &p, "blk.5.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.5.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.5.attn_v.weight", h, nkv * hd);

    let l5_in = upload(&s, &d, &out_gpu)?;
    let l5_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l5_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l5_in,
        &l5_zero,
        &upload(&s, &d, &l5w)?,
        &l5_norm,
        &l5_norm,
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
    let l5_norm_gpu = download(&s, &l5_norm, n * h)?;
    let l5_norm_cpu = out_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l5w, eps))
        .collect::<Vec<_>>();
    report(
        "layer5_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l5_norm_gpu,
        &l5_norm_cpu,
    );

    let l5_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l5_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l5_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l5_q_info = r.get_tensor("blk.5.attn_q.weight").unwrap();
    let l5_k_info = r.get_tensor("blk.5.attn_k.weight").unwrap();
    let l5_v_info = r.get_tensor("blk.5.attn_v.weight").unwrap();
    g.gemm(
        &s,
        &upload_raw(&s, &d, l5qw.data)?,
        &l5_norm,
        &l5_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l5_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l5kw.data)?,
        &l5_norm,
        &l5_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l5_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l5vw.data)?,
        &l5_norm,
        &l5_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l5_v_info.ggml_type),
    )?;
    let l5_q_gpu = download(&s, &l5_q, n * nh * hd)?;
    let l5_k_gpu = download(&s, &l5_k, n * nkv * hd)?;
    let l5_v_gpu = download(&s, &l5_v, n * nkv * hd)?;
    let l5_q_cpu = norm_cpu_for_ffn(&l5_norm_cpu, &l5qw, nh * hd, h);
    let l5_k_cpu = norm_cpu_for_ffn(&l5_norm_cpu, &l5kw, nkv * hd, h);
    let l5_v_cpu = norm_cpu_for_ffn(&l5_norm_cpu, &l5vw, nkv * hd, h);
    report(
        "layer5_q_raw_gemm",
        &format!("{:?}", l5_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l5_q_gpu,
        &l5_q_cpu,
    );
    report(
        "layer5_k_raw_gemm",
        &format!("{:?}", l5_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l5_k_gpu,
        &l5_k_cpu,
    );
    report(
        "layer5_v_raw_gemm",
        &format!("{:?}", l5_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l5_v_gpu,
        &l5_v_cpu,
    );

    let l5_qn = vec32(&p, "blk.5.attn_q_norm.weight");
    let l5_kn = vec32(&p, "blk.5.attn_k_norm.weight");
    let l5_qn_g = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l5_kn_g = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l5_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l5_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l5_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l5_q,
        &l5_zq,
        &upload(&s, &d, &l5_qn)?,
        &l5_qn_g,
        &l5_qn_g,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l5_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l5_k,
        &l5_zk,
        &upload(&s, &d, &l5_kn)?,
        &l5_kn_g,
        &l5_kn_g,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l5_pos),
        n * nkv,
        nkv,
    )?;
    let l5_q_norm_rope_gpu = download(&s, &l5_qn_g, n * nh * hd)?;
    let l5_k_norm_rope_gpu = download(&s, &l5_kn_g, n * nkv * hd)?;
    let mut l5_q_norm_rope_cpu = Vec::with_capacity(n * nh * hd);
    let mut l5_k_norm_rope_cpu = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            l5_q_norm_rope_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l5_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l5_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            l5_k_norm_rope_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l5_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l5_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer5_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l5_q_norm_rope_gpu,
        &l5_q_norm_rope_cpu,
    );
    report(
        "layer5_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l5_k_norm_rope_gpu,
        &l5_k_norm_rope_cpu,
    );

    report(
        "layer5_v_gemm",
        &format!("{:?}", l5_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l5_v_gpu,
        &l5_v_cpu,
    );
    let l5_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l5_bt = upload(&s, &d, &[0.])?;
    let l5_kd = upload(&s, &d, &l5_k_norm_rope_gpu)?;
    let l5_vd = upload(&s, &d, &l5_v_gpu)?;
    kv.append_kv(&s, &layout, &l5_pool, &l5_kd, &l5_vd, &l5_bt, 0, n)?;
    let l5_qd = upload(&s, &d, &l5_q_norm_rope_gpu)?;
    let l5_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(&s, &l5_qd, &l5_pool, &l5_bt, &l5_att, nh, nkv, hd, 64, n, 0)?;
    let l5_att_gpu = download(&s, &l5_att, n * nh * hd)?;
    let mut l5_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd]
                .copy_from_slice(&l5_k_norm_rope_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l5_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l5_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &l5_q_norm_rope_cpu[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer5_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l5_att_gpu,
        &l5_att_cpu,
    );

    let l5_wo = tensor(&r, &p, "blk.5.attn_output.weight");
    let l5_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l5_wo.data)?,
        &l5_att,
        &l5_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.5.attn_output.weight").unwrap().ggml_type),
        Some(&l5_in),
    )?;
    let l5_h_gpu = download(&s, &l5_h, n * h)?;
    let l5_h_cpu = out_cpu
        .chunks_exact(h)
        .zip(l5_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l5_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer5_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l5_h_gpu,
        &l5_h_cpu,
    );

    let l5_fn = vec32(&p, "blk.5.ffn_norm.weight");
    let l5_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l5_h,
        &l5_zero,
        &upload(&s, &d, &l5_fn)?,
        &l5_f,
        &l5_f,
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
    let l5_f_gpu = download(&s, &l5_f, n * h)?;
    let l5_f_cpu = l5_h_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l5_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer5_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l5_f_gpu,
        &l5_f_cpu,
    );

    let i3 = r.get_tensor("blk.5.ffn_gate.weight").unwrap().dims[1] as usize;
    let l5_gw = tensor(&r, &p, "blk.5.ffn_gate.weight");
    let l5_uw = tensor(&r, &p, "blk.5.ffn_up.weight");
    let l5_dw = tensor(&r, &p, "blk.5.ffn_down.weight");
    let gd = upload_raw(&s, &d, l5_gw.data)?;
    let ud = upload_raw(&s, &d, l5_uw.data)?;
    let go = DeviceBuffer::alloc(d.clone(), n * i3 * 4)?;
    let uo = DeviceBuffer::alloc(d.clone(), n * i3 * 4)?;
    for (w, o, name) in [(&gd, &go, "layer5_gate_gemm"), (&ud, &uo, "layer5_up_gemm")] {
        g.gemm(
            &s,
            w,
            &l5_f,
            o,
            h,
            i3,
            n,
            gemm_fmt(
                r.get_tensor(if name.ends_with("gate_gemm") {
                    "blk.5.ffn_gate.weight"
                } else {
                    "blk.5.ffn_up.weight"
                })
                .unwrap()
                .ggml_type,
            ),
        )?;
        let gpu = download(&s, o, n * i3)?;
        let cpu = norm_cpu_for_ffn(
            &l5_f_cpu,
            if name.ends_with("gate_gemm") {
                &l5_gw
            } else {
                &l5_uw
            },
            i3,
            h,
        );
        report(name, "Q4_K", &format!("[{n},{i3}]"), n, &gpu, &cpu);
    }
    let gate_cpu = norm_cpu_for_ffn(&l5_f_cpu, &l5_gw, i3, h);
    let up_cpu = norm_cpu_for_ffn(&l5_f_cpu, &l5_uw, i3, h);
    let sw_cpu = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &gate_cpu[row * i3..(row + 1) * i3],
                &up_cpu[row * i3..(row + 1) * i3],
            )
        })
        .collect::<Vec<_>>();
    let sw = DeviceBuffer::alloc(d.clone(), n * i3 * 4)?;
    let zi = upload(&s, &d, &vec![0.; n * i3])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &go,
        &zi,
        &zi,
        &uo,
        &sw,
        eps,
        i3,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let sw_gpu = download(&s, &sw, n * i3)?;
    report(
        "layer5_swiglu",
        "F32",
        &format!("[{n},{i3}]"),
        n,
        &sw_gpu,
        &sw_cpu,
    );

    let out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l5_dw.data)?,
        &sw,
        &out,
        i3,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.5.ffn_down.weight").unwrap().ggml_type),
        Some(&l5_h),
    )?;
    let out_gpu = download(&s, &out, n * h)?;
    let mut out_cpu = Vec::with_capacity(n * h);
    for row in 0..n {
        let mut z = vec![0.; h];
        matmul(&mut z, &l5_dw, &sw_cpu[row * i3..(row + 1) * i3]);
        out_cpu.extend(
            l5_h_cpu[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b),
        );
    }
    report(
        "layer5_down_projection_residual",
        "Q6_K",
        &format!("[{n},{h}]"),
        n,
        &out_gpu,
        &out_cpu,
    );

    // Continue through the complete block 6 from the real layer-5 residual.
    let l6w = vec32(&p, "blk.6.attn_norm.weight");
    let l6qw = tensor(&r, &p, "blk.6.attn_q.weight");
    let l6kw = tensor(&r, &p, "blk.6.attn_k.weight");
    let l6vw = tensor(&r, &p, "blk.6.attn_v.weight");
    assert_eq!(l6w.len(), h);
    report_q_metadata(&r, &p, "blk.6.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.6.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.6.attn_v.weight", h, nkv * hd);

    let l6_in = upload(&s, &d, &out_gpu)?;
    let l6_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l6w_dev = upload(&s, &d, &l6w)?;
    let l6_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l6_in,
        &l6_zero,
        &l6w_dev,
        &l6_norm,
        &l6_norm,
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
    let l6_norm_gpu = download(&s, &l6_norm, n * h)?;
    let l6_norm_cpu = out_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l6w, eps))
        .collect::<Vec<_>>();
    report(
        "layer6_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l6_norm_gpu,
        &l6_norm_cpu,
    );

    let l6_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l6_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l6_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l6_q_info = r.get_tensor("blk.6.attn_q.weight").unwrap();
    let l6_k_info = r.get_tensor("blk.6.attn_k.weight").unwrap();
    let l6_v_info = r.get_tensor("blk.6.attn_v.weight").unwrap();
    g.gemm(
        &s,
        &upload_raw(&s, &d, l6qw.data)?,
        &l6_norm,
        &l6_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l6_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l6kw.data)?,
        &l6_norm,
        &l6_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l6_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l6vw.data)?,
        &l6_norm,
        &l6_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l6_v_info.ggml_type),
    )?;
    let l6_q_gpu = download(&s, &l6_q, n * nh * hd)?;
    let l6_k_gpu = download(&s, &l6_k, n * nkv * hd)?;
    let l6_v_gpu = download(&s, &l6_v, n * nkv * hd)?;
    let l6_q_cpu = norm_cpu_for_ffn(&l6_norm_cpu, &l6qw, nh * hd, h);
    let l6_k_cpu = norm_cpu_for_ffn(&l6_norm_cpu, &l6kw, nkv * hd, h);
    let l6_v_cpu = norm_cpu_for_ffn(&l6_norm_cpu, &l6vw, nkv * hd, h);
    report(
        "layer6_q_raw_gemm",
        &format!("{:?}", l6_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l6_q_gpu,
        &l6_q_cpu,
    );
    report(
        "layer6_k_raw_gemm",
        &format!("{:?}", l6_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l6_k_gpu,
        &l6_k_cpu,
    );
    report(
        "layer6_v_raw_gemm",
        &format!("{:?}", l6_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l6_v_gpu,
        &l6_v_cpu,
    );

    let l6_qn = vec32(&p, "blk.6.attn_q_norm.weight");
    let l6_kn = vec32(&p, "blk.6.attn_k_norm.weight");
    let l6_qnr = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l6_knr = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l6_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l6_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l6_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l6_q,
        &l6_zq,
        &upload(&s, &d, &l6_qn)?,
        &l6_qnr,
        &l6_qnr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l6_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l6_k,
        &l6_zk,
        &upload(&s, &d, &l6_kn)?,
        &l6_knr,
        &l6_knr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l6_pos),
        n * nkv,
        nkv,
    )?;
    let q6g = download(&s, &l6_qnr, n * nh * hd)?;
    let k6g = download(&s, &l6_knr, n * nkv * hd)?;
    let mut q6c = Vec::with_capacity(n * nh * hd);
    let mut k6c = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            q6c.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l6_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l6_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            k6c.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l6_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l6_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer6_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &q6g,
        &q6c,
    );
    report(
        "layer6_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &k6g,
        &k6c,
    );
    report(
        "layer6_v_gemm",
        &format!("{:?}", l6_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l6_v_gpu,
        &l6_v_cpu,
    );

    let l6_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l6_bt = upload(&s, &d, &[0.])?;
    let l6_kd = upload(&s, &d, &k6g)?;
    let l6_vd = upload(&s, &d, &l6_v_gpu)?;
    kv.append_kv(&s, &layout, &l6_pool, &l6_kd, &l6_vd, &l6_bt, 0, n)?;
    let l6_qd = upload(&s, &d, &q6g)?;
    let l6_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(&s, &l6_qd, &l6_pool, &l6_bt, &l6_att, nh, nkv, hd, 64, n, 0)?;
    let l6_att_gpu = download(&s, &l6_att, n * nh * hd)?;
    let mut l6_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd].copy_from_slice(&k6c[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l6_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l6_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &q6c[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer6_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l6_att_gpu,
        &l6_att_cpu,
    );

    let l6_wo = tensor(&r, &p, "blk.6.attn_output.weight");
    let l6_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    report(
        "hidden5_before_layer6",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &out_gpu,
        &out_cpu,
    );
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l6_wo.data)?,
        &l6_att,
        &l6_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.6.attn_output.weight").unwrap().ggml_type),
        Some(&l6_in),
    )?;
    let l6_hg = download(&s, &l6_h, n * h)?;
    let l6_hc = out_cpu
        .chunks_exact(h)
        .zip(l6_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l6_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer6_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l6_hg,
        &l6_hc,
    );

    let l6_fn = vec32(&p, "blk.6.ffn_norm.weight");
    let l6_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l6_h,
        &l6_zero,
        &upload(&s, &d, &l6_fn)?,
        &l6_f,
        &l6_f,
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
    let l6_fg = download(&s, &l6_f, n * h)?;
    let l6_fc = l6_hc
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l6_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer6_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l6_fg,
        &l6_fc,
    );

    let i6 = r.get_tensor("blk.6.ffn_gate.weight").unwrap().dims[1] as usize;
    let l6_gw = tensor(&r, &p, "blk.6.ffn_gate.weight");
    let l6_uw = tensor(&r, &p, "blk.6.ffn_up.weight");
    let l6_dw = tensor(&r, &p, "blk.6.ffn_down.weight");
    let gd = upload_raw(&s, &d, l6_gw.data)?;
    let ud = upload_raw(&s, &d, l6_uw.data)?;
    let go = DeviceBuffer::alloc(d.clone(), n * i6 * 4)?;
    let uo = DeviceBuffer::alloc(d.clone(), n * i6 * 4)?;
    g.gemm(
        &s,
        &gd,
        &l6_f,
        &go,
        h,
        i6,
        n,
        gemm_fmt(r.get_tensor("blk.6.ffn_gate.weight").unwrap().ggml_type),
    )?;
    g.gemm(
        &s,
        &ud,
        &l6_f,
        &uo,
        h,
        i6,
        n,
        gemm_fmt(r.get_tensor("blk.6.ffn_up.weight").unwrap().ggml_type),
    )?;
    let gog = download(&s, &go, n * i6)?;
    let uog = download(&s, &uo, n * i6)?;
    let goc = norm_cpu_for_ffn(&l6_fc, &l6_gw, i6, h);
    let uoc = norm_cpu_for_ffn(&l6_fc, &l6_uw, i6, h);
    report(
        "layer6_gate_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.6.ffn_gate.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i6}]"),
        n,
        &gog,
        &goc,
    );
    report(
        "layer6_up_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.6.ffn_up.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i6}]"),
        n,
        &uog,
        &uoc,
    );
    let swc = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &goc[row * i6..(row + 1) * i6],
                &uoc[row * i6..(row + 1) * i6],
            )
        })
        .collect::<Vec<_>>();
    let swo = DeviceBuffer::alloc(d.clone(), n * i6 * 4)?;
    let zi = upload(&s, &d, &vec![0.; n * i6])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &go,
        &zi,
        &zi,
        &uo,
        &swo,
        eps,
        i6,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let swg = download(&s, &swo, n * i6)?;
    report(
        "layer6_swiglu",
        "F32",
        &format!("[{n},{i6}]"),
        n,
        &swg,
        &swc,
    );
    let finalg = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l6_dw.data)?,
        &swo,
        &finalg,
        i6,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.6.ffn_down.weight").unwrap().ggml_type),
        Some(&l6_h),
    )?;
    let final_gpu = download(&s, &finalg, n * h)?;
    let mut final_cpu = Vec::with_capacity(n * h);
    for row in 0..n {
        let mut z = vec![0.; h];
        matmul(&mut z, &l6_dw, &swc[row * i6..(row + 1) * i6]);
        final_cpu.extend(
            l6_hc[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b),
        );
    }
    report(
        "layer6_down_projection_residual",
        &format!(
            "{:?}",
            r.get_tensor("blk.6.ffn_down.weight").unwrap().ggml_type
        ),
        &format!("[{n},{h}]"),
        n,
        &final_gpu,
        &final_cpu,
    );

    // Continue through the complete block 7 from the real layer-6 residual.
    let l7w = vec32(&p, "blk.7.attn_norm.weight");
    let l7qw = tensor(&r, &p, "blk.7.attn_q.weight");
    let l7kw = tensor(&r, &p, "blk.7.attn_k.weight");
    let l7vw = tensor(&r, &p, "blk.7.attn_v.weight");
    assert_eq!(l7w.len(), h);
    report_q_metadata(&r, &p, "blk.7.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.7.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.7.attn_v.weight", h, nkv * hd);

    let l7_in = upload(&s, &d, &final_gpu)?;
    let l7_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l7w_dev = upload(&s, &d, &l7w)?;
    let l7_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l7_in,
        &l7_zero,
        &l7w_dev,
        &l7_norm,
        &l7_norm,
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
    let l7_norm_gpu = download(&s, &l7_norm, n * h)?;
    let l7_norm_cpu = final_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l7w, eps))
        .collect::<Vec<_>>();
    report(
        "layer7_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l7_norm_gpu,
        &l7_norm_cpu,
    );

    let l7_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l7_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l7_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l7_q_info = r.get_tensor("blk.7.attn_q.weight").unwrap();
    let l7_k_info = r.get_tensor("blk.7.attn_k.weight").unwrap();
    let l7_v_info = r.get_tensor("blk.7.attn_v.weight").unwrap();
    g.gemm(
        &s,
        &upload_raw(&s, &d, l7qw.data)?,
        &l7_norm,
        &l7_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l7_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l7kw.data)?,
        &l7_norm,
        &l7_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l7_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l7vw.data)?,
        &l7_norm,
        &l7_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l7_v_info.ggml_type),
    )?;
    let l7_q_gpu = download(&s, &l7_q, n * nh * hd)?;
    let l7_k_gpu = download(&s, &l7_k, n * nkv * hd)?;
    let l7_v_gpu = download(&s, &l7_v, n * nkv * hd)?;
    let l7_q_cpu = norm_cpu_for_ffn(&l7_norm_cpu, &l7qw, nh * hd, h);
    let l7_k_cpu = norm_cpu_for_ffn(&l7_norm_cpu, &l7kw, nkv * hd, h);
    let l7_v_cpu = norm_cpu_for_ffn(&l7_norm_cpu, &l7vw, nkv * hd, h);
    report(
        "layer7_q_raw_gemm",
        &format!("{:?}", l7_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l7_q_gpu,
        &l7_q_cpu,
    );
    report(
        "layer7_k_raw_gemm",
        &format!("{:?}", l7_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l7_k_gpu,
        &l7_k_cpu,
    );
    report(
        "layer7_v_raw_gemm",
        &format!("{:?}", l7_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l7_v_gpu,
        &l7_v_cpu,
    );

    let l7_qn = vec32(&p, "blk.7.attn_q_norm.weight");
    let l7_kn = vec32(&p, "blk.7.attn_k_norm.weight");
    let l7_qnr = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l7_knr = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l7_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l7_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l7_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l7_q,
        &l7_zq,
        &upload(&s, &d, &l7_qn)?,
        &l7_qnr,
        &l7_qnr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l7_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l7_k,
        &l7_zk,
        &upload(&s, &d, &l7_kn)?,
        &l7_knr,
        &l7_knr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l7_pos),
        n * nkv,
        nkv,
    )?;
    let q6g = download(&s, &l7_qnr, n * nh * hd)?;
    let k6g = download(&s, &l7_knr, n * nkv * hd)?;
    let mut q6c = Vec::with_capacity(n * nh * hd);
    let mut k6c = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            q6c.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l7_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l7_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            k6c.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l7_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l7_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer7_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &q6g,
        &q6c,
    );
    report(
        "layer7_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &k6g,
        &k6c,
    );
    report(
        "layer7_v_gemm",
        &format!("{:?}", l7_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l7_v_gpu,
        &l7_v_cpu,
    );

    let l7_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l7_bt = upload(&s, &d, &[0.])?;
    let l7_kd = upload(&s, &d, &k6g)?;
    let l7_vd = upload(&s, &d, &l7_v_gpu)?;
    kv.append_kv(&s, &layout, &l7_pool, &l7_kd, &l7_vd, &l7_bt, 0, n)?;
    let l7_qd = upload(&s, &d, &q6g)?;
    let l7_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(&s, &l7_qd, &l7_pool, &l7_bt, &l7_att, nh, nkv, hd, 64, n, 0)?;
    let l7_att_gpu = download(&s, &l7_att, n * nh * hd)?;
    let mut l7_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd].copy_from_slice(&k6c[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l7_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l7_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &q6c[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer7_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l7_att_gpu,
        &l7_att_cpu,
    );

    let l7_wo = tensor(&r, &p, "blk.7.attn_output.weight");
    let l7_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l7_wo.data)?,
        &l7_att,
        &l7_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.7.attn_output.weight").unwrap().ggml_type),
        Some(&l7_in),
    )?;
    let l7_hg = download(&s, &l7_h, n * h)?;
    let l7_hc = final_cpu
        .chunks_exact(h)
        .zip(l7_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l7_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer7_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l7_hg,
        &l7_hc,
    );

    let l7_fn = vec32(&p, "blk.7.ffn_norm.weight");
    let l7_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l7_h,
        &l7_zero,
        &upload(&s, &d, &l7_fn)?,
        &l7_f,
        &l7_f,
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
    let l7_fg = download(&s, &l7_f, n * h)?;
    let l7_fc = l7_hc
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l7_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer7_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l7_fg,
        &l7_fc,
    );

    let i7 = r.get_tensor("blk.7.ffn_gate.weight").unwrap().dims[1] as usize;
    let l7_gw = tensor(&r, &p, "blk.7.ffn_gate.weight");
    let l7_uw = tensor(&r, &p, "blk.7.ffn_up.weight");
    let l7_dw = tensor(&r, &p, "blk.7.ffn_down.weight");
    let gd = upload_raw(&s, &d, l7_gw.data)?;
    let ud = upload_raw(&s, &d, l7_uw.data)?;
    let go = DeviceBuffer::alloc(d.clone(), n * i7 * 4)?;
    let uo = DeviceBuffer::alloc(d.clone(), n * i7 * 4)?;
    g.gemm(
        &s,
        &gd,
        &l7_f,
        &go,
        h,
        i7,
        n,
        gemm_fmt(r.get_tensor("blk.7.ffn_gate.weight").unwrap().ggml_type),
    )?;
    g.gemm(
        &s,
        &ud,
        &l7_f,
        &uo,
        h,
        i7,
        n,
        gemm_fmt(r.get_tensor("blk.7.ffn_up.weight").unwrap().ggml_type),
    )?;
    let gog = download(&s, &go, n * i7)?;
    let uog = download(&s, &uo, n * i7)?;
    let goc = norm_cpu_for_ffn(&l7_fc, &l7_gw, i7, h);
    let uoc = norm_cpu_for_ffn(&l7_fc, &l7_uw, i7, h);
    report(
        "layer7_gate_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.7.ffn_gate.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i7}]"),
        n,
        &gog,
        &goc,
    );
    report(
        "layer7_up_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.7.ffn_up.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i7}]"),
        n,
        &uog,
        &uoc,
    );
    let swc = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &goc[row * i7..(row + 1) * i7],
                &uoc[row * i7..(row + 1) * i7],
            )
        })
        .collect::<Vec<_>>();
    let swo = DeviceBuffer::alloc(d.clone(), n * i7 * 4)?;
    let zi = upload(&s, &d, &vec![0.; n * i7])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &go,
        &zi,
        &zi,
        &uo,
        &swo,
        eps,
        i7,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let swg = download(&s, &swo, n * i7)?;
    report(
        "layer7_swiglu",
        "F32",
        &format!("[{n},{i7}]"),
        n,
        &swg,
        &swc,
    );
    let l7_finalg = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l7_dw.data)?,
        &swo,
        &l7_finalg,
        i7,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.7.ffn_down.weight").unwrap().ggml_type),
        Some(&l7_h),
    )?;
    let l7_final_gpu = download(&s, &l7_finalg, n * h)?;
    let mut l7_final_cpu = Vec::with_capacity(n * h);
    for row in 0..n {
        let mut z = vec![0.; h];
        matmul(&mut z, &l7_dw, &swc[row * i7..(row + 1) * i7]);
        l7_final_cpu.extend(
            l7_hc[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b),
        );
    }
    report(
        "layer7_down_projection_residual",
        &format!(
            "{:?}",
            r.get_tensor("blk.7.ffn_down.weight").unwrap().ggml_type
        ),
        &format!("[{n},{h}]"),
        n,
        &l7_final_gpu,
        &l7_final_cpu,
    );
    // Continue through the complete block 8 from the real layer-7 residual.
    let l8w = vec32(&p, "blk.8.attn_norm.weight");
    let l8qw = tensor(&r, &p, "blk.8.attn_q.weight");
    let l8kw = tensor(&r, &p, "blk.8.attn_k.weight");
    let l8vw = tensor(&r, &p, "blk.8.attn_v.weight");
    assert_eq!(l8w.len(), h);
    report_q_metadata(&r, &p, "blk.8.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.8.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.8.attn_v.weight", h, nkv * hd);

    let l8_in = upload(&s, &d, &final_gpu)?;
    let l8_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l8w_dev = upload(&s, &d, &l8w)?;
    let l8_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l8_in,
        &l8_zero,
        &l8w_dev,
        &l8_norm,
        &l8_norm,
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
    let l8_norm_gpu = download(&s, &l8_norm, n * h)?;
    let l8_norm_cpu = final_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l8w, eps))
        .collect::<Vec<_>>();
    report(
        "layer8_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l8_norm_gpu,
        &l8_norm_cpu,
    );

    let l8_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l8_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l8_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l8_q_info = r.get_tensor("blk.8.attn_q.weight").unwrap();
    let l8_k_info = r.get_tensor("blk.8.attn_k.weight").unwrap();
    let l8_v_info = r.get_tensor("blk.8.attn_v.weight").unwrap();
    g.gemm(
        &s,
        &upload_raw(&s, &d, l8qw.data)?,
        &l8_norm,
        &l8_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l8_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l8kw.data)?,
        &l8_norm,
        &l8_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l8_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l8vw.data)?,
        &l8_norm,
        &l8_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l8_v_info.ggml_type),
    )?;
    let l8_q_gpu = download(&s, &l8_q, n * nh * hd)?;
    let l8_k_gpu = download(&s, &l8_k, n * nkv * hd)?;
    let l8_v_gpu = download(&s, &l8_v, n * nkv * hd)?;
    let l8_q_cpu = norm_cpu_for_ffn(&l8_norm_cpu, &l8qw, nh * hd, h);
    let l8_k_cpu = norm_cpu_for_ffn(&l8_norm_cpu, &l8kw, nkv * hd, h);
    let l8_v_cpu = norm_cpu_for_ffn(&l8_norm_cpu, &l8vw, nkv * hd, h);
    report(
        "layer8_q_raw_gemm",
        &format!("{:?}", l8_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l8_q_gpu,
        &l8_q_cpu,
    );
    report(
        "layer8_k_raw_gemm",
        &format!("{:?}", l8_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l8_k_gpu,
        &l8_k_cpu,
    );
    report(
        "layer8_v_raw_gemm",
        &format!("{:?}", l8_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l8_v_gpu,
        &l8_v_cpu,
    );

    let l8_qn = vec32(&p, "blk.8.attn_q_norm.weight");
    let l8_kn = vec32(&p, "blk.8.attn_k_norm.weight");
    let l8_qnr = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l8_knr = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l8_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l8_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l8_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l8_q,
        &l8_zq,
        &upload(&s, &d, &l8_qn)?,
        &l8_qnr,
        &l8_qnr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l8_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l8_k,
        &l8_zk,
        &upload(&s, &d, &l8_kn)?,
        &l8_knr,
        &l8_knr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l8_pos),
        n * nkv,
        nkv,
    )?;
    let l8_qnr_gpu = download(&s, &l8_qnr, n * nh * hd)?;
    let l8_knr_gpu = download(&s, &l8_knr, n * nkv * hd)?;
    let mut l8_qnr_cpu = Vec::with_capacity(n * nh * hd);
    let mut l8_knr_cpu = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            l8_qnr_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l8_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l8_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            l8_knr_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l8_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l8_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer8_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l8_qnr_gpu,
        &l8_qnr_cpu,
    );
    report(
        "layer8_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l8_knr_gpu,
        &l8_knr_cpu,
    );
    report(
        "layer8_v_gemm",
        &format!("{:?}", l8_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l8_v_gpu,
        &l8_v_cpu,
    );

    let l8_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l8_bt = upload(&s, &d, &[0.])?;
    let l8_kd = upload(&s, &d, &l8_knr_gpu)?;
    let l8_vd = upload(&s, &d, &l8_v_gpu)?;
    kv.append_kv(&s, &layout, &l8_pool, &l8_kd, &l8_vd, &l8_bt, 0, n)?;
    let l8_qd = upload(&s, &d, &l8_qnr_gpu)?;
    let l8_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(&s, &l8_qd, &l8_pool, &l8_bt, &l8_att, nh, nkv, hd, 64, n, 0)?;
    let l8_att_gpu = download(&s, &l8_att, n * nh * hd)?;
    let mut l8_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd].copy_from_slice(&l8_knr_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l8_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l8_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &l8_qnr_cpu[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer8_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l8_att_gpu,
        &l8_att_cpu,
    );

    let l8_wo = tensor(&r, &p, "blk.8.attn_output.weight");
    let l8_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l8_wo.data)?,
        &l8_att,
        &l8_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.8.attn_output.weight").unwrap().ggml_type),
        Some(&l8_in),
    )?;
    let l8_hg = download(&s, &l8_h, n * h)?;
    let l8_hc = final_cpu
        .chunks_exact(h)
        .zip(l8_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l8_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer8_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l8_hg,
        &l8_hc,
    );

    let l8_fn = vec32(&p, "blk.8.ffn_norm.weight");
    let l8_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l8_h,
        &l8_zero,
        &upload(&s, &d, &l8_fn)?,
        &l8_f,
        &l8_f,
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
    let l8_fg = download(&s, &l8_f, n * h)?;
    let l8_fc = l8_hc
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l8_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer8_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l8_fg,
        &l8_fc,
    );

    let i8 = r.get_tensor("blk.8.ffn_gate.weight").unwrap().dims[1] as usize;
    let l8_gw = tensor(&r, &p, "blk.8.ffn_gate.weight");
    let l8_uw = tensor(&r, &p, "blk.8.ffn_up.weight");
    let l8_dw = tensor(&r, &p, "blk.8.ffn_down.weight");
    let gd = upload_raw(&s, &d, l8_gw.data)?;
    let ud = upload_raw(&s, &d, l8_uw.data)?;
    let l8_go = DeviceBuffer::alloc(d.clone(), n * i8 * 4)?;
    let l8_uo = DeviceBuffer::alloc(d.clone(), n * i8 * 4)?;
    g.gemm(
        &s,
        &gd,
        &l8_f,
        &l8_go,
        h,
        i8,
        n,
        gemm_fmt(r.get_tensor("blk.8.ffn_gate.weight").unwrap().ggml_type),
    )?;
    g.gemm(
        &s,
        &ud,
        &l8_f,
        &l8_uo,
        h,
        i8,
        n,
        gemm_fmt(r.get_tensor("blk.8.ffn_up.weight").unwrap().ggml_type),
    )?;
    let l8_gog = download(&s, &l8_go, n * i8)?;
    let l8_uog = download(&s, &l8_uo, n * i8)?;
    let l8_goc = norm_cpu_for_ffn(&l8_fc, &l8_gw, i8, h);
    let l8_uoc = norm_cpu_for_ffn(&l8_fc, &l8_uw, i8, h);
    report(
        "layer8_gate_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.8.ffn_gate.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i8}]"),
        n,
        &l8_gog,
        &l8_goc,
    );
    report(
        "layer8_up_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.8.ffn_up.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i8}]"),
        n,
        &l8_uog,
        &l8_uoc,
    );
    let l8_swc = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &l8_goc[row * i8..(row + 1) * i8],
                &l8_uoc[row * i8..(row + 1) * i8],
            )
        })
        .collect::<Vec<_>>();
    let l8_swo = DeviceBuffer::alloc(d.clone(), n * i8 * 4)?;
    let zi = upload(&s, &d, &vec![0.; n * i8])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l8_go,
        &zi,
        &zi,
        &l8_uo,
        &l8_swo,
        eps,
        i8,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let swg = download(&s, &l8_swo, n * i8)?;
    report(
        "layer8_swiglu",
        "F32",
        &format!("[{n},{i8}]"),
        n,
        &swg,
        &l8_swc,
    );
    let l8_finalg = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l8_dw.data)?,
        &l8_swo,
        &l8_finalg,
        i8,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.8.ffn_down.weight").unwrap().ggml_type),
        Some(&l8_h),
    )?;
    let l8_final_gpu = download(&s, &l8_finalg, n * h)?;
    let mut l8_final_cpu = Vec::with_capacity(n * h);
    for row in 0..n {
        let mut z = vec![0.; h];
        matmul(&mut z, &l8_dw, &l8_swc[row * i8..(row + 1) * i8]);
        l8_final_cpu.extend(
            l8_hc[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b),
        );
    }
    report(
        "layer8_down_projection_residual",
        &format!(
            "{:?}",
            r.get_tensor("blk.8.ffn_down.weight").unwrap().ggml_type
        ),
        &format!("[{n},{h}]"),
        n,
        &l8_final_gpu,
        &l8_final_cpu,
    );

    // Continue from the real layer-8 post-down residual into block 9 only
    // through input RMSNorm and raw Q/K/V GEMMs.
    let l9w = vec32(&p, "blk.9.attn_norm.weight");
    let l9qw = tensor(&r, &p, "blk.9.attn_q.weight");
    let l9kw = tensor(&r, &p, "blk.9.attn_k.weight");
    let l9vw = tensor(&r, &p, "blk.9.attn_v.weight");
    assert_eq!(l9w.len(), h);
    report_q_metadata(&r, &p, "blk.9.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.9.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.9.attn_v.weight", h, nkv * hd);

    let l9_in = upload(&s, &d, &l8_final_gpu)?;
    let l9_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l9_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l9_in,
        &l9_zero,
        &upload(&s, &d, &l9w)?,
        &l9_norm,
        &l9_norm,
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
    let l9_norm_gpu = download(&s, &l9_norm, n * h)?;
    let l9_norm_cpu = l8_final_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l9w, eps))
        .collect::<Vec<_>>();
    report(
        "layer9_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l9_norm_gpu,
        &l9_norm_cpu,
    );

    let l9_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l9_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l9_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l9_q_info = r.get_tensor("blk.9.attn_q.weight").unwrap();
    let l9_k_info = r.get_tensor("blk.9.attn_k.weight").unwrap();
    let l9_v_info = r.get_tensor("blk.9.attn_v.weight").unwrap();
    let l9_qd = upload_raw(&s, &d, l9qw.data)?;
    let l9_kd = upload_raw(&s, &d, l9kw.data)?;
    let l9_vd = upload_raw(&s, &d, l9vw.data)?;
    g.gemm(
        &s,
        &l9_qd,
        &l9_norm,
        &l9_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l9_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l9_kd,
        &l9_norm,
        &l9_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l9_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l9_vd,
        &l9_norm,
        &l9_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l9_v_info.ggml_type),
    )?;
    let l9_q_gpu = download(&s, &l9_q, n * nh * hd)?;
    let l9_k_gpu = download(&s, &l9_k, n * nkv * hd)?;
    let l9_v_gpu = download(&s, &l9_v, n * nkv * hd)?;
    let l9_q_cpu = norm_cpu_for_ffn(&l9_norm_cpu, &l9qw, nh * hd, h);
    let l9_k_cpu = norm_cpu_for_ffn(&l9_norm_cpu, &l9kw, nkv * hd, h);
    let l9_v_cpu = norm_cpu_for_ffn(&l9_norm_cpu, &l9vw, nkv * hd, h);
    report(
        "layer9_q_raw_gemm",
        &format!("{:?}", l9_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l9_q_gpu,
        &l9_q_cpu,
    );
    report(
        "layer9_k_raw_gemm",
        &format!("{:?}", l9_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l9_k_gpu,
        &l9_k_cpu,
    );
    report(
        "layer9_v_raw_gemm",
        &format!("{:?}", l9_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l9_v_gpu,
        &l9_v_cpu,
    );

    let l9_qn = vec32(&p, "blk.9.attn_q_norm.weight");
    let l9_kn = vec32(&p, "blk.9.attn_k_norm.weight");
    let l9_wo = tensor(&r, &p, "blk.9.attn_output.weight");
    let l9_fn = vec32(&p, "blk.9.ffn_norm.weight");
    let l9_gw = tensor(&r, &p, "blk.9.ffn_gate.weight");
    let l9_uw = tensor(&r, &p, "blk.9.ffn_up.weight");
    let l9_dw = tensor(&r, &p, "blk.9.ffn_down.weight");
    report_q_metadata(&r, &p, "blk.9.attn_output.weight", nh * hd, h);

    let l9_qr = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l9_kr = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l9_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l9_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l9_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l9_q,
        &l9_zq,
        &upload(&s, &d, &l9_qn)?,
        &l9_qr,
        &l9_qr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l9_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l9_k,
        &l9_zk,
        &upload(&s, &d, &l9_kn)?,
        &l9_kr,
        &l9_kr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l9_pos),
        n * nkv,
        nkv,
    )?;
    let l9_qn_gpu = download(&s, &l9_qr, n * nh * hd)?;
    let l9_kn_gpu = download(&s, &l9_kr, n * nkv * hd)?;
    let mut l9_qn_cpu = Vec::with_capacity(n * nh * hd);
    let mut l9_kn_cpu = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            l9_qn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l9_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l9_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            l9_kn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l9_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l9_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer9_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l9_qn_gpu,
        &l9_qn_cpu,
    );
    report(
        "layer9_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l9_kn_gpu,
        &l9_kn_cpu,
    );
    report(
        "layer9_v_gemm",
        &format!("{:?}", l9_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l9_v_gpu,
        &l9_v_cpu,
    );

    let l9_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l9_bt = upload(&s, &d, &[0.])?;
    kv.append_kv(
        &s,
        &layout,
        &l9_pool,
        &upload(&s, &d, &l9_kn_gpu)?,
        &upload(&s, &d, &l9_v_gpu)?,
        &l9_bt,
        0,
        n,
    )?;
    let l9_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(
        &s,
        &upload(&s, &d, &l9_qn_gpu)?,
        &l9_pool,
        &l9_bt,
        &l9_att,
        nh,
        nkv,
        hd,
        64,
        n,
        0,
    )?;
    let l9_att_gpu = download(&s, &l9_att, n * nh * hd)?;
    let mut l9_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd].copy_from_slice(&l9_kn_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l9_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l9_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &l9_qn_cpu[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer9_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l9_att_gpu,
        &l9_att_cpu,
    );

    let l9_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l9_wo.data)?,
        &l9_att,
        &l9_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.9.attn_output.weight").unwrap().ggml_type),
        Some(&l9_in),
    )?;
    let l9_h_gpu = download(&s, &l9_h, n * h)?;
    let l9_h_cpu = l8_final_cpu
        .chunks_exact(h)
        .zip(l9_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l9_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer9_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l9_h_gpu,
        &l9_h_cpu,
    );

    let l9_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l9_h,
        &l9_zero,
        &upload(&s, &d, &l9_fn)?,
        &l9_f,
        &l9_f,
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
    let l9_f_gpu = download(&s, &l9_f, n * h)?;
    let l9_f_cpu = l9_h_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l9_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer9_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l9_f_gpu,
        &l9_f_cpu,
    );
    let i9 = r.get_tensor("blk.9.ffn_gate.weight").unwrap().dims[1] as usize;
    let l9_go = DeviceBuffer::alloc(d.clone(), n * i9 * 4)?;
    let l9_uo = DeviceBuffer::alloc(d.clone(), n * i9 * 4)?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l9_gw.data)?,
        &l9_f,
        &l9_go,
        h,
        i9,
        n,
        gemm_fmt(r.get_tensor("blk.9.ffn_gate.weight").unwrap().ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l9_uw.data)?,
        &l9_f,
        &l9_uo,
        h,
        i9,
        n,
        gemm_fmt(r.get_tensor("blk.9.ffn_up.weight").unwrap().ggml_type),
    )?;
    let l9_gg = download(&s, &l9_go, n * i9)?;
    let l9_ug = download(&s, &l9_uo, n * i9)?;
    let l9_gc = norm_cpu_for_ffn(&l9_f_cpu, &l9_gw, i9, h);
    let l9_uc = norm_cpu_for_ffn(&l9_f_cpu, &l9_uw, i9, h);
    report(
        "layer9_gate_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.9.ffn_gate.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i9}]"),
        n,
        &l9_gg,
        &l9_gc,
    );
    report(
        "layer9_up_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.9.ffn_up.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i9}]"),
        n,
        &l9_ug,
        &l9_uc,
    );
    let l9_sw_cpu = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &l9_gc[row * i9..(row + 1) * i9],
                &l9_uc[row * i9..(row + 1) * i9],
            )
        })
        .collect::<Vec<_>>();
    let l9_sw = DeviceBuffer::alloc(d.clone(), n * i9 * 4)?;
    let l9_zi = upload(&s, &d, &vec![0.; n * i9])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l9_go,
        &l9_zi,
        &l9_zi,
        &l9_uo,
        &l9_sw,
        eps,
        i9,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let l9_sw_gpu = download(&s, &l9_sw, n * i9)?;
    report(
        "layer9_swiglu",
        "F32",
        &format!("[{n},{i9}]"),
        n,
        &l9_sw_gpu,
        &l9_sw_cpu,
    );
    let l9_out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l9_dw.data)?,
        &l9_sw,
        &l9_out,
        i9,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.9.ffn_down.weight").unwrap().ggml_type),
        Some(&l9_h),
    )?;
    let l9_out_gpu = download(&s, &l9_out, n * h)?;
    let l9_out_cpu = (0..n)
        .flat_map(|row| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l9_dw, &l9_sw_cpu[row * i9..(row + 1) * i9]);
            l9_h_cpu[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer9_down_projection_residual",
        &format!(
            "{:?}",
            r.get_tensor("blk.9.ffn_down.weight").unwrap().ggml_type
        ),
        &format!("[{n},{h}]"),
        n,
        &l9_out_gpu,
        &l9_out_cpu,
    );
    // Continue from the real layer-9 post-down residual into block 10 only
    // through input RMSNorm and raw Q/K/V GEMMs.
    let l9w = vec32(&p, "blk.10.attn_norm.weight");
    let l9qw = tensor(&r, &p, "blk.10.attn_q.weight");
    let l9kw = tensor(&r, &p, "blk.10.attn_k.weight");
    let l9vw = tensor(&r, &p, "blk.10.attn_v.weight");
    assert_eq!(l9w.len(), h);
    report_q_metadata(&r, &p, "blk.10.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.10.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.10.attn_v.weight", h, nkv * hd);

    let l10_in = upload(&s, &d, &l9_out_gpu)?;
    let l10_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l10_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l10_in,
        &l10_zero,
        &upload(&s, &d, &l9w)?,
        &l10_norm,
        &l10_norm,
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
    let l10_norm_gpu = download(&s, &l10_norm, n * h)?;
    let l10_norm_cpu = l9_out_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l9w, eps))
        .collect::<Vec<_>>();
    report(
        "layer10_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l10_norm_gpu,
        &l10_norm_cpu,
    );

    let l10_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l10_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l10_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l10_q_info = r.get_tensor("blk.10.attn_q.weight").unwrap();
    let l10_k_info = r.get_tensor("blk.10.attn_k.weight").unwrap();
    let l10_v_info = r.get_tensor("blk.10.attn_v.weight").unwrap();
    let l10_qd = upload_raw(&s, &d, l9qw.data)?;
    let l10_kd = upload_raw(&s, &d, l9kw.data)?;
    let l10_vd = upload_raw(&s, &d, l9vw.data)?;
    g.gemm(
        &s,
        &l10_qd,
        &l10_norm,
        &l10_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l10_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l10_kd,
        &l10_norm,
        &l10_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l10_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l10_vd,
        &l10_norm,
        &l10_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l10_v_info.ggml_type),
    )?;
    let l10_q_gpu = download(&s, &l10_q, n * nh * hd)?;
    let l10_k_gpu = download(&s, &l10_k, n * nkv * hd)?;
    let l10_v_gpu = download(&s, &l10_v, n * nkv * hd)?;
    let l10_q_cpu = norm_cpu_for_ffn(&l10_norm_cpu, &l9qw, nh * hd, h);
    let l10_k_cpu = norm_cpu_for_ffn(&l10_norm_cpu, &l9kw, nkv * hd, h);
    let l10_v_cpu = norm_cpu_for_ffn(&l10_norm_cpu, &l9vw, nkv * hd, h);
    report(
        "layer10_q_raw_gemm",
        &format!("{:?}", l10_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l10_q_gpu,
        &l10_q_cpu,
    );
    report(
        "layer10_k_raw_gemm",
        &format!("{:?}", l10_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l10_k_gpu,
        &l10_k_cpu,
    );
    report(
        "layer10_v_raw_gemm",
        &format!("{:?}", l10_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l10_v_gpu,
        &l10_v_cpu,
    );

    let l10_qn = vec32(&p, "blk.10.attn_q_norm.weight");
    let l10_kn = vec32(&p, "blk.10.attn_k_norm.weight");
    let l10_wo = tensor(&r, &p, "blk.10.attn_output.weight");
    let l10_fn = vec32(&p, "blk.10.ffn_norm.weight");
    let l10_gw = tensor(&r, &p, "blk.10.ffn_gate.weight");
    let l10_uw = tensor(&r, &p, "blk.10.ffn_up.weight");
    let l10_dw = tensor(&r, &p, "blk.10.ffn_down.weight");
    report_q_metadata(&r, &p, "blk.10.attn_output.weight", nh * hd, h);

    let l10_qr = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l10_kr = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l10_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l10_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l10_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l10_q,
        &l10_zq,
        &upload(&s, &d, &l10_qn)?,
        &l10_qr,
        &l10_qr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l10_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l10_k,
        &l10_zk,
        &upload(&s, &d, &l10_kn)?,
        &l10_kr,
        &l10_kr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l10_pos),
        n * nkv,
        nkv,
    )?;
    let l10_qn_gpu = download(&s, &l10_qr, n * nh * hd)?;
    let l10_kn_gpu = download(&s, &l10_kr, n * nkv * hd)?;
    let mut l10_qn_cpu = Vec::with_capacity(n * nh * hd);
    let mut l10_kn_cpu = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            l10_qn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l10_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l10_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            l10_kn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l10_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l10_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer10_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l10_qn_gpu,
        &l10_qn_cpu,
    );
    report(
        "layer10_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l10_kn_gpu,
        &l10_kn_cpu,
    );
    report(
        "layer10_v_gemm",
        &format!("{:?}", l10_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l10_v_gpu,
        &l10_v_cpu,
    );

    let l10_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l10_bt = upload(&s, &d, &[0.])?;
    kv.append_kv(
        &s,
        &layout,
        &l10_pool,
        &upload(&s, &d, &l10_kn_gpu)?,
        &upload(&s, &d, &l10_v_gpu)?,
        &l10_bt,
        0,
        n,
    )?;
    let l10_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(
        &s,
        &upload(&s, &d, &l10_qn_gpu)?,
        &l10_pool,
        &l10_bt,
        &l10_att,
        nh,
        nkv,
        hd,
        64,
        n,
        0,
    )?;
    let l10_att_gpu = download(&s, &l10_att, n * nh * hd)?;
    let mut l10_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd].copy_from_slice(&l10_kn_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l10_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l10_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &l10_qn_cpu[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer10_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l10_att_gpu,
        &l10_att_cpu,
    );

    let l10_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l10_wo.data)?,
        &l10_att,
        &l10_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.10.attn_output.weight").unwrap().ggml_type),
        Some(&l10_in),
    )?;
    let l10_h_gpu = download(&s, &l10_h, n * h)?;
    let l10_h_cpu = l9_out_cpu
        .chunks_exact(h)
        .zip(l10_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l10_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer10_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l10_h_gpu,
        &l10_h_cpu,
    );

    let l10_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l10_h,
        &l10_zero,
        &upload(&s, &d, &l10_fn)?,
        &l10_f,
        &l10_f,
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
    let l10_f_gpu = download(&s, &l10_f, n * h)?;
    let l10_f_cpu = l10_h_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l10_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer10_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l10_f_gpu,
        &l10_f_cpu,
    );
    let i9 = r.get_tensor("blk.10.ffn_gate.weight").unwrap().dims[1] as usize;
    let l10_go = DeviceBuffer::alloc(d.clone(), n * i9 * 4)?;
    let l10_uo = DeviceBuffer::alloc(d.clone(), n * i9 * 4)?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l10_gw.data)?,
        &l10_f,
        &l10_go,
        h,
        i9,
        n,
        gemm_fmt(r.get_tensor("blk.10.ffn_gate.weight").unwrap().ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l10_uw.data)?,
        &l10_f,
        &l10_uo,
        h,
        i9,
        n,
        gemm_fmt(r.get_tensor("blk.10.ffn_up.weight").unwrap().ggml_type),
    )?;
    let l10_gg = download(&s, &l10_go, n * i9)?;
    let l10_ug = download(&s, &l10_uo, n * i9)?;
    let l10_gc = norm_cpu_for_ffn(&l10_f_cpu, &l10_gw, i9, h);
    let l10_uc = norm_cpu_for_ffn(&l10_f_cpu, &l10_uw, i9, h);
    report(
        "layer10_gate_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.10.ffn_gate.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i9}]"),
        n,
        &l10_gg,
        &l10_gc,
    );
    report(
        "layer10_up_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.10.ffn_up.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i9}]"),
        n,
        &l10_ug,
        &l10_uc,
    );
    let l10_sw_cpu = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &l10_gc[row * i9..(row + 1) * i9],
                &l10_uc[row * i9..(row + 1) * i9],
            )
        })
        .collect::<Vec<_>>();
    let l10_sw = DeviceBuffer::alloc(d.clone(), n * i9 * 4)?;
    let l10_zi = upload(&s, &d, &vec![0.; n * i9])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l10_go,
        &l10_zi,
        &l10_zi,
        &l10_uo,
        &l10_sw,
        eps,
        i9,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let l10_sw_gpu = download(&s, &l10_sw, n * i9)?;
    report(
        "layer10_swiglu",
        "F32",
        &format!("[{n},{i9}]"),
        n,
        &l10_sw_gpu,
        &l10_sw_cpu,
    );
    let l10_out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l10_dw.data)?,
        &l10_sw,
        &l10_out,
        i9,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.10.ffn_down.weight").unwrap().ggml_type),
        Some(&l10_h),
    )?;
    let l10_out_gpu = download(&s, &l10_out, n * h)?;
    let l10_out_cpu = (0..n)
        .flat_map(|row| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l10_dw, &l10_sw_cpu[row * i9..(row + 1) * i9]);
            l10_h_cpu[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer10_down_projection_residual",
        &format!(
            "{:?}",
            r.get_tensor("blk.10.ffn_down.weight").unwrap().ggml_type
        ),
        &format!("[{n},{h}]"),
        n,
        &l10_out_gpu,
        &l10_out_cpu,
    );
    // Continue from the real layer-10 post-down residual into block 11 only
    // through input RMSNorm and raw Q/K/V GEMMs.
    let l10w = vec32(&p, "blk.12.attn_norm.weight");
    let l10qw = tensor(&r, &p, "blk.12.attn_q.weight");
    let l10kw = tensor(&r, &p, "blk.12.attn_k.weight");
    let l10vw = tensor(&r, &p, "blk.12.attn_v.weight");
    assert_eq!(l10w.len(), h);
    report_q_metadata(&r, &p, "blk.12.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.12.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.12.attn_v.weight", h, nkv * hd);

    let l11_in = upload(&s, &d, &l10_out_gpu)?;
    let l11_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l11_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l11_in,
        &l11_zero,
        &upload(&s, &d, &l10w)?,
        &l11_norm,
        &l11_norm,
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
    let l11_norm_gpu = download(&s, &l11_norm, n * h)?;
    let l11_norm_cpu = l10_out_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l10w, eps))
        .collect::<Vec<_>>();
    report(
        "layer12_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l11_norm_gpu,
        &l11_norm_cpu,
    );

    let l11_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l11_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l11_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l11_q_info = r.get_tensor("blk.12.attn_q.weight").unwrap();
    let l11_k_info = r.get_tensor("blk.12.attn_k.weight").unwrap();
    let l11_v_info = r.get_tensor("blk.12.attn_v.weight").unwrap();
    let l11_qd = upload_raw(&s, &d, l10qw.data)?;
    let l11_kd = upload_raw(&s, &d, l10kw.data)?;
    let l11_vd = upload_raw(&s, &d, l10vw.data)?;
    g.gemm(
        &s,
        &l11_qd,
        &l11_norm,
        &l11_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l11_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l11_kd,
        &l11_norm,
        &l11_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l11_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l11_vd,
        &l11_norm,
        &l11_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l11_v_info.ggml_type),
    )?;
    let l11_q_gpu = download(&s, &l11_q, n * nh * hd)?;
    let l11_k_gpu = download(&s, &l11_k, n * nkv * hd)?;
    let l11_v_gpu = download(&s, &l11_v, n * nkv * hd)?;
    let l11_q_cpu = norm_cpu_for_ffn(&l11_norm_cpu, &l10qw, nh * hd, h);
    let l11_k_cpu = norm_cpu_for_ffn(&l11_norm_cpu, &l10kw, nkv * hd, h);
    let l11_v_cpu = norm_cpu_for_ffn(&l11_norm_cpu, &l10vw, nkv * hd, h);
    report(
        "layer12_q_raw_gemm",
        &format!("{:?}", l11_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l11_q_gpu,
        &l11_q_cpu,
    );
    report(
        "layer12_k_raw_gemm",
        &format!("{:?}", l11_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l11_k_gpu,
        &l11_k_cpu,
    );
    report(
        "layer12_v_raw_gemm",
        &format!("{:?}", l11_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l11_v_gpu,
        &l11_v_cpu,
    );

    let l11_qn = vec32(&p, "blk.12.attn_q_norm.weight");
    let l11_kn = vec32(&p, "blk.12.attn_k_norm.weight");
    let l11_wo = tensor(&r, &p, "blk.12.attn_output.weight");
    let l11_fn = vec32(&p, "blk.12.ffn_norm.weight");
    let l11_gw = tensor(&r, &p, "blk.12.ffn_gate.weight");
    let l11_uw = tensor(&r, &p, "blk.12.ffn_up.weight");
    let l11_dw = tensor(&r, &p, "blk.12.ffn_down.weight");
    report_q_metadata(&r, &p, "blk.12.attn_output.weight", nh * hd, h);

    let l11_qr = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l11_kr = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l11_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l11_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l11_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l11_q,
        &l11_zq,
        &upload(&s, &d, &l11_qn)?,
        &l11_qr,
        &l11_qr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l11_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l11_k,
        &l11_zk,
        &upload(&s, &d, &l11_kn)?,
        &l11_kr,
        &l11_kr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l11_pos),
        n * nkv,
        nkv,
    )?;
    let l11_qn_gpu = download(&s, &l11_qr, n * nh * hd)?;
    let l11_kn_gpu = download(&s, &l11_kr, n * nkv * hd)?;
    let mut l11_qn_cpu = Vec::with_capacity(n * nh * hd);
    let mut l11_kn_cpu = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            l11_qn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l11_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l11_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            l11_kn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l11_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l11_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer12_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l11_qn_gpu,
        &l11_qn_cpu,
    );
    report(
        "layer12_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l11_kn_gpu,
        &l11_kn_cpu,
    );
    report(
        "layer12_v_gemm",
        &format!("{:?}", l11_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l11_v_gpu,
        &l11_v_cpu,
    );

    let l11_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l11_bt = upload(&s, &d, &[0.])?;
    kv.append_kv(
        &s,
        &layout,
        &l11_pool,
        &upload(&s, &d, &l11_kn_gpu)?,
        &upload(&s, &d, &l11_v_gpu)?,
        &l11_bt,
        0,
        n,
    )?;
    let l11_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(
        &s,
        &upload(&s, &d, &l11_qn_gpu)?,
        &l11_pool,
        &l11_bt,
        &l11_att,
        nh,
        nkv,
        hd,
        64,
        n,
        0,
    )?;
    let l11_att_gpu = download(&s, &l11_att, n * nh * hd)?;
    let mut l11_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd].copy_from_slice(&l11_kn_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l11_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l11_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &l11_qn_cpu[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer12_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l11_att_gpu,
        &l11_att_cpu,
    );

    let l11_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l11_wo.data)?,
        &l11_att,
        &l11_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.12.attn_output.weight").unwrap().ggml_type),
        Some(&l11_in),
    )?;
    let l11_h_gpu = download(&s, &l11_h, n * h)?;
    let l11_h_cpu = l10_out_cpu
        .chunks_exact(h)
        .zip(l11_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l11_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer12_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l11_h_gpu,
        &l11_h_cpu,
    );

    let l11_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l11_h,
        &l11_zero,
        &upload(&s, &d, &l11_fn)?,
        &l11_f,
        &l11_f,
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
    let l11_f_gpu = download(&s, &l11_f, n * h)?;
    let l11_f_cpu = l11_h_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l11_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer12_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l11_f_gpu,
        &l11_f_cpu,
    );
    let i9 = r.get_tensor("blk.12.ffn_gate.weight").unwrap().dims[1] as usize;
    let l11_go = DeviceBuffer::alloc(d.clone(), n * i9 * 4)?;
    let l11_uo = DeviceBuffer::alloc(d.clone(), n * i9 * 4)?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l11_gw.data)?,
        &l11_f,
        &l11_go,
        h,
        i9,
        n,
        gemm_fmt(r.get_tensor("blk.12.ffn_gate.weight").unwrap().ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l11_uw.data)?,
        &l11_f,
        &l11_uo,
        h,
        i9,
        n,
        gemm_fmt(r.get_tensor("blk.12.ffn_up.weight").unwrap().ggml_type),
    )?;
    let l11_gg = download(&s, &l11_go, n * i9)?;
    let l11_ug = download(&s, &l11_uo, n * i9)?;
    let l11_gc = norm_cpu_for_ffn(&l11_f_cpu, &l11_gw, i9, h);
    let l11_uc = norm_cpu_for_ffn(&l11_f_cpu, &l11_uw, i9, h);
    report(
        "layer12_gate_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.12.ffn_gate.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i9}]"),
        n,
        &l11_gg,
        &l11_gc,
    );
    report(
        "layer12_up_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.12.ffn_up.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i9}]"),
        n,
        &l11_ug,
        &l11_uc,
    );
    let l11_sw_cpu = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &l11_gc[row * i9..(row + 1) * i9],
                &l11_uc[row * i9..(row + 1) * i9],
            )
        })
        .collect::<Vec<_>>();
    let l11_sw = DeviceBuffer::alloc(d.clone(), n * i9 * 4)?;
    let l11_zi = upload(&s, &d, &vec![0.; n * i9])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l11_go,
        &l11_zi,
        &l11_zi,
        &l11_uo,
        &l11_sw,
        eps,
        i9,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let l11_sw_gpu = download(&s, &l11_sw, n * i9)?;
    report(
        "layer12_swiglu",
        "F32",
        &format!("[{n},{i9}]"),
        n,
        &l11_sw_gpu,
        &l11_sw_cpu,
    );
    let l11_out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l11_dw.data)?,
        &l11_sw,
        &l11_out,
        i9,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.12.ffn_down.weight").unwrap().ggml_type),
        Some(&l11_h),
    )?;
    let l11_out_gpu = download(&s, &l11_out, n * h)?;
    let l11_out_cpu = (0..n)
        .flat_map(|row| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l11_dw, &l11_sw_cpu[row * i9..(row + 1) * i9]);
            l11_h_cpu[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer12_down_projection_residual",
        &format!(
            "{:?}",
            r.get_tensor("blk.12.ffn_down.weight").unwrap().ggml_type
        ),
        &format!("[{n},{h}]"),
        n,
        &l11_out_gpu,
        &l11_out_cpu,
    );
    // Continue from the real layer-11 post-down residual into block 12 only
    // through input RMSNorm and raw Q/K/V GEMMs.
    let l12w = vec32(&p, "blk.12.attn_norm.weight");
    let l12qw = tensor(&r, &p, "blk.12.attn_q.weight");
    let l12kw = tensor(&r, &p, "blk.12.attn_k.weight");
    let l12vw = tensor(&r, &p, "blk.12.attn_v.weight");
    assert_eq!(l12w.len(), h);
    report_q_metadata(&r, &p, "blk.12.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.12.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.12.attn_v.weight", h, nkv * hd);

    let l12_in = upload(&s, &d, &l11_out_gpu)?;
    let l12_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l12_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l12_in,
        &l12_zero,
        &upload(&s, &d, &l12w)?,
        &l12_norm,
        &l12_norm,
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
    let l12_norm_gpu = download(&s, &l12_norm, n * h)?;
    let l12_norm_cpu = l11_out_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l12w, eps))
        .collect::<Vec<_>>();
    report(
        "layer12_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l12_norm_gpu,
        &l12_norm_cpu,
    );

    let l12_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l12_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l12_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l12_q_info = r.get_tensor("blk.12.attn_q.weight").unwrap();
    let l12_k_info = r.get_tensor("blk.12.attn_k.weight").unwrap();
    let l12_v_info = r.get_tensor("blk.12.attn_v.weight").unwrap();
    let l12_qd = upload_raw(&s, &d, l12qw.data)?;
    let l12_kd = upload_raw(&s, &d, l12kw.data)?;
    let l12_vd = upload_raw(&s, &d, l12vw.data)?;
    g.gemm(
        &s,
        &l12_qd,
        &l12_norm,
        &l12_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l12_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l12_kd,
        &l12_norm,
        &l12_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l12_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l12_vd,
        &l12_norm,
        &l12_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l12_v_info.ggml_type),
    )?;
    let l12_q_gpu = download(&s, &l12_q, n * nh * hd)?;
    let l12_k_gpu = download(&s, &l12_k, n * nkv * hd)?;
    let l12_v_gpu = download(&s, &l12_v, n * nkv * hd)?;
    let l12_q_cpu = norm_cpu_for_ffn(&l12_norm_cpu, &l12qw, nh * hd, h);
    let l12_k_cpu = norm_cpu_for_ffn(&l12_norm_cpu, &l12kw, nkv * hd, h);
    let l12_v_cpu = norm_cpu_for_ffn(&l12_norm_cpu, &l12vw, nkv * hd, h);
    report(
        "layer12_q_raw_gemm",
        &format!("{:?}", l12_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l12_q_gpu,
        &l12_q_cpu,
    );
    report(
        "layer12_k_raw_gemm",
        &format!("{:?}", l12_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l12_k_gpu,
        &l12_k_cpu,
    );
    report(
        "layer12_v_raw_gemm",
        &format!("{:?}", l12_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l12_v_gpu,
        &l12_v_cpu,
    );

    let l12_qn = vec32(&p, "blk.12.attn_q_norm.weight");
    let l12_kn = vec32(&p, "blk.12.attn_k_norm.weight");
    let l12_wo = tensor(&r, &p, "blk.12.attn_output.weight");
    let l12_fn = vec32(&p, "blk.12.ffn_norm.weight");
    let l12_gw = tensor(&r, &p, "blk.12.ffn_gate.weight");
    let l12_uw = tensor(&r, &p, "blk.12.ffn_up.weight");
    let l12_dw = tensor(&r, &p, "blk.12.ffn_down.weight");
    report_q_metadata(&r, &p, "blk.12.attn_output.weight", nh * hd, h);

    let l12_qr = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l12_kr = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l12_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l12_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l12_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l12_q,
        &l12_zq,
        &upload(&s, &d, &l12_qn)?,
        &l12_qr,
        &l12_qr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l12_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l12_k,
        &l12_zk,
        &upload(&s, &d, &l12_kn)?,
        &l12_kr,
        &l12_kr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l12_pos),
        n * nkv,
        nkv,
    )?;
    let l12_qn_gpu = download(&s, &l12_qr, n * nh * hd)?;
    let l12_kn_gpu = download(&s, &l12_kr, n * nkv * hd)?;
    let mut l12_qn_cpu = Vec::with_capacity(n * nh * hd);
    let mut l12_kn_cpu = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            l12_qn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l12_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l12_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            l12_kn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l12_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l12_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer12_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l12_qn_gpu,
        &l12_qn_cpu,
    );
    report(
        "layer12_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l12_kn_gpu,
        &l12_kn_cpu,
    );
    report(
        "layer12_v_gemm",
        &format!("{:?}", l12_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l12_v_gpu,
        &l12_v_cpu,
    );

    let l12_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l12_bt = upload(&s, &d, &[0.])?;
    kv.append_kv(
        &s,
        &layout,
        &l12_pool,
        &upload(&s, &d, &l12_kn_gpu)?,
        &upload(&s, &d, &l12_v_gpu)?,
        &l12_bt,
        0,
        n,
    )?;
    let l12_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(
        &s,
        &upload(&s, &d, &l12_qn_gpu)?,
        &l12_pool,
        &l12_bt,
        &l12_att,
        nh,
        nkv,
        hd,
        64,
        n,
        0,
    )?;
    let l12_att_gpu = download(&s, &l12_att, n * nh * hd)?;
    let mut l12_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd].copy_from_slice(&l12_kn_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l12_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l12_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &l12_qn_cpu[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer12_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l12_att_gpu,
        &l12_att_cpu,
    );

    let l12_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l12_wo.data)?,
        &l12_att,
        &l12_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.12.attn_output.weight").unwrap().ggml_type),
        Some(&l12_in),
    )?;
    let l12_h_gpu = download(&s, &l12_h, n * h)?;
    let l12_h_cpu = l11_out_cpu
        .chunks_exact(h)
        .zip(l12_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l12_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer12_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l12_h_gpu,
        &l12_h_cpu,
    );

    let l12_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l12_h,
        &l12_zero,
        &upload(&s, &d, &l12_fn)?,
        &l12_f,
        &l12_f,
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
    let l12_f_gpu = download(&s, &l12_f, n * h)?;
    let l12_f_cpu = l12_h_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l12_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer12_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l12_f_gpu,
        &l12_f_cpu,
    );
    let i9 = r.get_tensor("blk.12.ffn_gate.weight").unwrap().dims[1] as usize;
    let l12_go = DeviceBuffer::alloc(d.clone(), n * i9 * 4)?;
    let l12_uo = DeviceBuffer::alloc(d.clone(), n * i9 * 4)?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l12_gw.data)?,
        &l12_f,
        &l12_go,
        h,
        i9,
        n,
        gemm_fmt(r.get_tensor("blk.12.ffn_gate.weight").unwrap().ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l12_uw.data)?,
        &l12_f,
        &l12_uo,
        h,
        i9,
        n,
        gemm_fmt(r.get_tensor("blk.12.ffn_up.weight").unwrap().ggml_type),
    )?;
    let l12_gg = download(&s, &l12_go, n * i9)?;
    let l12_ug = download(&s, &l12_uo, n * i9)?;
    let l12_gc = norm_cpu_for_ffn(&l12_f_cpu, &l12_gw, i9, h);
    let l12_uc = norm_cpu_for_ffn(&l12_f_cpu, &l12_uw, i9, h);
    report(
        "layer12_gate_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.12.ffn_gate.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i9}]"),
        n,
        &l12_gg,
        &l12_gc,
    );
    report(
        "layer12_up_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.12.ffn_up.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i9}]"),
        n,
        &l12_ug,
        &l12_uc,
    );
    let l12_sw_cpu = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &l12_gc[row * i9..(row + 1) * i9],
                &l12_uc[row * i9..(row + 1) * i9],
            )
        })
        .collect::<Vec<_>>();
    let l12_sw = DeviceBuffer::alloc(d.clone(), n * i9 * 4)?;
    let l12_zi = upload(&s, &d, &vec![0.; n * i9])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l12_go,
        &l12_zi,
        &l12_zi,
        &l12_uo,
        &l12_sw,
        eps,
        i9,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let l12_sw_gpu = download(&s, &l12_sw, n * i9)?;
    report(
        "layer12_swiglu",
        "F32",
        &format!("[{n},{i9}]"),
        n,
        &l12_sw_gpu,
        &l12_sw_cpu,
    );
    let l12_out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l12_dw.data)?,
        &l12_sw,
        &l12_out,
        i9,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.12.ffn_down.weight").unwrap().ggml_type),
        Some(&l12_h),
    )?;
    let l12_out_gpu = download(&s, &l12_out, n * h)?;
    let l12_out_cpu = (0..n)
        .flat_map(|row| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l12_dw, &l12_sw_cpu[row * i9..(row + 1) * i9]);
            l12_h_cpu[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer12_down_projection_residual",
        &format!(
            "{:?}",
            r.get_tensor("blk.12.ffn_down.weight").unwrap().ggml_type
        ),
        &format!("[{n},{h}]"),
        n,
        &l12_out_gpu,
        &l12_out_cpu,
    );
    // Continue from the real layer-12 post-down residual into block 13 only
    // through input RMSNorm and raw Q/K/V GEMMs.
    let l13w = vec32(&p, "blk.13.attn_norm.weight");
    let l13qw = tensor(&r, &p, "blk.13.attn_q.weight");
    let l13kw = tensor(&r, &p, "blk.13.attn_k.weight");
    let l13vw = tensor(&r, &p, "blk.13.attn_v.weight");
    assert_eq!(l13w.len(), h);
    report_q_metadata(&r, &p, "blk.13.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.13.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.13.attn_v.weight", h, nkv * hd);

    let l13_in = upload(&s, &d, &l12_out_gpu)?;
    let l13_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l13_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l13_in,
        &l13_zero,
        &upload(&s, &d, &l13w)?,
        &l13_norm,
        &l13_norm,
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
    let l13_norm_gpu = download(&s, &l13_norm, n * h)?;
    let l13_norm_cpu = l12_out_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l13w, eps))
        .collect::<Vec<_>>();
    report(
        "layer13_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l13_norm_gpu,
        &l13_norm_cpu,
    );

    let l13_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l13_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l13_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l13_q_info = r.get_tensor("blk.13.attn_q.weight").unwrap();
    let l13_k_info = r.get_tensor("blk.13.attn_k.weight").unwrap();
    let l13_v_info = r.get_tensor("blk.13.attn_v.weight").unwrap();
    let l13_qd = upload_raw(&s, &d, l13qw.data)?;
    let l13_kd = upload_raw(&s, &d, l13kw.data)?;
    let l13_vd = upload_raw(&s, &d, l13vw.data)?;
    g.gemm(
        &s,
        &l13_qd,
        &l13_norm,
        &l13_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l13_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l13_kd,
        &l13_norm,
        &l13_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l13_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l13_vd,
        &l13_norm,
        &l13_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l13_v_info.ggml_type),
    )?;
    let l13_q_gpu = download(&s, &l13_q, n * nh * hd)?;
    let l13_k_gpu = download(&s, &l13_k, n * nkv * hd)?;
    let l13_v_gpu = download(&s, &l13_v, n * nkv * hd)?;
    let l13_q_cpu = norm_cpu_for_ffn(&l13_norm_cpu, &l13qw, nh * hd, h);
    let l13_k_cpu = norm_cpu_for_ffn(&l13_norm_cpu, &l13kw, nkv * hd, h);
    let l13_v_cpu = norm_cpu_for_ffn(&l13_norm_cpu, &l13vw, nkv * hd, h);
    report(
        "layer13_q_raw_gemm",
        &format!("{:?}", l13_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l13_q_gpu,
        &l13_q_cpu,
    );
    report(
        "layer13_k_raw_gemm",
        &format!("{:?}", l13_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l13_k_gpu,
        &l13_k_cpu,
    );
    report(
        "layer13_v_raw_gemm",
        &format!("{:?}", l13_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l13_v_gpu,
        &l13_v_cpu,
    );

    let l13_qn = vec32(&p, "blk.13.attn_q_norm.weight");
    let l13_kn = vec32(&p, "blk.13.attn_k_norm.weight");
    let l13_wo = tensor(&r, &p, "blk.13.attn_output.weight");
    let l13_fn = vec32(&p, "blk.13.ffn_norm.weight");
    let l13_gw = tensor(&r, &p, "blk.13.ffn_gate.weight");
    let l13_uw = tensor(&r, &p, "blk.13.ffn_up.weight");
    let l13_dw = tensor(&r, &p, "blk.13.ffn_down.weight");
    report_q_metadata(&r, &p, "blk.13.attn_output.weight", nh * hd, h);

    let l13_qr = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l13_kr = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l13_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l13_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l13_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l13_q,
        &l13_zq,
        &upload(&s, &d, &l13_qn)?,
        &l13_qr,
        &l13_qr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l13_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l13_k,
        &l13_zk,
        &upload(&s, &d, &l13_kn)?,
        &l13_kr,
        &l13_kr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l13_pos),
        n * nkv,
        nkv,
    )?;
    let l13_qn_gpu = download(&s, &l13_qr, n * nh * hd)?;
    let l13_kn_gpu = download(&s, &l13_kr, n * nkv * hd)?;
    let mut l13_qn_cpu = Vec::with_capacity(n * nh * hd);
    let mut l13_kn_cpu = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            l13_qn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l13_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l13_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            l13_kn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l13_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l13_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer13_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l13_qn_gpu,
        &l13_qn_cpu,
    );
    report(
        "layer13_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l13_kn_gpu,
        &l13_kn_cpu,
    );
    report(
        "layer13_v_gemm",
        &format!("{:?}", l13_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l13_v_gpu,
        &l13_v_cpu,
    );

    let l13_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l13_bt = upload(&s, &d, &[0.])?;
    kv.append_kv(
        &s,
        &layout,
        &l13_pool,
        &upload(&s, &d, &l13_kn_gpu)?,
        &upload(&s, &d, &l13_v_gpu)?,
        &l13_bt,
        0,
        n,
    )?;
    let l13_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(
        &s,
        &upload(&s, &d, &l13_qn_gpu)?,
        &l13_pool,
        &l13_bt,
        &l13_att,
        nh,
        nkv,
        hd,
        64,
        n,
        0,
    )?;
    let l13_att_gpu = download(&s, &l13_att, n * nh * hd)?;
    let mut l13_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd].copy_from_slice(&l13_kn_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l13_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l13_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &l13_qn_cpu[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer13_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l13_att_gpu,
        &l13_att_cpu,
    );

    let l13_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l13_wo.data)?,
        &l13_att,
        &l13_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.13.attn_output.weight").unwrap().ggml_type),
        Some(&l13_in),
    )?;
    let l13_h_gpu = download(&s, &l13_h, n * h)?;
    let l13_h_cpu = l12_out_cpu
        .chunks_exact(h)
        .zip(l13_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l13_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer13_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l13_h_gpu,
        &l13_h_cpu,
    );

    let l13_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l13_h,
        &l13_zero,
        &upload(&s, &d, &l13_fn)?,
        &l13_f,
        &l13_f,
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
    let l13_f_gpu = download(&s, &l13_f, n * h)?;
    let l13_f_cpu = l13_h_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l13_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer13_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l13_f_gpu,
        &l13_f_cpu,
    );
    let i10 = r.get_tensor("blk.13.ffn_gate.weight").unwrap().dims[1] as usize;
    let l13_go = DeviceBuffer::alloc(d.clone(), n * i10 * 4)?;
    let l13_uo = DeviceBuffer::alloc(d.clone(), n * i10 * 4)?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l13_gw.data)?,
        &l13_f,
        &l13_go,
        h,
        i10,
        n,
        gemm_fmt(r.get_tensor("blk.13.ffn_gate.weight").unwrap().ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l13_uw.data)?,
        &l13_f,
        &l13_uo,
        h,
        i10,
        n,
        gemm_fmt(r.get_tensor("blk.13.ffn_up.weight").unwrap().ggml_type),
    )?;
    let l13_gg = download(&s, &l13_go, n * i10)?;
    let l13_ug = download(&s, &l13_uo, n * i10)?;
    let l13_gc = norm_cpu_for_ffn(&l13_f_cpu, &l13_gw, i10, h);
    let l13_uc = norm_cpu_for_ffn(&l13_f_cpu, &l13_uw, i10, h);
    report(
        "layer13_gate_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.13.ffn_gate.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i10}]"),
        n,
        &l13_gg,
        &l13_gc,
    );
    report(
        "layer13_up_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.13.ffn_up.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i10}]"),
        n,
        &l13_ug,
        &l13_uc,
    );
    let l13_sw_cpu = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &l13_gc[row * i10..(row + 1) * i10],
                &l13_uc[row * i10..(row + 1) * i10],
            )
        })
        .collect::<Vec<_>>();
    let l13_sw = DeviceBuffer::alloc(d.clone(), n * i10 * 4)?;
    let l13_zi = upload(&s, &d, &vec![0.; n * i10])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l13_go,
        &l13_zi,
        &l13_zi,
        &l13_uo,
        &l13_sw,
        eps,
        i10,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let l13_sw_gpu = download(&s, &l13_sw, n * i10)?;
    report(
        "layer13_swiglu",
        "F32",
        &format!("[{n},{i10}]"),
        n,
        &l13_sw_gpu,
        &l13_sw_cpu,
    );
    let l13_out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l13_dw.data)?,
        &l13_sw,
        &l13_out,
        i10,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.13.ffn_down.weight").unwrap().ggml_type),
        Some(&l13_h),
    )?;
    let l13_out_gpu = download(&s, &l13_out, n * h)?;
    let l13_out_cpu = (0..n)
        .flat_map(|row| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l13_dw, &l13_sw_cpu[row * i10..(row + 1) * i10]);
            l13_h_cpu[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer13_down_projection_residual",
        &format!(
            "{:?}",
            r.get_tensor("blk.13.ffn_down.weight").unwrap().ggml_type
        ),
        &format!("[{n},{h}]"),
        n,
        &l13_out_gpu,
        &l13_out_cpu,
    );
    // Continue from the real layer-13 post-down residual into block 14 only
    // through input RMSNorm and raw Q/K/V GEMMs.
    let l14w = vec32(&p, "blk.14.attn_norm.weight");
    let l14qw = tensor(&r, &p, "blk.14.attn_q.weight");
    let l14kw = tensor(&r, &p, "blk.14.attn_k.weight");
    let l14vw = tensor(&r, &p, "blk.14.attn_v.weight");
    assert_eq!(l14w.len(), h);
    report_q_metadata(&r, &p, "blk.14.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.14.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.14.attn_v.weight", h, nkv * hd);

    let l14_in = upload(&s, &d, &l12_out_gpu)?;
    let l14_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l14_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l14_in,
        &l14_zero,
        &upload(&s, &d, &l14w)?,
        &l14_norm,
        &l14_norm,
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
    let l14_norm_gpu = download(&s, &l14_norm, n * h)?;
    let l14_norm_cpu = l12_out_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l14w, eps))
        .collect::<Vec<_>>();
    report(
        "layer14_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l14_norm_gpu,
        &l14_norm_cpu,
    );

    let l14_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l14_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l14_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l14_q_info = r.get_tensor("blk.14.attn_q.weight").unwrap();
    let l14_k_info = r.get_tensor("blk.14.attn_k.weight").unwrap();
    let l14_v_info = r.get_tensor("blk.14.attn_v.weight").unwrap();
    let l14_qd = upload_raw(&s, &d, l14qw.data)?;
    let l14_kd = upload_raw(&s, &d, l14kw.data)?;
    let l14_vd = upload_raw(&s, &d, l14vw.data)?;
    g.gemm(
        &s,
        &l14_qd,
        &l14_norm,
        &l14_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l14_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l14_kd,
        &l14_norm,
        &l14_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l14_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l14_vd,
        &l14_norm,
        &l14_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l14_v_info.ggml_type),
    )?;
    let l14_q_gpu = download(&s, &l14_q, n * nh * hd)?;
    let l14_k_gpu = download(&s, &l14_k, n * nkv * hd)?;
    let l14_v_gpu = download(&s, &l14_v, n * nkv * hd)?;
    let l14_q_cpu = norm_cpu_for_ffn(&l14_norm_cpu, &l14qw, nh * hd, h);
    let l14_k_cpu = norm_cpu_for_ffn(&l14_norm_cpu, &l14kw, nkv * hd, h);
    let l14_v_cpu = norm_cpu_for_ffn(&l14_norm_cpu, &l14vw, nkv * hd, h);
    report(
        "layer14_q_raw_gemm",
        &format!("{:?}", l14_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l14_q_gpu,
        &l14_q_cpu,
    );
    report(
        "layer14_k_raw_gemm",
        &format!("{:?}", l14_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l14_k_gpu,
        &l14_k_cpu,
    );
    report(
        "layer14_v_raw_gemm",
        &format!("{:?}", l14_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l14_v_gpu,
        &l14_v_cpu,
    );

    let l14_qn = vec32(&p, "blk.14.attn_q_norm.weight");
    let l14_kn = vec32(&p, "blk.14.attn_k_norm.weight");
    let l14_wo = tensor(&r, &p, "blk.14.attn_output.weight");
    let l14_fn = vec32(&p, "blk.14.ffn_norm.weight");
    let l14_gw = tensor(&r, &p, "blk.14.ffn_gate.weight");
    let l14_uw = tensor(&r, &p, "blk.14.ffn_up.weight");
    let l14_dw = tensor(&r, &p, "blk.14.ffn_down.weight");
    report_q_metadata(&r, &p, "blk.14.attn_output.weight", nh * hd, h);

    let l14_qr = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l14_kr = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l14_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l14_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l14_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l14_q,
        &l14_zq,
        &upload(&s, &d, &l14_qn)?,
        &l14_qr,
        &l14_qr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l14_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l14_k,
        &l14_zk,
        &upload(&s, &d, &l14_kn)?,
        &l14_kr,
        &l14_kr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l14_pos),
        n * nkv,
        nkv,
    )?;
    let l14_qn_gpu = download(&s, &l14_qr, n * nh * hd)?;
    let l14_kn_gpu = download(&s, &l14_kr, n * nkv * hd)?;
    let mut l14_qn_cpu = Vec::with_capacity(n * nh * hd);
    let mut l14_kn_cpu = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            l14_qn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l14_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l14_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            l14_kn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l14_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l14_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer14_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l14_qn_gpu,
        &l14_qn_cpu,
    );
    report(
        "layer14_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l14_kn_gpu,
        &l14_kn_cpu,
    );
    report(
        "layer14_v_gemm",
        &format!("{:?}", l14_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l14_v_gpu,
        &l14_v_cpu,
    );

    let l14_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l14_bt = upload(&s, &d, &[0.])?;
    kv.append_kv(
        &s,
        &layout,
        &l14_pool,
        &upload(&s, &d, &l14_kn_gpu)?,
        &upload(&s, &d, &l14_v_gpu)?,
        &l14_bt,
        0,
        n,
    )?;
    let l14_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(
        &s,
        &upload(&s, &d, &l14_qn_gpu)?,
        &l14_pool,
        &l14_bt,
        &l14_att,
        nh,
        nkv,
        hd,
        64,
        n,
        0,
    )?;
    let l14_att_gpu = download(&s, &l14_att, n * nh * hd)?;
    let mut l14_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd].copy_from_slice(&l14_kn_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l14_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l14_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &l14_qn_cpu[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer14_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l14_att_gpu,
        &l14_att_cpu,
    );

    let l14_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l14_wo.data)?,
        &l14_att,
        &l14_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.14.attn_output.weight").unwrap().ggml_type),
        Some(&l14_in),
    )?;
    let l14_h_gpu = download(&s, &l14_h, n * h)?;
    let l14_h_cpu = l12_out_cpu
        .chunks_exact(h)
        .zip(l14_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l14_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer14_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l14_h_gpu,
        &l14_h_cpu,
    );

    let l14_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l14_h,
        &l14_zero,
        &upload(&s, &d, &l14_fn)?,
        &l14_f,
        &l14_f,
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
    let l14_f_gpu = download(&s, &l14_f, n * h)?;
    let l14_f_cpu = l14_h_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l14_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer14_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l14_f_gpu,
        &l14_f_cpu,
    );
    let i10 = r.get_tensor("blk.14.ffn_gate.weight").unwrap().dims[1] as usize;
    let l14_go = DeviceBuffer::alloc(d.clone(), n * i10 * 4)?;
    let l14_uo = DeviceBuffer::alloc(d.clone(), n * i10 * 4)?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l14_gw.data)?,
        &l14_f,
        &l14_go,
        h,
        i10,
        n,
        gemm_fmt(r.get_tensor("blk.14.ffn_gate.weight").unwrap().ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l14_uw.data)?,
        &l14_f,
        &l14_uo,
        h,
        i10,
        n,
        gemm_fmt(r.get_tensor("blk.14.ffn_up.weight").unwrap().ggml_type),
    )?;
    let l14_gg = download(&s, &l14_go, n * i10)?;
    let l14_ug = download(&s, &l14_uo, n * i10)?;
    let l14_gc = norm_cpu_for_ffn(&l14_f_cpu, &l14_gw, i10, h);
    let l14_uc = norm_cpu_for_ffn(&l14_f_cpu, &l14_uw, i10, h);
    report(
        "layer14_gate_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.14.ffn_gate.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i10}]"),
        n,
        &l14_gg,
        &l14_gc,
    );
    report(
        "layer14_up_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.14.ffn_up.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i10}]"),
        n,
        &l14_ug,
        &l14_uc,
    );
    let l14_sw_cpu = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &l14_gc[row * i10..(row + 1) * i10],
                &l14_uc[row * i10..(row + 1) * i10],
            )
        })
        .collect::<Vec<_>>();
    let l14_sw = DeviceBuffer::alloc(d.clone(), n * i10 * 4)?;
    let l14_zi = upload(&s, &d, &vec![0.; n * i10])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l14_go,
        &l14_zi,
        &l14_zi,
        &l14_uo,
        &l14_sw,
        eps,
        i10,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let l14_sw_gpu = download(&s, &l14_sw, n * i10)?;
    report(
        "layer14_swiglu",
        "F32",
        &format!("[{n},{i10}]"),
        n,
        &l14_sw_gpu,
        &l14_sw_cpu,
    );
    let l14_out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l14_dw.data)?,
        &l14_sw,
        &l14_out,
        i10,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.14.ffn_down.weight").unwrap().ggml_type),
        Some(&l14_h),
    )?;
    let l14_out_gpu = download(&s, &l14_out, n * h)?;
    let l14_out_cpu = (0..n)
        .flat_map(|row| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l14_dw, &l14_sw_cpu[row * i10..(row + 1) * i10]);
            l14_h_cpu[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer14_down_projection_residual",
        &format!(
            "{:?}",
            r.get_tensor("blk.14.ffn_down.weight").unwrap().ggml_type
        ),
        &format!("[{n},{h}]"),
        n,
        &l14_out_gpu,
        &l14_out_cpu,
    );
    // Continue from the real layer-14 post-down residual into block 15
    // through input RMSNorm and raw Q/K/V GEMMs.
    let l15w = vec32(&p, "blk.15.attn_norm.weight");
    let l15qw = tensor(&r, &p, "blk.15.attn_q.weight");
    let l15kw = tensor(&r, &p, "blk.15.attn_k.weight");
    let l15vw = tensor(&r, &p, "blk.15.attn_v.weight");
    assert_eq!(l15w.len(), h);
    report_q_metadata(&r, &p, "blk.15.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.15.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.15.attn_v.weight", h, nkv * hd);

    let l15_in = upload(&s, &d, &l14_out_gpu)?;
    let l15_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l15_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l15_in,
        &l15_zero,
        &upload(&s, &d, &l15w)?,
        &l15_norm,
        &l15_norm,
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
    let l15_norm_gpu = download(&s, &l15_norm, n * h)?;
    let l15_norm_cpu = l14_out_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l15w, eps))
        .collect::<Vec<_>>();
    report(
        "layer15_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l15_norm_gpu,
        &l15_norm_cpu,
    );

    let l15_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l15_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l15_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l15_q_info = r.get_tensor("blk.15.attn_q.weight").unwrap();
    let l15_k_info = r.get_tensor("blk.15.attn_k.weight").unwrap();
    let l15_v_info = r.get_tensor("blk.15.attn_v.weight").unwrap();
    let l15_qd = upload_raw(&s, &d, l15qw.data)?;
    let l15_kd = upload_raw(&s, &d, l15kw.data)?;
    let l15_vd = upload_raw(&s, &d, l15vw.data)?;
    g.gemm(
        &s,
        &l15_qd,
        &l15_norm,
        &l15_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l15_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l15_kd,
        &l15_norm,
        &l15_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l15_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l15_vd,
        &l15_norm,
        &l15_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l15_v_info.ggml_type),
    )?;
    let l15_q_gpu = download(&s, &l15_q, n * nh * hd)?;
    let l15_k_gpu = download(&s, &l15_k, n * nkv * hd)?;
    let l15_v_gpu = download(&s, &l15_v, n * nkv * hd)?;
    let l15_q_cpu = norm_cpu_for_ffn(&l15_norm_cpu, &l15qw, nh * hd, h);
    let l15_k_cpu = norm_cpu_for_ffn(&l15_norm_cpu, &l15kw, nkv * hd, h);
    let l15_v_cpu = norm_cpu_for_ffn(&l15_norm_cpu, &l15vw, nkv * hd, h);
    report(
        "layer15_q_raw_gemm",
        &format!("{:?}", l15_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l15_q_gpu,
        &l15_q_cpu,
    );
    report(
        "layer15_k_raw_gemm",
        &format!("{:?}", l15_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l15_k_gpu,
        &l15_k_cpu,
    );
    report(
        "layer15_v_raw_gemm",
        &format!("{:?}", l15_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l15_v_gpu,
        &l15_v_cpu,
    );

    let l15_qn = vec32(&p, "blk.15.attn_q_norm.weight");
    let l15_kn = vec32(&p, "blk.15.attn_k_norm.weight");
    let l15_wo = tensor(&r, &p, "blk.15.attn_output.weight");
    let l15_fn = vec32(&p, "blk.15.ffn_norm.weight");
    let l15_gw = tensor(&r, &p, "blk.15.ffn_gate.weight");
    let l15_uw = tensor(&r, &p, "blk.15.ffn_up.weight");
    let l15_dw = tensor(&r, &p, "blk.15.ffn_down.weight");
    report_q_metadata(&r, &p, "blk.15.attn_output.weight", nh * hd, h);

    let l15_qr = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l15_kr = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l15_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l15_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l15_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l15_q,
        &l15_zq,
        &upload(&s, &d, &l15_qn)?,
        &l15_qr,
        &l15_qr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l15_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l15_k,
        &l15_zk,
        &upload(&s, &d, &l15_kn)?,
        &l15_kr,
        &l15_kr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l15_pos),
        n * nkv,
        nkv,
    )?;
    let l15_qn_gpu = download(&s, &l15_qr, n * nh * hd)?;
    let l15_kn_gpu = download(&s, &l15_kr, n * nkv * hd)?;
    let mut l15_qn_cpu = Vec::with_capacity(n * nh * hd);
    let mut l15_kn_cpu = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            l15_qn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l15_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l15_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            l15_kn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l15_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l15_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer15_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l15_qn_gpu,
        &l15_qn_cpu,
    );
    report(
        "layer15_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l15_kn_gpu,
        &l15_kn_cpu,
    );
    report(
        "layer15_v_gemm",
        &format!("{:?}", l15_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l15_v_gpu,
        &l15_v_cpu,
    );

    let l15_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l15_bt = upload(&s, &d, &[0.])?;
    kv.append_kv(
        &s,
        &layout,
        &l15_pool,
        &upload(&s, &d, &l15_kn_gpu)?,
        &upload(&s, &d, &l15_v_gpu)?,
        &l15_bt,
        0,
        n,
    )?;
    let l15_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(
        &s,
        &upload(&s, &d, &l15_qn_gpu)?,
        &l15_pool,
        &l15_bt,
        &l15_att,
        nh,
        nkv,
        hd,
        64,
        n,
        0,
    )?;
    let l15_att_gpu = download(&s, &l15_att, n * nh * hd)?;
    let mut l15_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd].copy_from_slice(&l15_kn_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l15_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l15_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &l15_qn_cpu[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer15_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l15_att_gpu,
        &l15_att_cpu,
    );

    let l15_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l15_wo.data)?,
        &l15_att,
        &l15_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.15.attn_output.weight").unwrap().ggml_type),
        Some(&l15_in),
    )?;
    let l15_h_gpu = download(&s, &l15_h, n * h)?;
    let l15_h_cpu = l14_out_cpu
        .chunks_exact(h)
        .zip(l15_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l15_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer15_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l15_h_gpu,
        &l15_h_cpu,
    );

    let l15_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l15_h,
        &l15_zero,
        &upload(&s, &d, &l15_fn)?,
        &l15_f,
        &l15_f,
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
    let l15_f_gpu = download(&s, &l15_f, n * h)?;
    let l15_f_cpu = l15_h_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l15_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer15_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l15_f_gpu,
        &l15_f_cpu,
    );
    let i10 = r.get_tensor("blk.15.ffn_gate.weight").unwrap().dims[1] as usize;
    let l15_go = DeviceBuffer::alloc(d.clone(), n * i10 * 4)?;
    let l15_uo = DeviceBuffer::alloc(d.clone(), n * i10 * 4)?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l15_gw.data)?,
        &l15_f,
        &l15_go,
        h,
        i10,
        n,
        gemm_fmt(r.get_tensor("blk.15.ffn_gate.weight").unwrap().ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l15_uw.data)?,
        &l15_f,
        &l15_uo,
        h,
        i10,
        n,
        gemm_fmt(r.get_tensor("blk.15.ffn_up.weight").unwrap().ggml_type),
    )?;
    let l15_gg = download(&s, &l15_go, n * i10)?;
    let l15_ug = download(&s, &l15_uo, n * i10)?;
    let l15_gc = norm_cpu_for_ffn(&l15_f_cpu, &l15_gw, i10, h);
    let l15_uc = norm_cpu_for_ffn(&l15_f_cpu, &l15_uw, i10, h);
    report(
        "layer15_gate_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.15.ffn_gate.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i10}]"),
        n,
        &l15_gg,
        &l15_gc,
    );
    report(
        "layer15_up_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.15.ffn_up.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i10}]"),
        n,
        &l15_ug,
        &l15_uc,
    );
    let l15_sw_cpu = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &l15_gc[row * i10..(row + 1) * i10],
                &l15_uc[row * i10..(row + 1) * i10],
            )
        })
        .collect::<Vec<_>>();
    let l15_sw = DeviceBuffer::alloc(d.clone(), n * i10 * 4)?;
    let l15_zi = upload(&s, &d, &vec![0.; n * i10])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l15_go,
        &l15_zi,
        &l15_zi,
        &l15_uo,
        &l15_sw,
        eps,
        i10,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let l15_sw_gpu = download(&s, &l15_sw, n * i10)?;
    report(
        "layer15_swiglu",
        "F32",
        &format!("[{n},{i10}]"),
        n,
        &l15_sw_gpu,
        &l15_sw_cpu,
    );
    let l15_out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l15_dw.data)?,
        &l15_sw,
        &l15_out,
        i10,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.15.ffn_down.weight").unwrap().ggml_type),
        Some(&l15_h),
    )?;
    let l15_out_gpu = download(&s, &l15_out, n * h)?;
    let l15_out_cpu = (0..n)
        .flat_map(|row| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l15_dw, &l15_sw_cpu[row * i10..(row + 1) * i10]);
            l15_h_cpu[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer15_down_projection_residual",
        &format!(
            "{:?}",
            r.get_tensor("blk.15.ffn_down.weight").unwrap().ggml_type
        ),
        &format!("[{n},{h}]"),
        n,
        &l15_out_gpu,
        &l15_out_cpu,
    );
    // Continue from the real layer-15 post-down residual into block 16
    // through input RMSNorm and raw Q/K/V GEMMs.
    let l16w = vec32(&p, "blk.16.attn_norm.weight");
    let l16qw = tensor(&r, &p, "blk.16.attn_q.weight");
    let l16kw = tensor(&r, &p, "blk.16.attn_k.weight");
    let l16vw = tensor(&r, &p, "blk.16.attn_v.weight");
    assert_eq!(l16w.len(), h);
    report_q_metadata(&r, &p, "blk.16.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.16.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.16.attn_v.weight", h, nkv * hd);

    let l16_in = upload(&s, &d, &l14_out_gpu)?;
    let l16_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l16_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l16_in,
        &l16_zero,
        &upload(&s, &d, &l16w)?,
        &l16_norm,
        &l16_norm,
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
    let l16_norm_gpu = download(&s, &l16_norm, n * h)?;
    let l16_norm_cpu = l14_out_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l16w, eps))
        .collect::<Vec<_>>();
    report(
        "layer16_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l16_norm_gpu,
        &l16_norm_cpu,
    );

    let l16_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l16_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l16_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l16_q_info = r.get_tensor("blk.16.attn_q.weight").unwrap();
    let l16_k_info = r.get_tensor("blk.16.attn_k.weight").unwrap();
    let l16_v_info = r.get_tensor("blk.16.attn_v.weight").unwrap();
    let l16_qd = upload_raw(&s, &d, l16qw.data)?;
    let l16_kd = upload_raw(&s, &d, l16kw.data)?;
    let l16_vd = upload_raw(&s, &d, l16vw.data)?;
    g.gemm(
        &s,
        &l16_qd,
        &l16_norm,
        &l16_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l16_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l16_kd,
        &l16_norm,
        &l16_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l16_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l16_vd,
        &l16_norm,
        &l16_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l16_v_info.ggml_type),
    )?;
    let l16_q_gpu = download(&s, &l16_q, n * nh * hd)?;
    let l16_k_gpu = download(&s, &l16_k, n * nkv * hd)?;
    let l16_v_gpu = download(&s, &l16_v, n * nkv * hd)?;
    let l16_q_cpu = norm_cpu_for_ffn(&l16_norm_cpu, &l16qw, nh * hd, h);
    let l16_k_cpu = norm_cpu_for_ffn(&l16_norm_cpu, &l16kw, nkv * hd, h);
    let l16_v_cpu = norm_cpu_for_ffn(&l16_norm_cpu, &l16vw, nkv * hd, h);
    report(
        "layer16_q_raw_gemm",
        &format!("{:?}", l16_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l16_q_gpu,
        &l16_q_cpu,
    );
    report(
        "layer16_k_raw_gemm",
        &format!("{:?}", l16_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l16_k_gpu,
        &l16_k_cpu,
    );
    report(
        "layer16_v_raw_gemm",
        &format!("{:?}", l16_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l16_v_gpu,
        &l16_v_cpu,
    );

    let l16_qn = vec32(&p, "blk.16.attn_q_norm.weight");
    let l16_kn = vec32(&p, "blk.16.attn_k_norm.weight");
    let l16_wo = tensor(&r, &p, "blk.16.attn_output.weight");
    let l16_fn = vec32(&p, "blk.16.ffn_norm.weight");
    let l16_gw = tensor(&r, &p, "blk.16.ffn_gate.weight");
    let l16_uw = tensor(&r, &p, "blk.16.ffn_up.weight");
    let l16_dw = tensor(&r, &p, "blk.16.ffn_down.weight");
    report_q_metadata(&r, &p, "blk.16.attn_output.weight", nh * hd, h);

    let l16_qr = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l16_kr = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l16_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l16_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l16_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l16_q,
        &l16_zq,
        &upload(&s, &d, &l16_qn)?,
        &l16_qr,
        &l16_qr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l16_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l16_k,
        &l16_zk,
        &upload(&s, &d, &l16_kn)?,
        &l16_kr,
        &l16_kr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l16_pos),
        n * nkv,
        nkv,
    )?;
    let l16_qn_gpu = download(&s, &l16_qr, n * nh * hd)?;
    let l16_kn_gpu = download(&s, &l16_kr, n * nkv * hd)?;
    let mut l16_qn_cpu = Vec::with_capacity(n * nh * hd);
    let mut l16_kn_cpu = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            l16_qn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l16_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l16_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            l16_kn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l16_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l16_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer16_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l16_qn_gpu,
        &l16_qn_cpu,
    );
    report(
        "layer16_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l16_kn_gpu,
        &l16_kn_cpu,
    );
    report(
        "layer16_v_gemm",
        &format!("{:?}", l16_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l16_v_gpu,
        &l16_v_cpu,
    );

    let l16_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l16_bt = upload(&s, &d, &[0.])?;
    kv.append_kv(
        &s,
        &layout,
        &l16_pool,
        &upload(&s, &d, &l16_kn_gpu)?,
        &upload(&s, &d, &l16_v_gpu)?,
        &l16_bt,
        0,
        n,
    )?;
    let l16_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(
        &s,
        &upload(&s, &d, &l16_qn_gpu)?,
        &l16_pool,
        &l16_bt,
        &l16_att,
        nh,
        nkv,
        hd,
        64,
        n,
        0,
    )?;
    let l16_att_gpu = download(&s, &l16_att, n * nh * hd)?;
    let mut l16_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd].copy_from_slice(&l16_kn_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l16_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l16_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &l16_qn_cpu[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer16_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l16_att_gpu,
        &l16_att_cpu,
    );

    let l16_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l16_wo.data)?,
        &l16_att,
        &l16_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.16.attn_output.weight").unwrap().ggml_type),
        Some(&l16_in),
    )?;
    let l16_h_gpu = download(&s, &l16_h, n * h)?;
    let l16_h_cpu = l14_out_cpu
        .chunks_exact(h)
        .zip(l16_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l16_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer16_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l16_h_gpu,
        &l16_h_cpu,
    );

    let l16_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l16_h,
        &l16_zero,
        &upload(&s, &d, &l16_fn)?,
        &l16_f,
        &l16_f,
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
    let l16_f_gpu = download(&s, &l16_f, n * h)?;
    let l16_f_cpu = l16_h_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l16_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer16_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l16_f_gpu,
        &l16_f_cpu,
    );
    let i11 = r.get_tensor("blk.16.ffn_gate.weight").unwrap().dims[1] as usize;
    let l16_go = DeviceBuffer::alloc(d.clone(), n * i11 * 4)?;
    let l16_uo = DeviceBuffer::alloc(d.clone(), n * i11 * 4)?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l16_gw.data)?,
        &l16_f,
        &l16_go,
        h,
        i11,
        n,
        gemm_fmt(r.get_tensor("blk.16.ffn_gate.weight").unwrap().ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l16_uw.data)?,
        &l16_f,
        &l16_uo,
        h,
        i11,
        n,
        gemm_fmt(r.get_tensor("blk.16.ffn_up.weight").unwrap().ggml_type),
    )?;
    let l16_gg = download(&s, &l16_go, n * i11)?;
    let l16_ug = download(&s, &l16_uo, n * i11)?;
    let l16_gc = norm_cpu_for_ffn(&l16_f_cpu, &l16_gw, i11, h);
    let l16_uc = norm_cpu_for_ffn(&l16_f_cpu, &l16_uw, i11, h);
    report(
        "layer16_gate_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.16.ffn_gate.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i11}]"),
        n,
        &l16_gg,
        &l16_gc,
    );
    report(
        "layer16_up_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.16.ffn_up.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i11}]"),
        n,
        &l16_ug,
        &l16_uc,
    );
    let l16_sw_cpu = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &l16_gc[row * i11..(row + 1) * i11],
                &l16_uc[row * i11..(row + 1) * i11],
            )
        })
        .collect::<Vec<_>>();
    let l16_sw = DeviceBuffer::alloc(d.clone(), n * i11 * 4)?;
    let l16_zi = upload(&s, &d, &vec![0.; n * i11])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l16_go,
        &l16_zi,
        &l16_zi,
        &l16_uo,
        &l16_sw,
        eps,
        i11,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let l16_sw_gpu = download(&s, &l16_sw, n * i11)?;
    report(
        "layer16_swiglu",
        "F32",
        &format!("[{n},{i11}]"),
        n,
        &l16_sw_gpu,
        &l16_sw_cpu,
    );
    let l16_out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l16_dw.data)?,
        &l16_sw,
        &l16_out,
        i11,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.16.ffn_down.weight").unwrap().ggml_type),
        Some(&l16_h),
    )?;
    let l16_out_gpu = download(&s, &l16_out, n * h)?;
    let l16_out_cpu = (0..n)
        .flat_map(|row| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l16_dw, &l16_sw_cpu[row * i11..(row + 1) * i11]);
            l16_h_cpu[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer16_down_projection_residual",
        &format!(
            "{:?}",
            r.get_tensor("blk.16.ffn_down.weight").unwrap().ggml_type
        ),
        &format!("[{n},{h}]"),
        n,
        &l16_out_gpu,
        &l16_out_cpu,
    );
    // Continue from the real layer-16 post-down residual into block 17
    // through input RMSNorm and raw Q/K/V GEMMs.
    let l17w = vec32(&p, "blk.17.attn_norm.weight");
    let l17qw = tensor(&r, &p, "blk.17.attn_q.weight");
    let l17kw = tensor(&r, &p, "blk.17.attn_k.weight");
    let l17vw = tensor(&r, &p, "blk.17.attn_v.weight");
    assert_eq!(l17w.len(), h);
    report_q_metadata(&r, &p, "blk.17.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.17.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.17.attn_v.weight", h, nkv * hd);

    let l17_in = upload(&s, &d, &l16_out_gpu)?;
    let l17_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l17_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l17_in,
        &l17_zero,
        &upload(&s, &d, &l17w)?,
        &l17_norm,
        &l17_norm,
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
    let l17_norm_gpu = download(&s, &l17_norm, n * h)?;
    let l17_norm_cpu = l16_out_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l17w, eps))
        .collect::<Vec<_>>();
    report(
        "layer17_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l17_norm_gpu,
        &l17_norm_cpu,
    );

    let l17_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l17_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l17_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l17_q_info = r.get_tensor("blk.17.attn_q.weight").unwrap();
    let l17_k_info = r.get_tensor("blk.17.attn_k.weight").unwrap();
    let l17_v_info = r.get_tensor("blk.17.attn_v.weight").unwrap();
    let l17_qd = upload_raw(&s, &d, l17qw.data)?;
    let l17_kd = upload_raw(&s, &d, l17kw.data)?;
    let l17_vd = upload_raw(&s, &d, l17vw.data)?;
    g.gemm(
        &s,
        &l17_qd,
        &l17_norm,
        &l17_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l17_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l17_kd,
        &l17_norm,
        &l17_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l17_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l17_vd,
        &l17_norm,
        &l17_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l17_v_info.ggml_type),
    )?;
    let l17_q_gpu = download(&s, &l17_q, n * nh * hd)?;
    let l17_k_gpu = download(&s, &l17_k, n * nkv * hd)?;
    let l17_v_gpu = download(&s, &l17_v, n * nkv * hd)?;
    let l17_q_cpu = norm_cpu_for_ffn(&l17_norm_cpu, &l17qw, nh * hd, h);
    let l17_k_cpu = norm_cpu_for_ffn(&l17_norm_cpu, &l17kw, nkv * hd, h);
    let l17_v_cpu = norm_cpu_for_ffn(&l17_norm_cpu, &l17vw, nkv * hd, h);
    report(
        "layer17_q_raw_gemm",
        &format!("{:?}", l17_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l17_q_gpu,
        &l17_q_cpu,
    );
    report(
        "layer17_k_raw_gemm",
        &format!("{:?}", l17_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l17_k_gpu,
        &l17_k_cpu,
    );
    report(
        "layer17_v_raw_gemm",
        &format!("{:?}", l17_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l17_v_gpu,
        &l17_v_cpu,
    );

    let l17_qn = vec32(&p, "blk.17.attn_q_norm.weight");
    let l17_kn = vec32(&p, "blk.17.attn_k_norm.weight");
    let l17_wo = tensor(&r, &p, "blk.17.attn_output.weight");
    let l17_fn = vec32(&p, "blk.17.ffn_norm.weight");
    let l17_gw = tensor(&r, &p, "blk.17.ffn_gate.weight");
    let l17_uw = tensor(&r, &p, "blk.17.ffn_up.weight");
    let l17_dw = tensor(&r, &p, "blk.17.ffn_down.weight");
    report_q_metadata(&r, &p, "blk.17.attn_output.weight", nh * hd, h);

    let l17_qr = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l17_kr = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l17_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l17_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l17_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l17_q,
        &l17_zq,
        &upload(&s, &d, &l17_qn)?,
        &l17_qr,
        &l17_qr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l17_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l17_k,
        &l17_zk,
        &upload(&s, &d, &l17_kn)?,
        &l17_kr,
        &l17_kr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l17_pos),
        n * nkv,
        nkv,
    )?;
    let l17_qn_gpu = download(&s, &l17_qr, n * nh * hd)?;
    let l17_kn_gpu = download(&s, &l17_kr, n * nkv * hd)?;
    let mut l17_qn_cpu = Vec::with_capacity(n * nh * hd);
    let mut l17_kn_cpu = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            l17_qn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l17_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l17_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            l17_kn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l17_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l17_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer17_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l17_qn_gpu,
        &l17_qn_cpu,
    );
    report(
        "layer17_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l17_kn_gpu,
        &l17_kn_cpu,
    );
    report(
        "layer17_v_gemm",
        &format!("{:?}", l17_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l17_v_gpu,
        &l17_v_cpu,
    );

    let l17_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l17_bt = upload(&s, &d, &[0.])?;
    kv.append_kv(
        &s,
        &layout,
        &l17_pool,
        &upload(&s, &d, &l17_kn_gpu)?,
        &upload(&s, &d, &l17_v_gpu)?,
        &l17_bt,
        0,
        n,
    )?;
    let l17_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(
        &s,
        &upload(&s, &d, &l17_qn_gpu)?,
        &l17_pool,
        &l17_bt,
        &l17_att,
        nh,
        nkv,
        hd,
        64,
        n,
        0,
    )?;
    let l17_att_gpu = download(&s, &l17_att, n * nh * hd)?;
    let mut l17_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd].copy_from_slice(&l17_kn_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l17_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l17_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &l17_qn_cpu[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer17_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l17_att_gpu,
        &l17_att_cpu,
    );

    let l17_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l17_wo.data)?,
        &l17_att,
        &l17_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.17.attn_output.weight").unwrap().ggml_type),
        Some(&l17_in),
    )?;
    let l17_h_gpu = download(&s, &l17_h, n * h)?;
    let l17_h_cpu = l16_out_cpu
        .chunks_exact(h)
        .zip(l17_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l17_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer17_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l17_h_gpu,
        &l17_h_cpu,
    );

    let l17_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l17_h,
        &l17_zero,
        &upload(&s, &d, &l17_fn)?,
        &l17_f,
        &l17_f,
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
    let l17_f_gpu = download(&s, &l17_f, n * h)?;
    let l17_f_cpu = l17_h_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l17_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer17_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l17_f_gpu,
        &l17_f_cpu,
    );
    let i12 = r.get_tensor("blk.17.ffn_gate.weight").unwrap().dims[1] as usize;
    let l17_go = DeviceBuffer::alloc(d.clone(), n * i12 * 4)?;
    let l17_uo = DeviceBuffer::alloc(d.clone(), n * i12 * 4)?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l17_gw.data)?,
        &l17_f,
        &l17_go,
        h,
        i12,
        n,
        gemm_fmt(r.get_tensor("blk.17.ffn_gate.weight").unwrap().ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l17_uw.data)?,
        &l17_f,
        &l17_uo,
        h,
        i12,
        n,
        gemm_fmt(r.get_tensor("blk.17.ffn_up.weight").unwrap().ggml_type),
    )?;
    let l17_gg = download(&s, &l17_go, n * i12)?;
    let l17_ug = download(&s, &l17_uo, n * i12)?;
    let l17_gc = norm_cpu_for_ffn(&l17_f_cpu, &l17_gw, i12, h);
    let l17_uc = norm_cpu_for_ffn(&l17_f_cpu, &l17_uw, i12, h);
    report(
        "layer17_gate_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.17.ffn_gate.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i12}]"),
        n,
        &l17_gg,
        &l17_gc,
    );
    report(
        "layer17_up_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.17.ffn_up.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i12}]"),
        n,
        &l17_ug,
        &l17_uc,
    );
    let l17_sw_cpu = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &l17_gc[row * i12..(row + 1) * i12],
                &l17_uc[row * i12..(row + 1) * i12],
            )
        })
        .collect::<Vec<_>>();
    let l17_sw = DeviceBuffer::alloc(d.clone(), n * i12 * 4)?;
    let l17_zi = upload(&s, &d, &vec![0.; n * i12])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l17_go,
        &l17_zi,
        &l17_zi,
        &l17_uo,
        &l17_sw,
        eps,
        i12,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let l17_sw_gpu = download(&s, &l17_sw, n * i12)?;
    report(
        "layer17_swiglu",
        "F32",
        &format!("[{n},{i12}]"),
        n,
        &l17_sw_gpu,
        &l17_sw_cpu,
    );
    let l17_out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l17_dw.data)?,
        &l17_sw,
        &l17_out,
        i12,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.17.ffn_down.weight").unwrap().ggml_type),
        Some(&l17_h),
    )?;
    let l17_out_gpu = download(&s, &l17_out, n * h)?;
    let l17_out_cpu = (0..n)
        .flat_map(|row| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l17_dw, &l17_sw_cpu[row * i12..(row + 1) * i12]);
            l17_h_cpu[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer17_down_projection_residual",
        &format!(
            "{:?}",
            r.get_tensor("blk.17.ffn_down.weight").unwrap().ggml_type
        ),
        &format!("[{n},{h}]"),
        n,
        &l17_out_gpu,
        &l17_out_cpu,
    );
    // Continue from the real layer-17 post-down residual into block 18
    // through input RMSNorm and raw Q/K/V GEMMs.
    let l18w = vec32(&p, "blk.18.attn_norm.weight");
    let l18qw = tensor(&r, &p, "blk.18.attn_q.weight");
    let l18kw = tensor(&r, &p, "blk.18.attn_k.weight");
    let l18vw = tensor(&r, &p, "blk.18.attn_v.weight");
    assert_eq!(l18w.len(), h);
    report_q_metadata(&r, &p, "blk.18.attn_q.weight", h, nh * hd);
    report_q_metadata(&r, &p, "blk.18.attn_k.weight", h, nkv * hd);
    report_q_metadata(&r, &p, "blk.18.attn_v.weight", h, nkv * hd);

    let l18_in = upload(&s, &d, &l17_out_gpu)?;
    let l18_zero = upload(&s, &d, &vec![0.; n * h])?;
    let l18_norm = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l18_in,
        &l18_zero,
        &upload(&s, &d, &l18w)?,
        &l18_norm,
        &l18_norm,
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
    let l18_norm_gpu = download(&s, &l18_norm, n * h)?;
    let l18_norm_cpu = l17_out_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l18w, eps))
        .collect::<Vec<_>>();
    report(
        "layer18_attn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l18_norm_gpu,
        &l18_norm_cpu,
    );

    let l18_q = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l18_k = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l18_v = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l18_q_info = r.get_tensor("blk.18.attn_q.weight").unwrap();
    let l18_k_info = r.get_tensor("blk.18.attn_k.weight").unwrap();
    let l18_v_info = r.get_tensor("blk.18.attn_v.weight").unwrap();
    let l18_qd = upload_raw(&s, &d, l18qw.data)?;
    let l18_kd = upload_raw(&s, &d, l18kw.data)?;
    let l18_vd = upload_raw(&s, &d, l18vw.data)?;
    g.gemm(
        &s,
        &l18_qd,
        &l18_norm,
        &l18_q,
        h,
        nh * hd,
        n,
        gemm_fmt(l18_q_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l18_kd,
        &l18_norm,
        &l18_k,
        h,
        nkv * hd,
        n,
        gemm_fmt(l18_k_info.ggml_type),
    )?;
    g.gemm(
        &s,
        &l18_vd,
        &l18_norm,
        &l18_v,
        h,
        nkv * hd,
        n,
        gemm_fmt(l18_v_info.ggml_type),
    )?;
    let l18_q_gpu = download(&s, &l18_q, n * nh * hd)?;
    let l18_k_gpu = download(&s, &l18_k, n * nkv * hd)?;
    let l18_v_gpu = download(&s, &l18_v, n * nkv * hd)?;
    let l18_q_cpu = norm_cpu_for_ffn(&l18_norm_cpu, &l18qw, nh * hd, h);
    let l18_k_cpu = norm_cpu_for_ffn(&l18_norm_cpu, &l18kw, nkv * hd, h);
    let l18_v_cpu = norm_cpu_for_ffn(&l18_norm_cpu, &l18vw, nkv * hd, h);
    report(
        "layer18_q_raw_gemm",
        &format!("{:?}", l18_q_info.ggml_type),
        &format!("[{n},{nh},{hd}]"),
        n,
        &l18_q_gpu,
        &l18_q_cpu,
    );
    report(
        "layer18_k_raw_gemm",
        &format!("{:?}", l18_k_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l18_k_gpu,
        &l18_k_cpu,
    );
    report(
        "layer18_v_raw_gemm",
        &format!("{:?}", l18_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l18_v_gpu,
        &l18_v_cpu,
    );

    let l18_qn = vec32(&p, "blk.18.attn_q_norm.weight");
    let l18_kn = vec32(&p, "blk.18.attn_k_norm.weight");
    let l18_wo = tensor(&r, &p, "blk.18.attn_output.weight");
    let l18_fn = vec32(&p, "blk.18.ffn_norm.weight");
    let l18_gw = tensor(&r, &p, "blk.18.ffn_gate.weight");
    let l18_uw = tensor(&r, &p, "blk.18.ffn_up.weight");
    let l18_dw = tensor(&r, &p, "blk.18.ffn_down.weight");
    report_q_metadata(&r, &p, "blk.18.attn_output.weight", nh * hd, h);

    let l18_qr = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    let l18_kr = DeviceBuffer::alloc(d.clone(), n * nkv * hd * 4)?;
    let l18_zq = upload(&s, &d, &vec![0.; n * nh * hd])?;
    let l18_zk = upload(&s, &d, &vec![0.; n * nkv * hd])?;
    let l18_pos = upload_raw(
        &s,
        &d,
        &(0..n)
            .flat_map(|pos| (pos as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l18_q,
        &l18_zq,
        &upload(&s, &d, &l18_qn)?,
        &l18_qr,
        &l18_qr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l18_pos),
        n * nh,
        nh,
    )?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l18_k,
        &l18_zk,
        &upload(&s, &d, &l18_kn)?,
        &l18_kr,
        &l18_kr,
        eps,
        hd,
        hd,
        c.rope_freq_base,
        0,
        MODE_NORM | MODE_ROPE,
        Some(&l18_pos),
        n * nkv,
        nkv,
    )?;
    let l18_qn_gpu = download(&s, &l18_qr, n * nh * hd)?;
    let l18_kn_gpu = download(&s, &l18_kr, n * nkv * hd)?;
    let mut l18_qn_cpu = Vec::with_capacity(n * nh * hd);
    let mut l18_kn_cpu = Vec::with_capacity(n * nkv * hd);
    for pos in 0..n {
        for head in 0..nh {
            l18_qn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l18_q_cpu[(pos * nh + head) * hd..(pos * nh + head + 1) * hd],
                    &l18_qn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        for head in 0..nkv {
            l18_kn_cpu.extend(engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(
                    &l18_k_cpu[(pos * nkv + head) * hd..(pos * nkv + head + 1) * hd],
                    &l18_kn,
                    eps,
                ),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
    }
    report(
        "layer18_q_norm_rope",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l18_qn_gpu,
        &l18_qn_cpu,
    );
    report(
        "layer18_k_norm_rope",
        "F32",
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l18_kn_gpu,
        &l18_kn_cpu,
    );
    report(
        "layer18_v_gemm",
        &format!("{:?}", l18_v_info.ggml_type),
        &format!("[{n},{nkv},{hd}]"),
        n,
        &l18_v_gpu,
        &l18_v_cpu,
    );

    let l18_pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let l18_bt = upload(&s, &d, &[0.])?;
    kv.append_kv(
        &s,
        &layout,
        &l18_pool,
        &upload(&s, &d, &l18_kn_gpu)?,
        &upload(&s, &d, &l18_v_gpu)?,
        &l18_bt,
        0,
        n,
    )?;
    let l18_att = DeviceBuffer::alloc(d.clone(), n * nh * hd * 4)?;
    fa.launch(
        &s,
        &upload(&s, &d, &l18_qn_gpu)?,
        &l18_pool,
        &l18_bt,
        &l18_att,
        nh,
        nkv,
        hd,
        64,
        n,
        0,
    )?;
    let l18_att_gpu = download(&s, &l18_att, n * nh * hd)?;
    let mut l18_att_cpu = Vec::with_capacity(n * nh * hd);
    for pos in 0..n {
        let mut ph = vec![0.; layout.floats_total()];
        for t in 0..n {
            let off = t * 2 * nkv * hd;
            ph[off..off + nkv * hd].copy_from_slice(&l18_kn_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
            ph[off + nkv * hd..off + 2 * nkv * hd]
                .copy_from_slice(&l18_v_cpu[t * nkv * hd..(t + 1) * nkv * hd]);
        }
        l18_att_cpu.extend(sdpa_decode(
            &ph,
            &[0],
            64,
            n,
            &l18_qn_cpu[pos * nh * hd..(pos + 1) * nh * hd],
            nh,
            nkv,
            hd,
            true,
            pos,
        ));
    }
    report(
        "layer18_attention",
        "F32",
        &format!("[{n},{nh},{hd}]"),
        n,
        &l18_att_gpu,
        &l18_att_cpu,
    );

    let l18_h = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l18_wo.data)?,
        &l18_att,
        &l18_h,
        nh * hd,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.18.attn_output.weight").unwrap().ggml_type),
        Some(&l18_in),
    )?;
    let l18_h_gpu = download(&s, &l18_h, n * h)?;
    let l18_h_cpu = l17_out_cpu
        .chunks_exact(h)
        .zip(l18_att_cpu.chunks_exact(nh * hd))
        .flat_map(|(x, a)| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l18_wo, a);
            x.iter().zip(z).map(|(&u, v)| u + v).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer18_output_projection_residual",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l18_h_gpu,
        &l18_h_cpu,
    );

    let l18_f = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l18_h,
        &l18_zero,
        &upload(&s, &d, &l18_fn)?,
        &l18_f,
        &l18_f,
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
    let l18_f_gpu = download(&s, &l18_f, n * h)?;
    let l18_f_cpu = l18_h_cpu
        .chunks_exact(h)
        .flat_map(|row| rms_norm(row, &l18_fn, eps))
        .collect::<Vec<_>>();
    report(
        "layer18_ffn_input_rmsnorm",
        "F32",
        &format!("[{n},{h}]"),
        n,
        &l18_f_gpu,
        &l18_f_cpu,
    );
    let i12 = r.get_tensor("blk.18.ffn_gate.weight").unwrap().dims[1] as usize;
    let l18_go = DeviceBuffer::alloc(d.clone(), n * i12 * 4)?;
    let l18_uo = DeviceBuffer::alloc(d.clone(), n * i12 * 4)?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l18_gw.data)?,
        &l18_f,
        &l18_go,
        h,
        i12,
        n,
        gemm_fmt(r.get_tensor("blk.18.ffn_gate.weight").unwrap().ggml_type),
    )?;
    g.gemm(
        &s,
        &upload_raw(&s, &d, l18_uw.data)?,
        &l18_f,
        &l18_uo,
        h,
        i12,
        n,
        gemm_fmt(r.get_tensor("blk.18.ffn_up.weight").unwrap().ggml_type),
    )?;
    let l18_gg = download(&s, &l18_go, n * i12)?;
    let l18_ug = download(&s, &l18_uo, n * i12)?;
    let l18_gc = norm_cpu_for_ffn(&l18_f_cpu, &l18_gw, i12, h);
    let l18_uc = norm_cpu_for_ffn(&l18_f_cpu, &l18_uw, i12, h);
    report(
        "layer18_gate_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.18.ffn_gate.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i12}]"),
        n,
        &l18_gg,
        &l18_gc,
    );
    report(
        "layer18_up_gemm",
        &format!(
            "{:?}",
            r.get_tensor("blk.18.ffn_up.weight").unwrap().ggml_type
        ),
        &format!("[{n},{i12}]"),
        n,
        &l18_ug,
        &l18_uc,
    );
    let l18_sw_cpu = (0..n)
        .flat_map(|row| {
            engine_core::forward_cpu::swiglu(
                &l18_gc[row * i12..(row + 1) * i12],
                &l18_uc[row * i12..(row + 1) * i12],
            )
        })
        .collect::<Vec<_>>();
    let l18_sw = DeviceBuffer::alloc(d.clone(), n * i12 * 4)?;
    let l18_zi = upload(&s, &d, &vec![0.; n * i12])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &l18_go,
        &l18_zi,
        &l18_zi,
        &l18_uo,
        &l18_sw,
        eps,
        i12,
        0,
        c.rope_freq_base,
        0,
        MODE_SWIGLU,
        None,
        n,
        1,
    )?;
    let l18_sw_gpu = download(&s, &l18_sw, n * i12)?;
    report(
        "layer18_swiglu",
        "F32",
        &format!("[{n},{i12}]"),
        n,
        &l18_sw_gpu,
        &l18_sw_cpu,
    );
    let l18_out = DeviceBuffer::alloc(d.clone(), n * h * 4)?;
    g.gemm_with_residual(
        &s,
        &upload_raw(&s, &d, l18_dw.data)?,
        &l18_sw,
        &l18_out,
        i12,
        h,
        n,
        gemm_fmt(r.get_tensor("blk.18.ffn_down.weight").unwrap().ggml_type),
        Some(&l18_h),
    )?;
    let l18_out_gpu = download(&s, &l18_out, n * h)?;
    let l18_out_cpu = (0..n)
        .flat_map(|row| {
            let mut z = vec![0.; h];
            matmul(&mut z, &l18_dw, &l18_sw_cpu[row * i12..(row + 1) * i12]);
            l18_h_cpu[row * h..(row + 1) * h]
                .iter()
                .zip(z)
                .map(|(&a, b)| a + b)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    report(
        "layer18_down_projection_residual",
        &format!(
            "{:?}",
            r.get_tensor("blk.18.ffn_down.weight").unwrap().ggml_type
        ),
        &format!("[{n},{h}]"),
        n,
        &l18_out_gpu,
        &l18_out_cpu,
    );
    Ok(())
}

#[test]
#[ignore]
fn real_two_layer_transition_parity() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    for chunk in [1usize, 2, 4] {
        println!("=== transition chunk={chunk} ===");
        run_transition(chunk)?;
    }
    Ok(())
}

fn norm_cpu_for_ffn(normalized: &[f32], w: &Tensor<'_>, out_dim: usize, h: usize) -> Vec<f32> {
    normalized
        .chunks_exact(h)
        .flat_map(|row| {
            let mut y = vec![0.; out_dim];
            matmul(&mut y, w, row);
            y
        })
        .collect()
}
