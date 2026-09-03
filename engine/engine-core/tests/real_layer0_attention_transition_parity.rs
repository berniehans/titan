//! Real Qwen3 layer-0 Q/K/V -> paged KV -> FlashAttention transition probe.
//!
//! This is intentionally ignored: it requires CUDA, NVRTC and the committed
//! Qwen3 fixture.  Unlike the synthetic CUDA parity test, Q/K/V here are made
//! from the real layer-0 weights and the same CPU reference used by golden
//! tests (`sdpa_decode`).

use cudarc::driver::CudaDevice;
use engine_core::forward_cpu::{
    Tensor, TensorType, matmul, rms_norm, rope_neox_partial, sdpa_decode,
};
use engine_cuda::{
    CudaError, CudaStream, DeviceBuffer, FlashAttention2, KvDataType, PagedKvGpu, PagedKvLayout,
};
use engine_io::{GgmlType, GgufReader, LoadedPinned, ModelConfig, load_to_pinned};
use std::{path::PathBuf, sync::Arc};

fn fixture() -> Option<PathBuf> {
    let m = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    [
        std::env::var_os("ENGINE_TESTDATA").map(PathBuf::from),
        Some(m.join("../../testdata/Qwen3-0.6B-Q4_K_M.gguf")),
        Some(m.join("../testdata/Qwen3-0.6B-Q4_K_M.gguf")),
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
fn bytes(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn upload(s: &CudaStream, d: &Arc<CudaDevice>, x: &[f32]) -> Result<DeviceBuffer, CudaError> {
    let b = DeviceBuffer::alloc(d.clone(), x.len() * 4)?;
    b.copy_from_host(s, &bytes(x))?;
    Ok(b)
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
fn tensor<'a>(r: &GgufReader, p: &'a LoadedPinned, name: &str) -> Tensor<'a> {
    let i = r.get_tensor(name).unwrap();
    Tensor {
        ty: match i.ggml_type {
            GgmlType::F32 => TensorType::F32,
            GgmlType::Q4_K => TensorType::Q4K,
            GgmlType::Q6_K => TensorType::Q6K,
            x => panic!("unsupported {x:?}"),
        },
        data: p.tensor(name).unwrap(),
        ne0: i.dims[0] as usize,
        ne1: i.dims[1] as usize,
        n_rot: 0,
    }
}
fn vec32(p: &LoadedPinned, n: &str) -> Vec<f32> {
    f32s(p.tensor(n).unwrap())
}

#[test]
#[ignore]
fn real_layer0_attention_transition_parity() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
{
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
    let emb = tensor(&r, &p, "token_embd.weight");
    let an = vec32(&p, "blk.0.attn_norm.weight");
    let qn = vec32(&p, "blk.0.attn_q_norm.weight");
    let kn = vec32(&p, "blk.0.attn_k_norm.weight");
    let qw = tensor(&r, &p, "blk.0.attn_q.weight");
    let kw = tensor(&r, &p, "blk.0.attn_k.weight");
    let vw = tensor(&r, &p, "blk.0.attn_v.weight");
    let mut q = Vec::with_capacity(n * nh * hd);
    let mut k = Vec::with_capacity(n * nkv * hd);
    let mut v = Vec::with_capacity(n * nkv * hd);
    for (pos, &id) in ids.iter().enumerate() {
        let x = rms_norm(
            &engine_core::forward_cpu::embed_lookup(&emb, id),
            &an,
            c.rms_norm_eps,
        );
        let mut z = vec![0.; nh * hd];
        matmul(&mut z, &qw, &x);
        for head in 0..nh {
            q.extend(rope_neox_partial(
                &rms_norm(&z[head * hd..(head + 1) * hd], &qn, c.rms_norm_eps),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        let mut z = vec![0.; nkv * hd];
        matmul(&mut z, &kw, &x);
        for head in 0..nkv {
            k.extend(rope_neox_partial(
                &rms_norm(&z[head * hd..(head + 1) * hd], &kn, c.rms_norm_eps),
                pos as u32,
                hd,
                c.rope_freq_base,
            ));
        }
        matmul(&mut z, &vw, &x);
        v.extend(z);
    }
    let dev = CudaDevice::new(0)?;
    let s = CudaStream::new(dev.clone())?;
    let kv = PagedKvGpu::new(dev.clone())?;
    let fa = FlashAttention2::new(dev.clone())?;
    println!(
        "real layer0 attention dims hidden={h} q_heads={nh} kv_heads={nkv} head_dim={hd} positions=0..{}",
        n - 1
    );
    for &chunk in &[1usize, 2, 4] {
        let layout = PagedKvLayout {
            n_blocks: 1,
            block_tokens: 64,
            row_len: nkv * hd,
            data_type: KvDataType::F32,
        };
        let pool = DeviceBuffer::alloc(dev.clone(), layout.bytes_total())?;
        let bt = upload(&s, &dev, &[0.])?;
        let kd = upload(&s, &dev, &k[..chunk * nkv * hd])?;
        let vd = upload(&s, &dev, &v[..chunk * nkv * hd])?;
        kv.append_kv(&s, &layout, &pool, &kd, &vd, &bt, 0, chunk)?;
        let qd = upload(&s, &dev, &q[..chunk * nh * hd])?;
        let out = DeviceBuffer::alloc(dev.clone(), chunk * nh * hd * 4)?;
        fa.launch(&s, &qd, &pool, &bt, &out, nh, nkv, hd, 64, chunk, 0)?;
        let mut raw = vec![0u8; chunk * nh * hd * 4];
        out.copy_to_host(&s, &mut raw)?;
        let gpu = f32s(&raw);
        let mut cpu = Vec::new();
        for pos in 0..chunk {
            cpu.extend(sdpa_decode(
                &{
                    let mut pool_h = vec![0.; layout.floats_total()];
                    for t in 0..chunk {
                        let base = t * 2 * nkv * hd;
                        pool_h[base..base + nkv * hd]
                            .copy_from_slice(&k[t * nkv * hd..(t + 1) * nkv * hd]);
                        pool_h[base + nkv * hd..base + 2 * nkv * hd]
                            .copy_from_slice(&v[t * nkv * hd..(t + 1) * nkv * hd]);
                    }
                    pool_h
                },
                &[0],
                64,
                chunk,
                &q[pos * nh * hd..(pos + 1) * nh * hd],
                nh,
                nkv,
                hd,
                true,
                pos,
            ));
        }
        for pos in 0..chunk {
            for head in 0..nh {
                let o = (pos * nh + head) * hd;
                let (rel, cos) = metric(&gpu[o..o + hd], &cpu[o..o + hd]);
                println!(
                    "chunk={chunk} positions=0..{} head={head} token={pos} rel_l2={rel:.6e} cosine={cos:.9}",
                    chunk - 1
                );
                assert!(rel < 1e-3 && cos > 0.9999);
            }
        }
    }
    Ok(())
}
