//! Focused diagnostic for the real Qwen3 layer-0 post-QKV transition.
//!
//! The current public fused Q/K launcher is single-token.  This probe therefore
//! uses the existing batched norm+RoPE launcher on the exact Qwen3 head rows;
//! it deliberately does not invent a batched attention composition.

use cudarc::driver::CudaDevice;
use engine_core::forward_cpu::{rms_norm, rope_neox_partial};
use engine_cuda::{
    CudaError, CudaStream, DeviceBuffer, MODE_BROADCAST_RESIDUAL, MODE_NORM, MODE_ROPE, NormRope,
};
use engine_io::{GgufReader, LoadedPinned, load_to_pinned};
use std::path::PathBuf;
use std::sync::Arc;

const HIDDEN: usize = 1024;
const HEAD_DIM: usize = 128;
const Q_HEADS: usize = 16;
const K_HEADS: usize = 8;
const ROT: usize = 128;
const EPS: f32 = 1e-6;
const BASE: f32 = 1_000_000.0;

fn fixture() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    [
        std::env::var_os("ENGINE_TESTDATA").map(PathBuf::from),
        Some(manifest.join("../../testdata/Qwen3-0.6B-Q4_K_M.gguf")),
        Some(manifest.join("../testdata/Qwen3-0.6B-Q4_K_M.gguf")),
        Some(PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf")),
    ]
    .into_iter()
    .flatten()
    .find(|p| p.exists())
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_f32(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn upload(s: &CudaStream, d: &Arc<CudaDevice>, v: &[f32]) -> Result<DeviceBuffer, CudaError> {
    let b = DeviceBuffer::alloc(Arc::clone(d), v.len() * 4)?;
    b.copy_from_host(s, &f32_bytes(v))?;
    Ok(b)
}
fn upload_u32(s: &CudaStream, d: &Arc<CudaDevice>, value: u32) -> Result<DeviceBuffer, CudaError> {
    let b = DeviceBuffer::alloc(Arc::clone(d), 4)?;
    b.copy_from_host(s, &value.to_le_bytes())?;
    Ok(b)
}
fn download(s: &CudaStream, b: &DeviceBuffer, n: usize) -> Result<Vec<f32>, CudaError> {
    let mut raw = vec![0; n * 4];
    b.copy_to_host(s, &mut raw)?;
    Ok(bytes_f32(&raw))
}
fn metric(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut n, mut d, mut dot, mut aa, mut bb) = (0., 0., 0., 0., 0.);
    for (&x, &y) in a.iter().zip(b) {
        let dx = x as f64 - y as f64;
        n += dx * dx;
        d += (y as f64) * (y as f64);
        dot += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    ((n / d).sqrt(), dot / (aa * bb).sqrt())
}
fn norm_weight(p: &LoadedPinned, name: &str) -> Vec<f32> {
    p.tensor(name)
        .unwrap()
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore]
fn diagnostic_batched_qkv_norm_rope_matches_cpu() -> Result<(), CudaError> {
    engine_cuda::ensure_cuda_dll_paths();
    let Some(path) = fixture() else {
        eprintln!("SKIP: Qwen3 fixture not present");
        return Ok(());
    };
    let reader = GgufReader::open(&path).expect("open Qwen3 GGUF");
    let pinned = load_to_pinned(&reader, &path).expect("load Qwen3 GGUF");
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let nr = NormRope::new(Arc::clone(&device))?;
    let q_w = norm_weight(&pinned, "blk.0.attn_q_norm.weight");
    let k_w = norm_weight(&pinned, "blk.0.attn_k_norm.weight");
    assert_eq!((q_w.len(), k_w.len()), (HEAD_DIM, HEAD_DIM));
    println!(
        "dims hidden={HIDDEN} q_heads={Q_HEADS} k_heads={K_HEADS} head_dim={HEAD_DIM} n_rot={ROT}"
    );

    for tokens in [1usize, 2, 4, 16] {
        let q_rows = tokens * Q_HEADS;
        let k_rows = tokens * K_HEADS;
        for (label, heads, weights) in [("Q", Q_HEADS, &q_w), ("K", K_HEADS, &k_w)] {
            let rows = tokens * heads;
            let input: Vec<f32> = (0..rows * HEAD_DIM)
                .map(|i| 0.07 + (i % HEAD_DIM) as f32 * 0.001 + (i / HEAD_DIM) as f32 * 0.003)
                .collect();
            let residual: Vec<f32> = (0..heads * HEAD_DIM)
                .map(|i| 0.011 + (i % HEAD_DIM) as f32 * 0.0001 + (i / HEAD_DIM) as f32 * 0.002)
                .collect();
            let x = upload(&stream, &device, &input)?;
            let residual_device = upload(&stream, &device, &residual)?;
            let w = upload(&stream, &device, weights)?;
            let out = upload(&stream, &device, &vec![0.0; rows * HEAD_DIM])?;
            let pos = upload_u32(&stream, &device, 11)?;
            nr.launch_batched_with_pos_ptr(
                &stream,
                &x,
                &residual_device,
                &w,
                &out,
                &out,
                EPS,
                HEAD_DIM,
                ROT,
                BASE,
                0,
                MODE_NORM | MODE_ROPE | MODE_BROADCAST_RESIDUAL,
                Some(&pos),
                rows,
                heads,
            )?;
            let gpu = download(&stream, &out, rows * HEAD_DIM)?;
            let mut cpu = Vec::with_capacity(rows * HEAD_DIM);
            for r in 0..rows {
                let residual_row = &residual[(r % heads) * HEAD_DIM..(r % heads + 1) * HEAD_DIM];
                let input_with_residual: Vec<f32> = input[r * HEAD_DIM..(r + 1) * HEAD_DIM]
                    .iter()
                    .zip(residual_row)
                    .map(|(&x, &residual)| x + residual)
                    .collect();
                let n = rms_norm(&input_with_residual, weights, EPS);
                cpu.extend(rope_neox_partial(&n, 11 + (r / heads) as u32, ROT, BASE));
            }
            let (rel, cos) = metric(&gpu, &cpu);
            println!(
                "role={label} dims=[tokens={tokens},heads={heads},head_dim={HEAD_DIM}] batch={rows} position=11..{} rel_l2={rel:.6e} cosine={cos:.9}",
                10 + tokens
            );
            assert!(
                rel < 2e-4 && cos > 0.999999,
                "{label} post-QKV parity failed"
            );
        }
        assert_eq!(q_rows, tokens * Q_HEADS);
        assert_eq!(k_rows, tokens * K_HEADS);
    }
    println!(
        "attention/V limitation: existing public PagedAttention is decode-only and launch_fused_qk is single-token; no fake chunk comparison was added"
    );
    Ok(())
}
