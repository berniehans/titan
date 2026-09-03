//! Diagnostic-only parity probe for prefill's mixed-format QKV projections.
//!
//! This deliberately calls the same generic `BatchedGEMM::gemm` entry point
//! used by prefill, independently for Q, K, and V. It does not change routing
//! or establish a production threshold.

use cudarc::driver::CudaDevice;
use engine_core::forward_cpu::{Tensor, TensorType, matmul};
use engine_cuda::{BatchedGEMM, CudaError, CudaStream, DeviceBuffer, GemvFormat};
use engine_io::{GgmlType, GgufReader, load_to_pinned};
use std::path::PathBuf;
use std::sync::Arc;

fn fixture_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("ENGINE_TESTDATA") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    [
        manifest.join("../../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        manifest.join("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
    ]
    .into_iter()
    .find(|path| path.exists())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bytes_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn rel_l2(actual: &[f32], expected: &[f32]) -> f32 {
    let (num, den) = actual
        .iter()
        .zip(expected)
        .fold((0.0f64, 0.0f64), |(n, d), (a, e)| {
            let delta = *a as f64 - *e as f64;
            (n + delta * delta, d + (*e as f64) * (*e as f64))
        });
    (num / den).sqrt() as f32
}

fn cosine(actual: &[f32], expected: &[f32]) -> f32 {
    let (dot, aa, bb) =
        actual
            .iter()
            .zip(expected)
            .fold((0.0f64, 0.0f64, 0.0f64), |(dot, aa, bb), (a, b)| {
                let (a, b) = (*a as f64, *b as f64);
                (dot + a * b, aa + a * a, bb + b * b)
            });
    (dot / (aa * bb).sqrt()) as f32
}

fn format_for(ty: GgmlType) -> (TensorType, GemvFormat, usize) {
    match ty {
        GgmlType::Q4_K => (TensorType::Q4K, GemvFormat::Q4K, 144),
        GgmlType::Q6_K => (TensorType::Q6K, GemvFormat::Q6K, 210),
        other => panic!("unsupported diagnostic QKV type: {other:?}"),
    }
}

#[test]
#[ignore]
fn diagnostic_mixed_qkv_batched_gemm_matches_cpu() -> Result<(), CudaError> {
    engine_cuda::ensure_cuda_dll_paths();
    let Some(fixture) = fixture_path() else {
        eprintln!("SKIP: Qwen3 fixture not present");
        return Ok(());
    };
    let reader = GgufReader::open(&fixture).expect("open Qwen3 GGUF");
    let loaded = load_to_pinned(&reader, &fixture).expect("load Qwen3 GGUF");
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let gemm = BatchedGEMM::new(Arc::clone(&device))?;

    for role in ["q", "k", "v"] {
        let name = format!("blk.0.attn_{role}.weight");
        let info = reader.get_tensor(&name).expect("QKV tensor present");
        assert_eq!(info.dims.len(), 2, "{name} must be 2-D");
        let ne0 = info.dims[0] as usize;
        let ne1 = info.dims[1] as usize;
        let (cpu_ty, gpu_format, block_bytes) = format_for(info.ggml_type);
        let weight_bytes = loaded.tensor(&name).expect("QKV tensor bytes");
        assert_eq!(
            weight_bytes.len(),
            (ne0 / 256) * block_bytes * ne1,
            "{name} byte span"
        );
        println!(
            "role={role} tensor={name} ggml_type={:?} shape=[{ne0}, {ne1}]",
            info.ggml_type
        );

        let weights = DeviceBuffer::alloc(Arc::clone(&device), weight_bytes.len())?;
        weights.copy_from_host(&stream, weight_bytes)?;
        let tensor = Tensor {
            ty: cpu_ty,
            data: weight_bytes,
            ne0,
            ne1,
            n_rot: 0,
        };
        for batch in [1usize, 2, 4] {
            let activations: Vec<f32> = (0..batch * ne0)
                .map(|i| 0.125 + 0.0007 * (i % ne0) as f32 + 0.013 * (i / ne0) as f32)
                .collect();
            let x = DeviceBuffer::alloc(Arc::clone(&device), batch * ne0 * 4)?;
            let out = DeviceBuffer::alloc(Arc::clone(&device), batch * ne1 * 4)?;
            x.copy_from_host(&stream, &f32_bytes(&activations))?;
            gemm.gemm(&stream, &weights, &x, &out, ne0, ne1, batch, gpu_format)?;
            let mut raw = vec![0u8; batch * ne1 * 4];
            out.copy_to_host(&stream, &mut raw)?;
            let gpu = bytes_f32(&raw);
            let mut cpu = Vec::with_capacity(batch * ne1);
            for row in 0..batch {
                let mut row_out = vec![0.0f32; ne1];
                matmul(
                    &mut row_out,
                    &tensor,
                    &activations[row * ne0..(row + 1) * ne0],
                );
                cpu.extend(row_out);
            }
            println!(
                "role={role} format={:?} ne0={ne0} ne1={ne1} batch={batch} rel_l2={:.6e} cosine={:.9}",
                info.ggml_type,
                rel_l2(&gpu, &cpu),
                cosine(&gpu, &cpu)
            );
        }
    }
    Ok(())
}
