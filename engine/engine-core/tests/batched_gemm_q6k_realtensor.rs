//! Diagnostic probe for the real Q6_K batched prefill path.
//!
//! This intentionally exercises `BatchedGEMM::gemm` and
//! `BatchedGEMM::gemm_with_residual` with real Qwen3 weights and batch sizes
//! that the isolated GEMV tests do not cover. It is diagnostic only.

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
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    [
        manifest_dir.join("../../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        manifest_dir.join("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
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
    let (numerator, denominator) = actual.iter().zip(expected).fold(
        (0.0f64, 0.0f64),
        |(numerator, denominator), (actual, expected)| {
            let delta = *actual as f64 - *expected as f64;
            (
                numerator + delta * delta,
                denominator + (*expected as f64) * (*expected as f64),
            )
        },
    );
    (numerator / denominator).sqrt() as f32
}

fn activation_row(row: usize, width: usize) -> Vec<f32> {
    (0..width)
        .map(|column| 0.25 + 0.003 * column as f32 + 0.01 * row as f32)
        .collect()
}

#[test]
#[ignore]
fn diagnostic_real_q6k_batched_gemm_prefill_batch_sizes() -> Result<(), CudaError> {
    engine_cuda::ensure_cuda_dll_paths();
    let Some(fixture) = fixture_path() else {
        eprintln!("SKIP: Qwen3-0.6B fixture not present");
        return Ok(());
    };

    let reader = GgufReader::open(&fixture).expect("open GGUF fixture");
    let loaded = load_to_pinned(&reader, &fixture).expect("load GGUF fixture");
    let info = reader
        .get_tensor("blk.0.ffn_down.weight")
        .expect("real Q6_K ffn_down tensor");
    assert_eq!(info.ggml_type, GgmlType::Q6_K);
    assert_eq!(info.dims.len(), 2);
    let ne0 = info.dims[0] as usize;
    let full_ne1 = info.dims[1] as usize;
    let ne1 = full_ne1.min(64);
    let bytes_per_column = (ne0 / 256) * 210;
    let weight_bytes = &loaded
        .tensor("blk.0.ffn_down.weight")
        .expect("weight bytes")[..bytes_per_column * ne1];

    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let gemm = BatchedGEMM::new(Arc::clone(&device))?;
    let weights = DeviceBuffer::alloc(Arc::clone(&device), weight_bytes.len())?;
    weights.copy_from_host(&stream, weight_bytes)?;

    println!(
        "diagnostic real Q6_K BatchedGEMM: tensor=blk.0.ffn_down.weight format=Q6K ne0={ne0} ne1={ne1}/{full_ne1}"
    );

    for batch_size in [1usize, 2, 4] {
        let activations: Vec<f32> = (0..batch_size)
            .flat_map(|row| activation_row(row, ne0))
            .collect();
        let residual: Vec<f32> = (0..batch_size * ne1)
            .map(|index| 0.001 * (index + 1) as f32)
            .collect();
        let x = DeviceBuffer::alloc(Arc::clone(&device), f32_bytes(&activations).len())?;
        let out = DeviceBuffer::alloc(Arc::clone(&device), batch_size * ne1 * 4)?;
        let residual_dev = DeviceBuffer::alloc(Arc::clone(&device), f32_bytes(&residual).len())?;
        x.copy_from_host(&stream, &f32_bytes(&activations))?;
        residual_dev.copy_from_host(&stream, &f32_bytes(&residual))?;

        gemm.gemm(
            &stream,
            &weights,
            &x,
            &out,
            ne0,
            ne1,
            batch_size,
            GemvFormat::Q6K,
        )?;
        let mut raw = vec![0u8; batch_size * ne1 * 4];
        out.copy_to_host(&stream, &mut raw)?;
        let plain_gpu = bytes_f32(&raw);

        let mut expected = vec![0.0f32; ne1];
        let tensor = Tensor {
            ty: TensorType::Q6K,
            data: weight_bytes,
            ne0,
            ne1,
            n_rot: 0,
        };
        let mut plain_cpu = Vec::with_capacity(batch_size * ne1);
        for row in 0..batch_size {
            matmul(
                &mut expected,
                &tensor,
                &activations[row * ne0..(row + 1) * ne0],
            );
            plain_cpu.extend_from_slice(&expected);
        }
        let plain_metric = rel_l2(&plain_gpu, &plain_cpu);

        gemm.gemm_with_residual(
            &stream,
            &weights,
            &x,
            &out,
            ne0,
            ne1,
            batch_size,
            GemvFormat::Q6K,
            Some(&residual_dev),
        )?;
        out.copy_to_host(&stream, &mut raw)?;
        let residual_gpu = bytes_f32(&raw);
        let residual_cpu: Vec<f32> = plain_cpu
            .iter()
            .zip(&residual)
            .map(|(value, addend)| value + addend)
            .collect();
        let residual_metric = rel_l2(&residual_gpu, &residual_cpu);

        println!(
            "batch_size={batch_size} dispatch=Q6K ne0={ne0} ne1={ne1} plain_rel_l2={plain_metric:.6e} residual_rel_l2={residual_metric:.6e}"
        );
        assert!(plain_metric.is_finite());
        assert!(residual_metric.is_finite());
    }
    Ok(())
}
