//! Diagnostic-only real-model differential: Q8 activation path vs direct FP32.
//! No production code, dispatch, kernel, threshold, or benchmark behavior is changed.

use cudarc::driver::CudaDevice;
use engine_core::forward_cpu::{Tensor, TensorType, matmul};
use engine_cuda::{BatchedGEMM, CudaError, CudaStream, DeviceBuffer, GemvFormat};
use engine_io::{GgmlType, GgufReader, LoadedPinned, load_to_pinned};
use serde_json::json;
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
fn tensor<'a>(r: &GgufReader, p: &'a LoadedPinned, name: &str) -> Tensor<'a> {
    let i = r.get_tensor(name).expect("missing tensor");
    Tensor {
        ty: match i.ggml_type {
            GgmlType::Q4_K => TensorType::Q4K,
            GgmlType::Q6_K => TensorType::Q6K,
            GgmlType::F32 => TensorType::F32,
            x => panic!("unsupported {x:?}"),
        },
        data: p.tensor(name).expect("tensor bytes"),
        ne0: i.dims[0] as usize,
        ne1: i.dims[1] as usize,
        n_rot: 0,
    }
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn metric(a: &[f32], b: &[f32]) -> (f64, f64, bool) {
    let finite = a.iter().chain(b).all(|x| x.is_finite()) && a.len() == b.len();
    if !finite {
        return (f64::NAN, f64::NAN, false);
    }
    let (mut n, mut d, mut dot, mut aa, mut bb) = (0., 0., 0., 0., 0.);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        n += (x - y) * (x - y);
        d += y * y;
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    (
        if d > 0. { (n / d).sqrt() } else { 0. },
        if aa > 0. && bb > 0. {
            dot / (aa * bb).sqrt()
        } else {
            1.
        },
        true,
    )
}
fn upload(s: &CudaStream, d: &Arc<CudaDevice>, v: &[f32]) -> Result<DeviceBuffer, CudaError> {
    let b = DeviceBuffer::alloc(d.clone(), v.len() * 4)?;
    b.copy_from_host(s, &bytes(v))?;
    Ok(b)
}
fn download(s: &CudaStream, b: &DeviceBuffer, n: usize) -> Result<Vec<f32>, CudaError> {
    s.sync()?;
    let mut x = vec![0; n * 4];
    b.copy_to_host(s, &mut x)?;
    Ok(f32s(&x))
}
fn download_bytes(s: &CudaStream, b: &DeviceBuffer, n: usize) -> Result<Vec<u8>, CudaError> {
    s.sync()?;
    let mut x = vec![0; n];
    b.copy_to_host(s, &mut x)?;
    Ok(x)
}
fn report(
    stages: &mut Vec<serde_json::Value>,
    name: &str,
    a: &[f32],
    b: &[f32],
    shape: String,
    batch: usize,
) {
    let (rel, cos, finite) = metric(a, b);
    println!(
        "stage={name} shape={shape} batch={batch} rel_l2={rel:.6e} cosine={cos:.9} finite={finite}"
    );
    stages.push(json!({"stage":name,"shape":shape,"batch":batch,"relative_l2":rel,"cosine":cos,"finite":finite}));
}

fn unsupported(stages: &mut Vec<serde_json::Value>, name: &str, shape: String, reason: &str) {
    println!("stage={name} shape={shape} unsupported: {reason}");
    stages.push(json!({"stage":name,"shape":shape,"status":"unsupported","reason":reason}));
}

#[test]
#[ignore] // Requires CUDA/NVRTC and the real Qwen3 GGUF fixture.
fn real_q8_vs_f32_differential() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    engine_cuda::ensure_cuda_dll_paths();
    let Some(path) = fixture() else {
        eprintln!("SKIP: Qwen3 fixture not present");
        return Ok(());
    };
    let r = GgufReader::open(&path)?;
    let p = load_to_pinned(&r, &path)?;
    let c = engine_io::ModelConfig::from_reader(&r)?;
    let h = c.hidden_size as usize;
    let dsize = r.get_tensor("blk.0.ffn_gate.weight").unwrap().dims[1] as usize;
    let input = engine_core::forward_cpu::embed_lookup(&tensor(&r, &p, "token_embd.weight"), 9707);
    let gate = tensor(&r, &p, "blk.0.ffn_gate.weight");
    let up = tensor(&r, &p, "blk.0.ffn_up.weight");

    let dev = CudaDevice::new(0)?;
    let s = CudaStream::new(dev.clone())?;
    let gemm = BatchedGEMM::new(dev.clone())?;
    let x = upload(&s, &dev, &input)?;
    let qx = DeviceBuffer::alloc(dev.clone(), h)?;
    let qd = DeviceBuffer::alloc(dev.clone(), (h / 32) * 4)?;
    let qs = DeviceBuffer::alloc(dev.clone(), (h / 32) * 4)?;
    gemm.quantize_q8_1(&s, &x, None, &qx, &qd, &qs, h, c.rms_norm_eps)?;
    // qx is int8 bytes; do not use the f32 helper (which requests h * 4 bytes).
    // No existing API exposes enough metadata for an honest dequantization here.
    let _qx_bytes = download_bytes(&s, &qx, h)?;
    let mut stages = Vec::new();
    report(
        &mut stages,
        "real_hidden_activation",
        &input,
        &input,
        format!("[{h}]"),
        1,
    );
    unsupported(
        &mut stages,
        "q8_activation_output",
        format!("[{h}]"),
        "qx is int8; no documented dequantized representation is exposed by existing APIs",
    );
    let gate_dev = DeviceBuffer::alloc(dev.clone(), gate.data.len())?;
    gate_dev.copy_from_host(&s, gate.data)?;
    let up_dev = DeviceBuffer::alloc(dev.clone(), up.data.len())?;
    up_dev.copy_from_host(&s, up.data)?;
    let go = DeviceBuffer::alloc(dev.clone(), dsize * 4)?;
    let uo = DeviceBuffer::alloc(dev.clone(), dsize * 4)?;
    gemm.gemm_q8_act_with_residual(
        &s,
        &gate_dev,
        &qx,
        &qd,
        &qs,
        &go,
        h,
        dsize,
        1,
        GemvFormat::Q4K,
        None,
    )?;
    gemm.gemm_q8_act_with_residual(
        &s,
        &up_dev,
        &qx,
        &qd,
        &qs,
        &uo,
        h,
        dsize,
        1,
        GemvFormat::Q4K,
        None,
    )?;
    let mut gc = vec![0.; dsize];
    let mut uc = vec![0.; dsize];
    matmul(&mut gc, &gate, &input);
    matmul(&mut uc, &up, &input);
    let gg = download(&s, &go, dsize)?;
    let ug = download(&s, &uo, dsize)?;
    report(
        &mut stages,
        "q8_gate_vs_direct_f32",
        &gg,
        &gc,
        format!("[1,{dsize}]"),
        1,
    );
    report(
        &mut stages,
        "q8_up_vs_direct_f32",
        &ug,
        &uc,
        format!("[1,{dsize}]"),
        1,
    );

    unsupported(
        &mut stages,
        "fused_vs_individual_q8_product",
        format!("[1,{dsize}]"),
        "separate fused gate/up product output is not exposed",
    );
    unsupported(
        &mut stages,
        "q8_ffn_vs_direct_f32_ffn",
        format!("[1,{dsize}]"),
        "full FFN comparison is unavailable without duplicating production code",
    );
    let artifact = json!({"schema_version":1,"status":"diagnostic_only","model_path":path,"model_identity":"unsupported: GGUF identity API not exposed","dimensions":{"hidden":h,"intermediate":dsize,"batch":1},"stages":stages,"conclusion":"unsupported comparison: full-layer output unavailable without duplicating production code; inspect first failing stage to distinguish ABI/kernel, activation approximation, or later residual/buffer mismatch"});
    let out = std::env::var_os("TITAN_Q8_F32_DIAGNOSTIC_JSON")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../local-artifacts/reviews/real-q8-vs-f32-differential.json")
        });
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out, serde_json::to_vec_pretty(&artifact)?)?;
    println!("diagnostic artifact: {}", out.display());
    Ok(())
}
