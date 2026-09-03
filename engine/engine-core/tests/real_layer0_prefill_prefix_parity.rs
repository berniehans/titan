//! Diagnostic parity for the real layer-0 prefill prefix.
//!
//! This intentionally stops after Q/K post-RoPE. It mirrors the prefill calls
//! in `ForwardDriver`: embedding lookup, input RMSNorm, three batched GEMMs,
//! then the public batched Q/K norm+RoPE launcher.

use cudarc::driver::CudaDevice;
use engine_core::forward_cpu::{
    Tensor, TensorType, embed_lookup, matmul, rms_norm, rope_neox_partial,
};
use engine_cuda::{
    BatchedGEMM, CudaError, CudaStream, DeviceBuffer, GemvFormat, MODE_BROADCAST_RESIDUAL,
    MODE_NORM, MODE_ROPE, NormRope,
};
use engine_io::{GgmlType, GgufReader, LoadedPinned, ModelConfig, load_to_pinned};
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

fn bank_type(ty: GgmlType) -> TensorType {
    match ty {
        GgmlType::F32 => TensorType::F32,
        GgmlType::Q4_K => TensorType::Q4K,
        GgmlType::Q6_K => TensorType::Q6K,
        other => panic!("unsupported diagnostic GGML type: {other:?}"),
    }
}

fn gemm_format(ty: GgmlType) -> GemvFormat {
    match ty {
        GgmlType::Q4_K => GemvFormat::Q4K,
        GgmlType::Q6_K => GemvFormat::Q6K,
        other => panic!("unsupported diagnostic GEMM type: {other:?}"),
    }
}

fn tensor<'a>(reader: &GgufReader, pinned: &'a LoadedPinned, name: &str) -> Tensor<'a> {
    let info = reader
        .get_tensor(name)
        .unwrap_or_else(|| panic!("missing {name}"));
    assert_eq!(info.dims.len(), 2, "{name} must be 2-D");
    Tensor {
        ty: bank_type(info.ggml_type),
        data: pinned
            .tensor(name)
            .unwrap_or_else(|| panic!("missing bytes for {name}")),
        ne0: info.dims[0] as usize,
        ne1: info.dims[1] as usize,
        n_rot: 0,
    }
}

fn f32_vector(pinned: &LoadedPinned, name: &str) -> Vec<f32> {
    pinned
        .tensor(name)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_f32(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn upload(
    stream: &CudaStream,
    device: &Arc<CudaDevice>,
    v: &[f32],
) -> Result<DeviceBuffer, CudaError> {
    let b = DeviceBuffer::alloc(Arc::clone(device), v.len() * 4)?;
    b.copy_from_host(stream, &f32_bytes(v))?;
    Ok(b)
}
fn download(stream: &CudaStream, b: &DeviceBuffer, n: usize) -> Result<Vec<f32>, CudaError> {
    let mut raw = vec![0u8; n * 4];
    b.copy_to_host(stream, &mut raw)?;
    Ok(bytes_f32(&raw))
}
fn metrics(actual: &[f32], expected: &[f32]) -> (f64, f64) {
    let (mut n, mut d, mut dot, mut aa, mut bb) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&a, &e) in actual.iter().zip(expected) {
        let (a, e) = (a as f64, e as f64);
        n += (a - e) * (a - e);
        d += e * e;
        dot += a * e;
        aa += a * a;
        bb += e * e;
    }
    ((n / d).sqrt(), dot / (aa * bb).sqrt())
}

#[test]
#[ignore]
fn real_layer0_prefill_prefix_matches_cpu() -> Result<(), CudaError> {
    engine_cuda::ensure_cuda_dll_paths();
    let Some(path) = fixture_path() else {
        eprintln!("SKIP: Qwen3 fixture not present");
        return Ok(());
    };
    let reader = GgufReader::open(&path).expect("open Qwen3 fixture");
    let pinned = load_to_pinned(&reader, &path).expect("load Qwen3 fixture");
    let cfg = ModelConfig::from_reader(&reader).expect("read GGUF config");
    let h = cfg.hidden_size as usize;
    let hd = cfg.head_dim as usize;
    let nh = cfg.n_head as usize;
    let nkv = cfg.n_head_kv as usize;
    let qdim = nh * hd;
    let kvd = nkv * hd;
    let tokens = [9707usize, 1128, 13, 198];
    let batch = tokens.len();
    let eps = cfg.rms_norm_eps;
    let base = cfg.rope_freq_base;
    assert_eq!(tensor(&reader, &pinned, "token_embd.weight").ne0, h);
    assert_eq!(f32_vector(&pinned, "blk.0.attn_norm.weight").len(), h);
    assert_eq!(f32_vector(&pinned, "blk.0.attn_q_norm.weight").len(), hd);
    assert_eq!(f32_vector(&pinned, "blk.0.attn_k_norm.weight").len(), hd);
    let emb = tensor(&reader, &pinned, "token_embd.weight");
    let an = f32_vector(&pinned, "blk.0.attn_norm.weight");
    let qn = f32_vector(&pinned, "blk.0.attn_q_norm.weight");
    let kn = f32_vector(&pinned, "blk.0.attn_k_norm.weight");
    let q = tensor(&reader, &pinned, "blk.0.attn_q.weight");
    let k = tensor(&reader, &pinned, "blk.0.attn_k.weight");
    let v = tensor(&reader, &pinned, "blk.0.attn_v.weight");
    assert_eq!((q.ne0, k.ne0, v.ne0), (h, h, h));
    assert_eq!((q.ne1, k.ne1, v.ne1), (qdim, kvd, kvd));
    println!(
        "config hidden={h} head_dim={hd} q_heads={nh} kv_heads={nkv} batch={batch} positions=0..{} eps={eps} base={base}",
        batch - 1
    );

    let mut x = Vec::with_capacity(batch * h);
    for &token_id in &tokens {
        x.extend(embed_lookup(&emb, token_id));
    }
    println!(
        "stage=embed shape=[{batch},{h}] format={:?} batch={batch}",
        reader.get_tensor("token_embd.weight").unwrap().ggml_type
    );
    let mut norm_cpu = Vec::with_capacity(batch * h);
    for row in x.chunks_exact(h) {
        norm_cpu.extend(rms_norm(row, &an, eps));
    }
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let nr = NormRope::new(Arc::clone(&device))?;
    let gemm = BatchedGEMM::new(Arc::clone(&device))?;
    let x_dev = upload(&stream, &device, &x)?;
    let zero_h = upload(&stream, &device, &vec![0.0; h])?;
    let norm_dev = upload(&stream, &device, &vec![0.0; batch * h])?;
    nr.launch_batched_with_pos_ptr(
        &stream,
        &x_dev,
        &zero_h,
        &upload(&stream, &device, &an)?,
        &norm_dev,
        &norm_dev,
        eps,
        h,
        0,
        base,
        0,
        MODE_NORM | MODE_BROADCAST_RESIDUAL,
        None,
        batch,
        1,
    )?;
    let norm_gpu = download(&stream, &norm_dev, batch * h)?;
    let (rel, cos) = metrics(&norm_gpu, &norm_cpu);
    println!(
        "stage=input_rmsnorm shape=[{batch},{h}] format=F32 batch={batch} rel_l2={rel:.6e} cosine={cos:.9}"
    );
    assert!(rel < 2e-4 && cos > 0.999999, "input RMSNorm parity failed");

    let mut projected = Vec::new();
    for (role, w, out_dim) in [("Q", q, qdim), ("K", k, kvd), ("V", v, kvd)] {
        let info = reader
            .get_tensor(&format!("blk.0.attn_{}.weight", role.to_lowercase()))
            .unwrap();
        let w_dev = DeviceBuffer::alloc(Arc::clone(&device), w.data.len())?;
        w_dev.copy_from_host(&stream, w.data)?;
        let out_dev = upload(&stream, &device, &vec![0.0; batch * out_dim])?;
        gemm.gemm(
            &stream,
            &w_dev,
            &norm_dev,
            &out_dev,
            h,
            out_dim,
            batch,
            gemm_format(info.ggml_type),
        )?;
        let gpu = download(&stream, &out_dev, batch * out_dim)?;
        let mut cpu = Vec::new();
        for row in norm_cpu.chunks_exact(h) {
            let mut dst = vec![0.0; out_dim];
            matmul(&mut dst, &w, row);
            cpu.extend(dst);
        }
        let (rel, cos) = metrics(&gpu, &cpu);
        println!(
            "stage={role}_gemm shape=[{batch},{out_dim}] format={:?} batch={batch} rel_l2={rel:.6e} cosine={cos:.9}",
            info.ggml_type
        );
        assert!(rel < 2e-4 && cos > 0.999999, "{role} GEMM parity failed");
        projected.push(gpu);
    }
    for (role, input, weights, heads) in [
        ("Q", &projected[0], &qn, nh),
        ("K", &projected[1], &kn, nkv),
    ] {
        let rows = batch * heads;
        let input_dev = upload(&stream, &device, input)?;
        let residual = upload(&stream, &device, &vec![0.0; heads * hd])?;
        let weights_dev = upload(&stream, &device, weights)?;
        let out_dev = upload(&stream, &device, &vec![0.0; rows * hd])?;
        nr.launch_batched_with_pos_ptr(
            &stream,
            &input_dev,
            &residual,
            &weights_dev,
            &out_dev,
            &out_dev,
            eps,
            hd,
            hd,
            base,
            0,
            MODE_NORM | MODE_ROPE | MODE_BROADCAST_RESIDUAL,
            None,
            rows,
            heads,
        )?;
        let gpu = download(&stream, &out_dev, rows * hd)?;
        let mut cpu = Vec::new();
        for row in 0..rows {
            let n = rms_norm(&input[row * hd..(row + 1) * hd], weights, eps);
            cpu.extend(rope_neox_partial(&n, (row / heads) as u32, hd, base));
        }
        let (rel, cos) = metrics(&gpu, &cpu);
        println!(
            "stage={role}_post_rope shape=[tokens={batch},heads={heads},head_dim={hd}] format=F32 batch={rows} positions=0..{} rel_l2={rel:.6e} cosine={cos:.9}",
            batch - 1
        );
        assert!(
            rel < 2e-4 && cos > 0.999999,
            "{role} post-RoPE parity failed"
        );
    }
    Ok(())
}
