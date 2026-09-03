//! Real-tensor parity (teacher-forced) — tasks 3.1/3.2/3.3 of change 6.3.
//!
//! Runs the device kernel through `MultiFormatGEMV` on a REAL tensor read from
//! the pinned fixture `Qwen3-0.6B-Q4_K_M.gguf`, versus the CPU forward bank
//! (6.2) as the reference authority (`#[ignore]` GPU test; requires the
//! `%LOCALAPPDATA%/Temp` NVRTC PATH trick).
//!
//! ## What is and isn't measurable on THIS fixture (structural honesty)
//!
//! - **`blk.0.attn_q.weight` is a real Q4_K tensor** (`GgmlType::Q4_K`,
//!   dims `[1024, 2048]`). Its GEMV is the one real-tensor comparison this
//!   kernel can make end-to-end: GPU `MultiFormatGEMV::Q4K` vs CPU
//!   `forward_cpu::matmul` over the same bytes. That is task 3.3's
//!   "first-block Q4 attention-weight matmul".
//!
//! - **`token_embd.weight` (embedding, task 3.1) is Q6_K, NOT Q8_0/F16.**
//!   `output.weight` does not exist (the head is TIED to `token_embd`, which
//!   Qwen3 GGUF represents as qwen3.tensor `token_embd.weight` = Q6_K). The
//!   `MultiFormatGEMV` kernel in this change ships Q4_K_M, Q8_0, and F16 paths
//!   only (per proposal non-goals / no over-engineering). A Q6_K tensor cannot
//!   be run through any of those three paths, so *embedding-row* and
//!   *output-head* real-tensor parity against THIS fixture is structurally
//!   impossible for the Q8_0/F16 paths. No numbers are invented for them.
//!
//! - **There is no Q8_0 and no F16 weight tensor anywhere in this fixture** —
//!   the type census is: F32 x113 (norm weights), Q4_K x168 (q/k/o/gate/up),
//!   Q6_K x29 (embd, attn_v, ffn_down). Validating the Q8_0/F16 *paths* is
//!   therefore done on controlled synthetic blocks (see `gemv_gpu.rs`,
//!   tasks 1.3/1.4), not on real GGUF data.
//!
//! - **llama golden L0 activations are NOT comparable to an attention Q GEMV.**
//!   The committed golden (`tests/fixtures/golden/activations/activations.json`)
//!   holds the *layer-0 residual-stream output* (post-attention+FFN), truncated
//!   to a 6-element vector by the 6.1 dump. An attention-query matmul has a
//!   different dimensionality (ne1 = n_heads*head_dim) and occupies a different
//!   position in the graph, so a direct cos-sim/rel-L2 against L0 is
//!   structurally unsound. (Full single-layer parity against the golden belongs
//!   to change 6.6.) The real-tensor check below therefore asserts GPU-vs-CPU
//!   agreement on the actual Q4_K attention weight — the honest, measurable
//!   teacher-forced statement for this change.

use cudarc::driver::CudaDevice;
use engine_core::forward_cpu::{Tensor, TensorType, matmul};
use engine_cuda::{CudaError, CudaStream, DeviceBuffer, GemvFormat, MultiFormatGEMV};
use engine_io::{GgufReader, load_to_pinned};
use std::path::PathBuf;
use std::sync::Arc;

/// Locates the gitignored real fixture (mirrors `engine-io/tests/loader_pinned.rs`).
fn get_fixture_path() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("ENGINE_TESTDATA") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        manifest_dir.join("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
    ];
    for c in &candidates {
        if c.exists() {
            return Some(c.canonicalize().unwrap_or_else(|_| c.clone()));
        }
    }
    None
}

/// Deterministic fp32 activation vector (reduction dim = ne0).
fn input_x(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 + 0.01 * (i as f32)).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}
fn bytes_f32(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// rel-L2 between two vectors.
fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64) - (*y as f64);
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    (num / den).sqrt() as f32
}

/// Cosine similarity.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    (dot / (na * nb).sqrt()) as f32
}

/// Runs a real Q4_K tensor slice through the GPU kernel and returns GPU+CPU out.
///
/// `n_first_out_blocks` bounds how many output columns we compare (first block
/// of the head), keeping the CPU reference fast while still exercising the
/// kernel against real quantized data.
fn real_q4k_gemv(
    gemv: &MultiFormatGEMV,
    stream: &CudaStream,
    w_bytes: &[u8],
    ne0: usize,
    n_out_cols: usize,
) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
    let x = input_x(ne0);
    // GPU path
    let x_dev = DeviceBuffer::alloc(Arc::clone(gemv.device()), x.len() * 4)?;
    x_dev.copy_from_host(stream, &f32_bytes(&x))?;
    let w_dev = DeviceBuffer::alloc(Arc::clone(gemv.device()), w_bytes.len())?;
    w_dev.copy_from_host(stream, w_bytes)?;
    let out_dev = DeviceBuffer::alloc(Arc::clone(gemv.device()), n_out_cols * 4)?;
    gemv.gemv(
        stream,
        GemvFormat::Q4K,
        &w_dev,
        &x_dev,
        &out_dev,
        ne0,
        n_out_cols,
    )?;
    let mut out_raw = vec![0u8; n_out_cols * 4];
    out_dev.copy_to_host(stream, &mut out_raw)?;
    let gpu = bytes_f32(&out_raw);

    // CPU reference with the 6.2 forward bank over the same real bytes.
    let col_bytes = (ne0 / 256) * 144;
    let t = Tensor {
        ty: TensorType::Q4K,
        data: &w_bytes[..col_bytes * n_out_cols],
        ne0,
        ne1: n_out_cols,
        n_rot: 0,
    };
    let mut cpu = vec![0.0f32; n_out_cols];
    matmul(&mut cpu, &t, &x);
    Ok((gpu, cpu))
}

#[test]
#[ignore]
fn real_q4k_attention_weight_gpu_matches_cpu_bank() -> Result<(), CudaError> {
    engine_cuda::ensure_cuda_dll_paths();
    let Some(fixture) = get_fixture_path() else {
        eprintln!("SKIP: fixture not present in this environment");
        return Ok(());
    };
    let reader = GgufReader::open(&fixture).expect("open GGUF");
    let loaded = load_to_pinned(&reader, &fixture).expect("load pinned");

    // Real first-block attention query weight: Q4_K, dims [1024, 2048].
    let info = reader
        .get_tensor("blk.0.attn_q.weight")
        .expect("blk.0.attn_q.weight present in fixture");
    assert_eq!(info.dims.len(), 2, "expected 2-D weight");
    let ne0 = info.dims[0] as usize;
    let ne1 = info.dims[1] as usize;
    assert_eq!((ne0, ne1), (1024, 2048), "blk.0.attn_q shape");
    assert_eq!(
        info.ggml_type,
        engine_io::GgmlType::Q4_K,
        "attn_q must be Q4_K"
    );

    let all_bytes = loaded.tensor("blk.0.attn_q.weight").expect("slice");
    // First output block: compare a bounded head of the query matrix.
    const N_OUT_COLS: usize = 512;
    let col_bytes = (ne0 / 256) * 144;
    let w_bytes = &all_bytes[..col_bytes * N_OUT_COLS];

    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let gemv = MultiFormatGEMV::new(Arc::clone(&device))?;
    let (gpu, cpu) = real_q4k_gemv(&gemv, &stream, w_bytes, ne0, N_OUT_COLS)?;

    let rl2 = rel_l2(&gpu, &cpu);
    let cs = cosine(&gpu, &cpu);
    println!(
        "REAL Q4_K blk.0.attn_q GEMV: {N_OUT_COLS} cols x {ne0} dims; rel-L2={rl2:.3e} cos-sim={cs:.6}"
    );

    assert!(
        rl2 < 1e-3,
        "real Q4_K GEMV rel-L2 {rl2:.3e} >= 1e-3 (GPU vs CPU bank)"
    );
    assert!(
        cs >= 0.999,
        "real Q4_K GEMV cos-sim {cs:.6} < 0.999 (GPU vs CPU bank)"
    );
    Ok(())
}

fn real_q6k_gemv(
    gemv: &MultiFormatGEMV,
    stream: &CudaStream,
    w_bytes: &[u8],
    ne0: usize,
    n_out_cols: usize,
) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
    let x = input_x(ne0);
    let x_dev = DeviceBuffer::alloc(Arc::clone(gemv.device()), x.len() * 4)?;
    x_dev.copy_from_host(stream, &f32_bytes(&x))?;
    let w_dev = DeviceBuffer::alloc(Arc::clone(gemv.device()), w_bytes.len())?;
    w_dev.copy_from_host(stream, w_bytes)?;
    let out_dev = DeviceBuffer::alloc(Arc::clone(gemv.device()), n_out_cols * 4)?;
    gemv.gemv(
        stream,
        GemvFormat::Q6K,
        &w_dev,
        &x_dev,
        &out_dev,
        ne0,
        n_out_cols,
    )?;
    let mut out_raw = vec![0u8; n_out_cols * 4];
    out_dev.copy_to_host(stream, &mut out_raw)?;
    let gpu = bytes_f32(&out_raw);

    let col_bytes = (ne0 / 256) * 210;
    let t = Tensor {
        ty: TensorType::Q6K,
        data: &w_bytes[..col_bytes * n_out_cols],
        ne0,
        ne1: n_out_cols,
        n_rot: 0,
    };
    let mut cpu = vec![0.0f32; n_out_cols];
    matmul(&mut cpu, &t, &x);
    Ok((gpu, cpu))
}

#[test]
#[ignore]
fn real_q6k_attn_v_and_ffn_down_gpu_matches_cpu_bank() -> Result<(), CudaError> {
    engine_cuda::ensure_cuda_dll_paths();
    let Some(fixture) = get_fixture_path() else {
        eprintln!("SKIP: fixture not present in this environment");
        return Ok(());
    };
    let reader = GgufReader::open(&fixture).expect("open GGUF");
    let loaded = load_to_pinned(&reader, &fixture).expect("load pinned");

    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let gemv = MultiFormatGEMV::new(Arc::clone(&device))?;

    // 1. blk.0.attn_v.weight (Q6_K)
    let info_v = reader.get_tensor("blk.0.attn_v.weight").expect("attn_v");
    let ne0_v = info_v.dims[0] as usize;
    let ne1_v = info_v.dims[1] as usize;
    let bytes_v = loaded.tensor("blk.0.attn_v.weight").expect("slice");

    for b in 0..4 {
        let blk = &bytes_v[b * 210..(b + 1) * 210];
        let d_bits = u16::from_le_bytes([blk[208], blk[209]]);
        let d = engine_core::forward_cpu::f16_to_f32(d_bits);
        let sc: Vec<i8> = blk[192..208].iter().map(|&s| s as i8).collect();
        println!(
            "Block {b}: d_bits={:#06x}, d={}, scales={:?}",
            d_bits, d, sc
        );
    }

    // Block 1 test
    let (gpu_b1, cpu_b1) = real_q6k_gemv(&gemv, &stream, &bytes_v[210..420], 256, 1)?;
    println!(
        "Block 1 attn_v: GPU={}, CPU={}, diff={}",
        gpu_b1[0],
        cpu_b1[0],
        (gpu_b1[0] - cpu_b1[0]).abs()
    );

    // Block 2 test
    let (gpu_b2, cpu_b2) = real_q6k_gemv(&gemv, &stream, &bytes_v[420..630], 256, 1)?;
    println!(
        "Block 2 attn_v: GPU={}, CPU={}, diff={}",
        gpu_b2[0],
        cpu_b2[0],
        (gpu_b2[0] - cpu_b2[0]).abs()
    );

    // Block 3 test
    let (gpu_b3, cpu_b3) = real_q6k_gemv(&gemv, &stream, &bytes_v[630..840], 256, 1)?;
    println!(
        "Block 3 attn_v: GPU={}, CPU={}, diff={}",
        gpu_b3[0],
        cpu_b3[0],
        (gpu_b3[0] - cpu_b3[0]).abs()
    );

    // Three blocks test
    let (gpu_v3, cpu_v3) = real_q6k_gemv(&gemv, &stream, &bytes_v[..630], 768, 1)?;
    println!(
        "Three blocks attn_v: GPU={}, CPU={}, diff={}",
        gpu_v3[0],
        cpu_v3[0],
        (gpu_v3[0] - cpu_v3[0]).abs()
    );

    // Four block (1 full col) test
    let (gpu_v4, cpu_v4) = real_q6k_gemv(&gemv, &stream, &bytes_v[..840], 1024, 1)?;
    println!(
        "Four block attn_v: GPU={}, CPU={}, diff={}",
        gpu_v4[0],
        cpu_v4[0],
        (gpu_v4[0] - cpu_v4[0]).abs()
    );

    let (gpu_v, cpu_v) = real_q6k_gemv(&gemv, &stream, bytes_v, ne0_v, ne1_v)?;
    let rl2_v = rel_l2(&gpu_v, &cpu_v);
    let cs_v = cosine(&gpu_v, &cpu_v);
    println!(
        "REAL Q6_K blk.0.attn_v GEMV: {ne1_v} cols x {ne0_v} dims; rel-L2={rl2_v:.3e} cos-sim={cs_v:.6}"
    );
    println!("GPU[:8]: {:?}", &gpu_v[..8]);
    println!("CPU[:8]: {:?}", &cpu_v[..8]);
    assert!(rl2_v < 1e-3, "real Q6_K attn_v rel-L2 >= 1e-3");
    assert!(cs_v >= 0.999, "real Q6_K attn_v cos-sim < 0.999");

    // 2. blk.0.ffn_down.weight (Q6_K)
    let info_down = reader
        .get_tensor("blk.0.ffn_down.weight")
        .expect("ffn_down");
    let ne0_down = info_down.dims[0] as usize;
    let ne1_down = info_down.dims[1] as usize;
    let bytes_down = loaded.tensor("blk.0.ffn_down.weight").expect("slice");
    let (gpu_down, cpu_down) = real_q6k_gemv(&gemv, &stream, bytes_down, ne0_down, ne1_down)?;
    let rl2_down = rel_l2(&gpu_down, &cpu_down);
    let cs_down = cosine(&gpu_down, &cpu_down);
    println!(
        "REAL Q6_K blk.0.ffn_down GEMV: {ne1_down} cols x {ne0_down} dims; rel-L2={rl2_down:.3e} cos-sim={cs_down:.6}"
    );
    assert!(rl2_down < 1e-3, "real Q6_K ffn_down rel-L2 >= 1e-3");
    assert!(cs_down >= 0.999, "real Q6_K ffn_down cos-sim < 0.999");

    Ok(())
}

/// Documents the structural impossibility of task 3.1 (embedding-row) and
/// 3.2 (output/logit-head) Q8_0/F16 real-tensor comparisons on this fixture.
///
/// The embedding and the tied output head are Q6_K, and the fixture contains
/// *no* Q8_0/F16 weight tensor, so a real GGUF Q8_0/F16 parity test is not
/// constructible here (the kernel has no Q6_K path by design scope). This test
/// pins those fixture facts so the reasoning stays executable — it intentionally
/// does NOT assert GPU glyphs on data that cannot exercise the Q8_0/F16 paths.
#[test]
#[ignore]
fn fixture_has_no_q8_f16_for_embedding_or_head() {
    let Some(fixture) = get_fixture_path() else {
        eprintln!("SKIP: fixture not present in this environment");
        return;
    };
    let reader = GgufReader::open(&fixture).expect("open GGUF");

    // output.weight absent: Qwen3 ties the head to token_embd.
    assert!(
        reader.get_tensor("output.weight").is_none(),
        "expected tied head (no output.weight) in this Qwen3 fixture"
    );
    let embd = reader
        .get_tensor("token_embd.weight")
        .expect("embedding present");
    assert_ne!(
        embd.ggml_type,
        engine_io::GgmlType::Q8_0,
        "embedding would have to be Q8_0 for task 3.1; it is not in this fixture"
    );
    assert_ne!(
        embd.ggml_type,
        engine_io::GgmlType::F16,
        "embedding would have to be F16 for task 3.1; it is not in this fixture"
    );

    // Census: confirm there is no Q8_0 / F16 weight tensor to compare against.
    let all = reader.tensor_infos();
    let q8 = all
        .iter()
        .filter(|t| t.ggml_type == engine_io::GgmlType::Q8_0)
        .count();
    let f16 = all
        .iter()
        .filter(|t| t.ggml_type == engine_io::GgmlType::F16)
        .count();
    println!(
        "fixture census: Q8_0 tensors = {q8}, F16 tensors = {f16} (expected 0/0); \
         embd ggml_type = {:?} (Q6_K) — Q8_0/F16 real-parity structurally impossible",
        embd.ggml_type
    );
    assert_eq!(q8, 0, "no Q8_0 weight tensor in this fixture");
    assert_eq!(f16, 0, "no F16 weight tensor in this fixture");
}
