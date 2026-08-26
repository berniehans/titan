//! VRAM footprint and per-kernel worst-case budget guard test (Phase 6.7, Group 3).
//!
//! Validates:
//! 1. Static declared worst-case VRAM model (pingpong + kv_pool + activations + logits) <= 5.2 GB.
//! 2. Live fixture ForwardDriver runtime VRAM footprint calculation and budget assertion <= 5.2 GB.
//! 3. Intentional budget overflow triggers a typed EngineError.

use engine_core::forward_driver::{ForwardDriver, VRAM_BUDGET_BYTES, VramFootprint};
use engine_io::{GgufReader, ModelConfig, load_to_pinned};
use std::path::PathBuf;

fn get_fixture_path() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("ENGINE_TESTDATA") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }
    let md = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for c in [
        md.join("../../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        md.join("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
    ] {
        if c.exists() {
            return Some(c.canonicalize().unwrap_or(c));
        }
    }
    None
}

#[test]
fn test_static_declared_worst_case_vram_model() {
    // Model configuration corresponding to Qwen3-0.6B (28 layers, hidden=1024, ff=3072, n_head=16, n_head_kv=8, head_dim=128, vocab=151936)
    let n_layer = 28usize;
    let hidden = 1024usize;
    let head_dim = 128usize;
    let n_head = 16usize;
    let n_head_kv = 8usize;
    let intermediate = 3072usize;
    let vocab_size = 151936usize;
    let capacity_tokens = 2048usize;

    // 1. Weights / ping-pong estimate (Q4_K for q, k, o, gate, up; F32 norms)
    // Q4_K block: 256 weights -> 144 bytes = 0.5625 bytes/weight
    let wq_bytes = (n_head * head_dim * hidden * 144) / 256;
    let wk_bytes = (n_head_kv * head_dim * hidden * 144) / 256;
    let wo_bytes = (hidden * n_head * head_dim * 144) / 256;
    let wgate_bytes = (intermediate * hidden * 144) / 256;
    let wup_bytes = (intermediate * hidden * 144) / 256;
    let norms_bytes = (hidden + head_dim + head_dim + hidden) * 4;
    let per_layer_weight_bytes =
        wq_bytes + wk_bytes + wo_bytes + wgate_bytes + wup_bytes + norms_bytes;
    let static_weights_bytes = per_layer_weight_bytes * n_layer;

    // 2. KV Cache pool estimate: 28 layers * 2048 tokens * (2 * 8 * 128) floats * 4 bytes
    let floats_per_token = 2 * n_head_kv * head_dim;
    let static_kv_pool_bytes = n_layer * capacity_tokens * floats_per_token * 4;

    // 3. Transient activations / scratch buffers
    let scratch_floats = hidden * 4 // x, input_norm, op, h1
        + (n_head * head_dim) * 2 // q, attn
        + (n_head_kv * head_dim) * 2 // k, v
        + head_dim // head scratch
        + hidden // ffin
        + intermediate * 3 // gate, up, proj
        + hidden + head_dim + intermediate; // zh, zhd, zff
    let static_activations_bytes = scratch_floats * 4 + 4; // + bt_dev (4 bytes)

    // 4. Logits buffer
    let static_logits_bytes = vocab_size * 4;

    let footprint = VramFootprint {
        pingpong_bytes: static_weights_bytes,
        kv_pool_bytes: static_kv_pool_bytes,
        activations_bytes: static_activations_bytes,
        logits_bytes: static_logits_bytes,
    };

    footprint.print_trace();

    assert!(
        footprint.total() <= VRAM_BUDGET_BYTES,
        "Static worst-case VRAM footprint {} bytes exceeded budget {} bytes",
        footprint.total(),
        VRAM_BUDGET_BYTES
    );
    assert!(footprint.assert_within_budget(VRAM_BUDGET_BYTES).is_ok());
    assert!(footprint.assert_within_budget(1024).is_err());
}

#[test]
#[ignore]
fn test_vram_guard_live_fixture_budget_trace() {
    let fixture_path = match get_fixture_path() {
        Some(p) => p,
        None => {
            eprintln!("test skipped: fixture not found");
            return;
        }
    };

    let reader = GgufReader::open(&fixture_path).expect("open fixture");
    let cfg = ModelConfig::from_reader(&reader).expect("parse config");
    let pinned = load_to_pinned(&reader, &fixture_path).expect("load to pinned");

    let mut driver = ForwardDriver::new(&reader, &pinned, &cfg, 2048).expect("create driver");
    let footprint = driver.vram_footprint();

    footprint.print_trace();

    assert!(
        footprint.total() <= VRAM_BUDGET_BYTES,
        "Live driver VRAM footprint {} bytes exceeded budget {}",
        footprint.total(),
        VRAM_BUDGET_BYTES
    );
    assert!(footprint.pingpong_bytes > 0);
    assert!(footprint.kv_pool_bytes > 0);
    assert!(footprint.activations_bytes > 0);
    assert!(footprint.logits_bytes > 0);
    assert!(footprint.assert_within_budget(VRAM_BUDGET_BYTES).is_ok());
    assert!(footprint.assert_within_budget(100).is_err());

    // Run one decode step to prove execution under guarded footprint
    let logits = driver.decode(9707).expect("run decode step");
    assert_eq!(logits.len(), cfg.vocab_size as usize);
}
