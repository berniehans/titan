//! Real VRAM audit and stage numbers verification (Phase 6.9, Group 2).
//!
//! Measures real device memory (VRAM) allocated during live generation on the
//! Qwen3-0.6B-Q4_K_M fixture and asserts that working set <= 5.2 GB budget.

use cudarc::driver::CudaDevice;
use engine_core::forward_driver::VRAM_BUDGET_BYTES;
use engine_core::vram_accounting::VramStageBreakdown;
use engine_io::{GgufReader, load_to_pinned};
use engine_server::runtime;
use std::path::PathBuf;

type DynError = Box<dyn std::error::Error + Send + Sync>;

fn fixture_path() -> Option<PathBuf> {
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

#[test]
#[ignore]
fn test_subgate_vram_real_audit_working_set() -> Result<(), DynError> {
    let fix = fixture_path().ok_or("fixture missing (GPU test)")?;
    let reader = GgufReader::open(&fix)?;
    let pinned = load_to_pinned(&reader, &fix)?;
    let _device = CudaDevice::new(0)?;

    const KV_BLOCK_TOKENS: usize = 128;
    let mut model = runtime::build_real_driver_model(&reader, &pinned, KV_BLOCK_TOKENS)?;

    // Run a real generation step to ensure all kernel device buffers and KV cache slots are active
    let generated = runtime::decode_run(&mut model, 151936, "Hello", 3)?;
    assert!(!generated.is_empty(), "must generate tokens");

    let driver = model.driver.as_ref().expect("driver must be initialized");

    let footprint = driver.vram_footprint();
    let breakdown: VramStageBreakdown = footprint.into();

    println!("\n{}", breakdown.format_trace(VRAM_BUDGET_BYTES));

    let tot = breakdown.total_bytes();
    println!("\n=== Measured Stage Totals (Phase 6.9) ===");
    println!(
        "  Ping-pong / Weights: {:>10} B ({:>7.2} MB)",
        breakdown.pingpong_bytes,
        breakdown.pingpong_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  Resident KV Cache:   {:>10} B ({:>7.2} MB)",
        breakdown.kv_pool_bytes,
        breakdown.kv_pool_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  Activation Scratch:  {:>10} B ({:>7.2} MB)",
        breakdown.activations_bytes,
        breakdown.activations_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  Logits Transfer:     {:>10} B ({:>7.2} MB)",
        breakdown.logits_bytes,
        breakdown.logits_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("  ---------------------------------------------");
    println!(
        "  Total Working Set:   {:>10} B ({:>7.2} MB, {:.2} GB)",
        tot,
        tot as f64 / (1024.0 * 1024.0),
        tot as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "  VRAM Budget Bound:   {:>10} B ({:>7.2} MB, {:.2} GB)",
        VRAM_BUDGET_BYTES,
        VRAM_BUDGET_BYTES as f64 / (1024.0 * 1024.0),
        VRAM_BUDGET_BYTES as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "  Budget Utilization:  {:>7.2}%",
        breakdown.budget_utilization(VRAM_BUDGET_BYTES) * 100.0
    );

    // Gate assertions
    assert!(
        tot <= VRAM_BUDGET_BYTES,
        "Measured working set ({tot} bytes) exceeded 5.2 GB budget ({VRAM_BUDGET_BYTES})"
    );
    assert!(breakdown.pingpong_bytes > 0);
    assert!(breakdown.kv_pool_bytes > 0);
    assert!(breakdown.activations_bytes > 0);
    assert!(breakdown.logits_bytes > 0);

    println!("\nSub-gate 6.9 VRAM Audit PASS: Measured working set <= 5.2 GB budget.");
    Ok(())
}
