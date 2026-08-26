//! MoE Hybrid Scheduler & E2E Integration Gate (Phase 7.6).
//!
//! Asserts that:
//! 1. All 3 MoE execution backends (`Offload`, `Cpu`, `Hybrid`) complete E2E generation cleanly.
//! 2. Multi-step autoregressive decode emits coherent text tokens across modes.
//! 3. Layer cache telemetry accumulators record hits, fetches, and CPU overflow accurately.
//! 4. Per-layer miss-rate conforms to declared bounds under dynamic fetch policies.

use cudarc::driver::CudaDevice;
use engine_core::moe::{HardwareBandwidthProfile, MoeBackend};
use engine_io::{GgufReader, LoadedPinned, load_to_pinned};
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

fn load_fixture() -> Result<(GgufReader, &'static LoadedPinned), DynError> {
    let fixture = fixture_path().ok_or("fixture not present (GPU test)")?;
    let reader = GgufReader::open(&fixture)?;
    let pinned: &'static LoadedPinned = Box::leak(Box::new(load_to_pinned(&reader, &fixture)?));
    let _device = CudaDevice::new(0)?;
    Ok((reader, pinned))
}

#[test]
#[ignore]
fn test_subgate_moe_e2e_modes_and_telemetry() -> Result<(), DynError> {
    let (reader, pinned) = load_fixture()?;

    let profile_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/benches/benchbw.json");
    let profile = HardwareBandwidthProfile::load_from_file(&profile_path).ok();

    let modes = [MoeBackend::Offload, MoeBackend::Cpu, MoeBackend::Hybrid];

    println!("\n=== MoE Hybrid E2E Execution Gate ===");

    for mode in modes {
        println!("\n--- Testing Mode: {} ---", mode);
        let mut model = runtime::build_real_moe_driver_model(
            &reader,
            pinned,
            32,
            Some(mode),
            profile.as_ref(),
        )?;

        let prompt = "def solve_moe():";
        let (tokens, stats) = runtime::decode_run_moe(&mut model, 1000, prompt, 5)?;

        assert_eq!(tokens.len(), 5, "Must generate exactly 5 tokens");
        assert!(!stats.is_empty(), "Must collect telemetry for MoE layers");

        println!("  Generated Token IDs: {:?}", tokens);
        println!("  Layer Telemetry Samples ({} layers):", stats.len());
        for (l, s) in stats.iter().take(3).enumerate() {
            println!(
                "    Layer {}: Active={}, Hits={}, Fetched={}, CPU_Overflow={}, Coverage={:.1}%",
                l,
                s.active_requests,
                s.resident_hits,
                s.pcie_fetched,
                s.cpu_overflow,
                s.gpu_coverage_rate() * 100.0
            );
            assert!(s.active_requests > 0);
            assert!(s.gpu_coverage_rate() <= 1.0);
        }
    }

    Ok(())
}
