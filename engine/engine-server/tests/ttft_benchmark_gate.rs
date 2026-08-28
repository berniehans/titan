//! TTFT Speedup Benchmark Gate (Phase 11, Sub-change 11.4).
//!
//! Benchmarks Time-To-First-Token (TTFT) across sequence lengths S in {16, 64, 128, 256}
//! comparing Chunked Prefill (Batched GEMM + FlashAttention-2) against Serial Prefill.

use engine_core::ForwardDriver;
use engine_core::tokenizer::BpeTokenizer;
use engine_io::{GgufReader, ModelConfig};
use std::path::PathBuf;
use std::time::Instant;

type DynError = Box<dyn std::error::Error + Send + Sync>;

fn fixture_path() -> PathBuf {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop();
    root.pop();
    root.join("testdata").join("Qwen3-0.6B-Q4_K_M.gguf")
}

#[test]
#[ignore]
fn test_ttft_speedup_benchmark() -> Result<(), DynError> {
    engine_cuda::ensure_cuda_dll_paths();
    let fix = fixture_path();
    let reader = GgufReader::open(&fix)?;
    let pinned = engine_io::load_to_pinned(&reader, &fix)?;
    let cfg = ModelConfig::from_reader(&reader)?;
    let tok = BpeTokenizer::from_reader(&reader)?;

    println!("\n==========================================================================");
    println!("  TITAN PHASE 11 BENCHMARK: TTFT & CHUNKED PREFILL SPEEDUP (28 LAYERS)   ");
    println!("==========================================================================");

    let test_lengths = [16, 64, 128, 256];
    let base_text = "The quick brown fox jumps over the lazy dog. In astronomy, Kepler discovered planetary motion laws. Machine learning transforms automated systems. ";
    let base_tokens = tok.encode(base_text)?;

    for &target_len in &test_lengths {
        let mut prompt_tokens = Vec::with_capacity(target_len);
        while prompt_tokens.len() < target_len {
            prompt_tokens.extend_from_slice(&base_tokens);
        }
        prompt_tokens.truncate(target_len);

        // 1. Measure Serial Prefill TTFT
        let mut drv_serial = ForwardDriver::new(&reader, &pinned, &cfg, target_len)?;
        let t0 = Instant::now();
        let _ = drv_serial.prefill(&prompt_tokens)?;
        let serial_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // 2. Measure Chunked Prefill TTFT (chunk_size = 128)
        let mut drv_chunked = ForwardDriver::new(&reader, &pinned, &cfg, target_len)?;
        let t1 = Instant::now();
        let _ = drv_chunked.prefill_chunked(&prompt_tokens, 128)?;
        let chunked_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let speedup = serial_ms / chunked_ms.max(0.001);
        let tok_per_sec = (target_len as f64) / (chunked_ms / 1000.0);

        println!(
            "| Prompt: {:>4} tok | Serial TTFT: {:>7.2} ms | Chunked TTFT: {:>7.2} ms | Speedup: {:>5.2}x | Prefill Throughput: {:>6.1} tok/s |",
            target_len, serial_ms, chunked_ms, speedup, tok_per_sec
        );

        assert!(
            chunked_ms <= serial_ms * 1.05,
            "Chunked prefill ({chunked_ms:.2} ms) is slower than serial ({serial_ms:.2} ms)"
        );
    }
    println!("==========================================================================\n");

    Ok(())
}
