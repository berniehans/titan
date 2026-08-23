//! KV append/read throughput bench — real generation-loop path
//! (f5-sse-server-batching, task 5.1).
//!
//! Measures the deferred Phase 4 number: the append/read throughput of the
//! paged KV cache *as it is actually driven by the generation loop* — i.e.
//! through `GenerationSession::step` / `BatchScheduler::advance`, which append
//! one token's key+value rows and read the row back per decode step — not a
//! raw flat `copy_from_slice` microbenchmark.
//!
//! Geometry mirrors a real attention workload (heads × head_dim floats per row)
//! and N concurrent sessions are multiplexed on one pool (continuous batching).
//! Reported metric = tokens/sec through the loop (append + read per token) and
//! aggregate KV bytes touched per second. MEDIAN of several isolated runs (the
//! decode loop is flaky under concurrent load, so a single sample is not
//! trustworthy — same policy as the Phase 3 pipeline bench).
//!
//! `#[ignore]`d (CPU reference, no GPU needed but isolated so it does not
//! disturb the fast CI suite):
//!   cargo test -p engine-server --test kv_loop_bench -- --ignored --nocapture

use engine_kvcache::PagedKvCacheConfig;
use engine_server::scheduler::BatchScheduler;
use std::time::Instant;

// Attention geometry (floats per key/value row = heads * head_dim).
const HEADS: usize = 4;
const HEAD_DIM: usize = 64; // 256 floats/row -> 512 floats/token (K+V)
const ROW_LEN: usize = HEADS * HEAD_DIM; // 256
const BLOCK_TOKENS: usize = 16;
const N_BLOCKS: usize = 4096;
const FLOATS_PER_TOKEN: usize = 2 * ROW_LEN; // key + value

// Workload: sessions × tokens per session.
const SESSIONS: usize = 64;
const TOKENS_PER_SESSION: usize = 512;

/// Median of a sorted sample (in-place).
fn median_ms(sorted_ms: &mut Vec<f64>) -> f64 {
    sorted_ms.sort_by(|a, b| a.partial_cmp(b).expect("f64 cmp"));
    sorted_ms[sorted_ms.len() / 2]
}

/// Runs the decode loop over `SESSIONS` concurrent sessions for
/// `TOKENS_PER_SESSION` steps, multiplexed through one shared scheduler
/// (continuous batching), so KV append + read are driven exactly as the server
/// would under load.
fn run_loop() -> f64 {
    let cfg = PagedKvCacheConfig {
        n_blocks: N_BLOCKS,
        block_tokens: BLOCK_TOKENS,
        heads: HEADS,
        head_dim: HEAD_DIM,
    };
    let mut scheduler = BatchScheduler::new(cfg).expect("kv pool");

    // Seed the batch: SESSIONS active sessions, all with a large budget so none
    // exits mid-iteration (saturated continuous batch).
    for i in 0..SESSIONS {
        scheduler
            .add(100_000, 1 + i as u32, 1_000_000)
            .expect("add session");
    }
    assert_eq!(scheduler.active_count(), SESSIONS);

    let start = Instant::now();
    for _ in 0..TOKENS_PER_SESSION {
        // One advance = every active session appends one token's KV rows and
        // reads one row back, all in lockstep.
        let _tokens = scheduler.advance();
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);

    let tokens = (SESSIONS * TOKENS_PER_SESSION) as f64;
    tokens / elapsed
}

#[test]
#[ignore]
fn bench_kv_throughput_through_generation_loop() {
    const RUNS: usize = 7;

    let mut samples_tok_per_s = Vec::<f64>::with_capacity(RUNS);
    let mut samples_gb_per_s = Vec::<f64>::with_capacity(RUNS);

    eprintln!(
        "\n=== KV loop bench (SESSIONS={SESSIONS}, TOKENS/SESSION={TOKENS_PER_SESSION}, row={ROW_LEN} floats, block={BLOCK_TOKENS}toks, n_blocks={N_BLOCKS}) ==="
    );

    for i in 0..RUNS {
        let tokens_per_s = run_loop();
        // Bytes touched: each token writes key+value rows and reads one back.
        let bytes_per_token = FLOATS_PER_TOKEN * std::mem::size_of::<f32>() * 2; // append+read
        let gb_per_s = tokens_per_s * bytes_per_token as f64 / 1e9;
        samples_tok_per_s.push(tokens_per_s);
        samples_gb_per_s.push(gb_per_s);
        eprintln!(
            "  run {}: {tokens_per_s:.1} tok/s  ({gb_per_s:.3} GB/s aggregate)",
            i + 1
        );
    }

    let med_tps = median_ms(&mut samples_tok_per_s);
    let med_gbps = median_ms(&mut samples_gb_per_s);
    eprintln!(
        "KV throughput through generation loop (median of {RUNS}): {med_tps:.1} tok/s  ({med_gbps:.3} GB/s aggregate)"
    );
    eprintln!("_F5_BENCH_RESULT_ kv_append_read_tok_per_s={med_tps:.1} gb_per_s={med_gbps:.3}");

    // The number is a sealed record for BENCHMARKS.md; don't gate on an exact
    // throughput target (hardware-dependent), only assert it is sane and hot.
    assert!(med_tps > 0.0, "throughput must be positive");
    assert!(
        med_gbps > 0.0,
        "aggregate KV GB/s must be positive (median {med_gbps:.3})"
    );
}

#[test]
#[ignore]
fn bench_kv_loop_determinism_smoke() {
    // Guard against the bench being a no-op: the decode loop must actually
    // advance the KV sequences (token count grows) so the measurement is real
    // work, and finished/budgeted sessions free their KV blocks.
    let cfg = PagedKvCacheConfig {
        n_blocks: N_BLOCKS,
        block_tokens: BLOCK_TOKENS,
        heads: HEADS,
        head_dim: HEAD_DIM,
    };
    let mut scheduler = BatchScheduler::new(cfg).expect("kv pool");
    // Two sessions with a small budget must finish and return their KV blocks.
    let a = scheduler.add(100_000, 2, 3).expect("session a");
    let b = scheduler.add(100_000, 5, 3).expect("session b");
    assert_eq!(scheduler.active_count(), 2);
    assert!(
        scheduler.blocks_used() > 0,
        "active sessions must hold KV blocks"
    );

    let mut produced = 0usize;
    while scheduler.active_count() > 0 {
        produced += scheduler.advance().len();
        let _ = (a, b);
    }

    assert_eq!(
        produced, 6,
        "both sessions must finish their 3-token budgets"
    );
    assert_eq!(
        scheduler.blocks_used(),
        0,
        "retired sessions must return KV blocks to the pool"
    );
    eprintln!("KV loop smoke: two sessions produced {produced} tokens and freed all KV");
}
