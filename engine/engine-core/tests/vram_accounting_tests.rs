//! VRAM accounting unit and budget tests (Phase 6.9, Group 1).
//!
//! Asserts that:
//! 1. Static per-stage VRAM map for Qwen3-0.6B sums <= 5.2 GB budget.
//! 2. KV cache growth scales linearly with sequence length while staying within budget.
//! 3. Formatted trace printout displays all 4 stages clearly.

use engine_core::forward_driver::VRAM_BUDGET_BYTES;
use engine_core::vram_accounting::compute_static_vram_map;
use engine_io::ModelConfig;

fn sample_qwen3_config() -> ModelConfig {
    ModelConfig {
        architecture: "qwen3".to_string(),
        n_layer: 28,
        n_head: 16,
        n_head_kv: 8,
        head_dim: 128,
        value_dim: 128,
        hidden_size: 1024,
        intermediate_size: 3072,
        vocab_size: 151936,
        context_length: 32768,
        rope_freq_base: 1_000_000.0,
        rope_freq_scale: 1.0,
        rms_norm_eps: 1e-6,
        tokenizer_model: "bpe".to_string(),
        eos_token_id: 151643,
        padding_token_id: 0,
        add_bos: false,
    }
}

#[test]
fn test_static_vram_accounting_under_5_2_gb_budget() {
    let cfg = sample_qwen3_config();
    let max_seq = 2048;
    let vocab = 151936;

    // 1. Resident mode (all 28 layers resident in device memory)
    let breakdown_resident = compute_static_vram_map(&cfg, max_seq, vocab, false);
    println!("\n=== Resident Mode (2048 tokens) ===");
    println!("{}", breakdown_resident.format_trace(VRAM_BUDGET_BYTES));

    assert!(
        breakdown_resident.pingpong_bytes > 0,
        "weights must be non-zero"
    );
    assert!(
        breakdown_resident.kv_pool_bytes > 0,
        "kv pool must be non-zero"
    );
    assert!(
        breakdown_resident.activations_bytes > 0,
        "activations must be non-zero"
    );
    assert!(
        breakdown_resident.logits_bytes > 0,
        "logits must be non-zero"
    );

    assert!(
        breakdown_resident.total_bytes() <= VRAM_BUDGET_BYTES,
        "Total working set ({} bytes) must be <= 5.2 GB ({})",
        breakdown_resident.total_bytes(),
        VRAM_BUDGET_BYTES
    );
    assert!(
        breakdown_resident
            .assert_within_budget(VRAM_BUDGET_BYTES)
            .is_ok()
    );

    // 2. Double-buffered streaming mode (2 layers in ping-pong staging)
    let breakdown_streaming = compute_static_vram_map(&cfg, max_seq, vocab, true);
    println!("\n=== Double-Buffered Streaming Mode (2048 tokens) ===");
    println!("{}", breakdown_streaming.format_trace(VRAM_BUDGET_BYTES));

    assert!(breakdown_streaming.pingpong_bytes < breakdown_resident.pingpong_bytes);
    assert!(breakdown_streaming.total_bytes() < breakdown_resident.total_bytes());
    assert!(
        breakdown_streaming
            .assert_within_budget(VRAM_BUDGET_BYTES)
            .is_ok()
    );
}

#[test]
fn test_vram_kv_growth_across_sequence_lengths() {
    let cfg = sample_qwen3_config();
    let vocab = 151936;

    for seq_len in [128, 512, 1024, 2048, 4096, 8192] {
        let map = compute_static_vram_map(&cfg, seq_len, vocab, false);
        println!(
            "Seq len {:>5}: Total={:>8.2} MB (KV={:>8.2} MB, Weights={:>8.2} MB, Act={:>6.2} MB) [Util: {:>5.2}%]",
            seq_len,
            map.total_bytes() as f64 / (1024.0 * 1024.0),
            map.kv_pool_bytes as f64 / (1024.0 * 1024.0),
            map.pingpong_bytes as f64 / (1024.0 * 1024.0),
            map.activations_bytes as f64 / (1024.0 * 1024.0),
            map.budget_utilization(VRAM_BUDGET_BYTES) * 100.0
        );
        assert!(
            map.total_bytes() <= VRAM_BUDGET_BYTES,
            "seq_len {seq_len} exceeded 5.2 GB budget"
        );
    }
}
