//! MoE VRAM budget planner and double buffer tests (Phase 7.5).
//!
//! Asserts that:
//! 1. `plan_moe_vram_budget` enforces strict VRAM limits: `total_allocated <= budget`.
//! 2. KV cache is prioritized and preserved.
//! 3. Prefill double buffer accurately alternates transfer and compute targets.
//! 4. Over-budget base configurations return clear typed errors.

use engine_core::moe::{PrefillDoubleBuffer, plan_moe_vram_budget};

#[test]
fn test_plan_moe_vram_budget_sweeps_and_invariants() {
    const N_LAYERS: usize = 24;
    const N_EXPERTS_PER_LAYER: usize = 64;
    const EXPERT_SLICE_BYTES: usize = 2 * 1024 * 1024; // 2 MB per expert

    let base_weights = 1024 * 1024 * 1024; // 1.0 GB base non-expert
    let kv_reserved = 512 * 1024 * 1024; // 512 MB KV reserve
    let activations = 64 * 1024 * 1024; // 64 MB activations

    // Test across various budget envelopes: 2.0 GB, 4.0 GB, 6.0 GB, 16.0 GB
    for budget_gb in [2.0, 4.0, 6.0, 16.0] {
        let budget_bytes = (budget_gb * 1024.0 * 1024.0 * 1024.0) as usize;
        let plan = plan_moe_vram_budget(
            budget_bytes,
            base_weights,
            kv_reserved,
            activations,
            N_LAYERS,
            N_EXPERTS_PER_LAYER,
            EXPERT_SLICE_BYTES,
        )
        .expect("plan budget");

        assert!(
            plan.total_allocated_bytes <= budget_bytes,
            "allocated {} exceeded budget {}",
            plan.total_allocated_bytes,
            budget_bytes
        );
        assert_eq!(plan.kv_reserved_bytes, kv_reserved);
        assert_eq!(plan.static_weights_bytes, base_weights);
        assert!(plan.n_slots_per_layer <= N_EXPERTS_PER_LAYER);
        assert!(plan.free_headroom_bytes() <= budget_bytes);
        assert!(plan.utilization_pct() <= 100.0);

        println!(
            "Budget {:.1} GB: Allocated {:.2} MB / {:.2} MB ({:.1}%), Slots/Layer: {}, Overlap: {}",
            budget_gb,
            plan.total_allocated_bytes as f64 / (1024.0 * 1024.0),
            budget_bytes as f64 / (1024.0 * 1024.0),
            plan.utilization_pct(),
            plan.n_slots_per_layer,
            plan.prefill_overlap_feasible
        );
    }
}

#[test]
fn test_plan_moe_vram_budget_exceeds_budget_error() {
    let budget_bytes = 1024 * 1024 * 1024; // 1 GB
    let base_weights = 800 * 1024 * 1024; // 800 MB
    let kv_reserved = 400 * 1024 * 1024; // 400 MB (Total 1.2 GB > 1.0 GB)
    let activations = 64 * 1024 * 1024;

    let res = plan_moe_vram_budget(
        budget_bytes,
        base_weights,
        kv_reserved,
        activations,
        12,
        8,
        1024 * 1024,
    );
    assert!(
        res.is_err(),
        "Must fail when base requirements exceed total budget"
    );
}

#[test]
fn test_prefill_double_buffer_ping_pong_swapping() {
    let mut double_buf = PrefillDoubleBuffer::new("BUFFER_A", "BUFFER_B");

    // Initial state: A is compute, B is transfer
    let (transfer, compute) = double_buf.ping_pong_pair();
    assert_eq!(*transfer, "BUFFER_B");
    assert_eq!(*compute, "BUFFER_A");

    // Swap 1
    double_buf.swap();
    let (transfer, compute) = double_buf.ping_pong_pair();
    assert_eq!(*transfer, "BUFFER_A");
    assert_eq!(*compute, "BUFFER_B");

    // Swap 2
    double_buf.swap();
    let (transfer, compute) = double_buf.ping_pong_pair();
    assert_eq!(*transfer, "BUFFER_B");
    assert_eq!(*compute, "BUFFER_A");
}
