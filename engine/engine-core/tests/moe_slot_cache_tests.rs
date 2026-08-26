//! GPU expert slot cache and capped-fetch LRU tests (Phase 7.3).
//!
//! Asserts that:
//! 1. `balanced_fetch` strictly minimizes the longer overlapping side (replicates upstream 0.415x3 -> 1 case).
//! 2. `ExpertSlotCache` rewrites routed IDs: resident hits -> slot, fetched -> slot, overflow -> -1.
//! 3. LRU eviction replaces the least recently used slot accurately.
//! 4. Layer telemetry counters track hits, fetches, and overflows cleanly.

use engine_core::moe::{ExpertSlotCache, balanced_fetch};

#[test]
fn test_balanced_fetch_tracks_fraction_and_regression_cases() {
    const Q: u32 = 1 << 16;

    // Upstream FreeToken regression tests:
    // With fetch fraction 0.415, 3 misses -> 1.24 expected.
    // Plain ceil would fetch 2, making PCIe 1.6x slower than balance; balanced fetch keeps it at 1.
    let frac_0415 = (0.415 * Q as f64).round() as u32;
    assert_eq!(
        balanced_fetch(3, frac_0415),
        1,
        "balanced_fetch(3, 0.415) must return 1"
    );
    assert_eq!(
        balanced_fetch(4, frac_0415),
        2,
        "balanced_fetch(4, 0.415) must return 2"
    );

    // General property assertions across test matrix
    for frac in [0.1, 0.25, 0.415, 0.5, 0.75, 1.0] {
        let q = (frac * Q as f64).round() as u32;
        for m in 0..=64 {
            let f = balanced_fetch(m, q);
            assert!(f <= m, "fetched {f} cannot exceed total misses {m}");
            let expected_float = frac * m as f64;
            assert!(
                (f as f64 - expected_float).abs() <= 1.01,
                "fetched {f} deviated more than 1 from float expected {expected_float}"
            );
        }
    }
}

#[test]
fn test_lru_slot_cache_routing_rewrite_and_eviction() {
    const N_LAYERS: usize = 2;
    const N_EXPERTS: usize = 16;
    const N_SLOTS: usize = 4; // 4 GPU slots per layer

    let mut cache = ExpertSlotCache::new(N_LAYERS, N_EXPERTS, N_SLOTS);

    // Step 1: Initial cold requests on layer 0: experts [0, 1, 2, 3] with 100% fetch
    let r1 = cache.step_layer(0, &[0, 1, 2, 3], 1.0, 100);
    assert_eq!(r1.slot_ids, vec![0, 1, 2, 3]);
    assert_eq!(r1.fetched_experts, vec![0, 1, 2, 3]);
    assert_eq!(r1.fetch_target_slots, vec![0, 1, 2, 3]);
    assert!(r1.cpu_experts.is_empty());

    // Verify stats for Step 1
    let stats1 = cache.stats(0).unwrap();
    assert_eq!(stats1.active_requests, 4);
    assert_eq!(stats1.resident_hits, 0);
    assert_eq!(stats1.pcie_fetched, 4);
    assert_eq!(stats1.cpu_overflow, 0);

    // Step 2: Requests [0, 1, 4, 5] at step 200 with fetch_fraction = 0.5
    // - Experts 0 and 1 are resident in slots 0 and 1 (hits)
    // - Experts 4 and 5 are misses (2 misses)
    // - balanced_fetch(2, 0.5) = 1 -> exactly 1 fetch allowed, 1 overflow to CPU
    // - Slot 2 (expert 2, last used step 100) is evicted for expert 4
    let r2 = cache.step_layer(0, &[0, 1, 4, 5], 0.5, 200);
    assert_eq!(r2.slot_ids, vec![0, 1, 2, -1]);
    assert_eq!(r2.fetched_experts, vec![4]);
    assert_eq!(r2.fetch_target_slots, vec![2]);
    assert_eq!(r2.cpu_experts, vec![5]);

    // Step 3: Verify stats for Step 2
    let stats2 = cache.stats(0).unwrap();
    assert_eq!(stats2.active_requests, 8);
    assert_eq!(stats2.resident_hits, 2);
    assert_eq!(stats2.pcie_fetched, 5);
    assert_eq!(stats2.cpu_overflow, 1);
    assert!((stats2.pre_cap_miss_rate() - (6.0 / 8.0)).abs() < 1e-6);

    // Step 4: Reset clears bindings
    cache.reset();
    let stats_reset = cache.stats(0).unwrap();
    assert_eq!(stats_reset.active_requests, 0);
}
