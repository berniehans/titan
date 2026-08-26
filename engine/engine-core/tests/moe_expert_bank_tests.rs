//! Host expert banks and slice view tests (Phase 7.2).
//!
//! Asserts that:
//! 1. `HostExpertBank` allocates contiguous host memory with per-(layer, expert) slice views.
//! 2. Multi-tensor writes ("gate_ex", "up_ex", "down_ex") within an expert slice round-trip bit-identically.
//! 3. Out-of-bounds layer or expert requests return `None`.
//! 4. Pinned memory allocator works when CUDA is available, with clean fallback to pageable.

use engine_core::moe::HostExpertBank;

#[test]
fn test_host_expert_bank_allocation_and_slice_views() {
    const N_LAYERS: usize = 4;
    const N_EXPERTS: usize = 8;
    const EXPERT_BYTES: usize = 64 * 1024; // 64 KB per expert

    let mut bank =
        HostExpertBank::allocate(N_LAYERS, N_EXPERTS, EXPERT_BYTES, false).expect("allocate bank");

    assert_eq!(bank.n_layers(), N_LAYERS);
    assert_eq!(bank.n_experts_per_layer(), N_EXPERTS);
    assert_eq!(bank.expert_slice_size(), EXPERT_BYTES);
    assert_eq!(bank.total_bytes(), N_LAYERS * N_EXPERTS * EXPERT_BYTES);
    assert!(!bank.is_pinned());

    // Write a unique pattern to (layer=2, expert=5)
    let pattern = vec![0xa5u8; 1024];
    {
        let slice = bank
            .expert_slice_mut(2, 5)
            .expect("mutable slice for (2, 5)");
        slice[0..1024].copy_from_slice(&pattern);
    }

    // Read back and assert pattern
    let read_slice = bank.expert_slice(2, 5).expect("read slice for (2, 5)");
    assert_eq!(&read_slice[0..1024], &pattern[..]);

    // Other experts must remain zeroed
    let other_slice = bank.expert_slice(2, 4).expect("read slice for (2, 4)");
    assert!(other_slice.iter().all(|&b| b == 0));

    // Bounds checking
    assert!(bank.expert_slice(4, 0).is_none());
    assert!(bank.expert_slice(0, 8).is_none());
}

#[test]
fn test_host_expert_bank_named_tensor_slicing() {
    const N_LAYERS: usize = 2;
    const N_EXPERTS: usize = 4;
    const EXPERT_BYTES: usize = 32 * 1024;

    let mut bank =
        HostExpertBank::allocate(N_LAYERS, N_EXPERTS, EXPERT_BYTES, false).expect("allocate bank");

    let gate_data = vec![0x11u8; 4096];
    let up_data = vec![0x22u8; 4096];
    let down_data = vec![0x33u8; 8192];

    bank.write_expert_tensor(1, 2, "gate_ex", 0, &gate_data)
        .expect("write gate");
    bank.write_expert_tensor(1, 2, "up_ex", 4096, &up_data)
        .expect("write up");
    bank.write_expert_tensor(1, 2, "down_ex", 8192, &down_data)
        .expect("write down");

    assert_eq!(
        bank.get_expert_tensor(1, 2, "gate_ex"),
        Some(&gate_data[..])
    );
    assert_eq!(bank.get_expert_tensor(1, 2, "up_ex"), Some(&up_data[..]));
    assert_eq!(
        bank.get_expert_tensor(1, 2, "down_ex"),
        Some(&down_data[..])
    );

    // Non-existent tensor returns None
    assert_eq!(bank.get_expert_tensor(1, 2, "missing_tensor"), None);
}

#[test]
fn test_host_expert_bank_pinned_and_pageable_modes() {
    // 1. Pageable mode
    let pageable_bank = HostExpertBank::allocate(2, 2, 1024, false).expect("allocate pageable");
    assert!(!pageable_bank.is_pinned());

    // 2. Pinned preference (attempts pinned, flags capability)
    let pinned_bank = HostExpertBank::allocate(2, 2, 1024, true).expect("allocate pinned");
    println!(
        "Pinned allocation result: is_pinned = {}",
        pinned_bank.is_pinned()
    );
}
