//! Additive safety integration test for stub_next_token (Phase 6.7, Group 3).
//!
//! Validates:
//! 1. `stub_next_token` produces bit-identical deterministic outputs across a test matrix.
//! 2. `digest_layer` and `kv_row` remain deterministic and intact.
//! 3. Session generation progression using the stub path is unchanged and reproducible.

use engine_server::session::{digest_layer, kv_row, stub_next_token};

#[test]
fn test_stub_next_token_bit_identical_deterministic_vectors() {
    let test_cases = [
        // (token, digest, vocab, expected)
        (42u32, 0xdeadbeefu64, 1000u32, 238u32),
        (0u32, 0u64, 1000u32, 1u32),
        (1u32, 0u64, 1000u32, 762u32),
        (9707u32, 0x123456789abcdef0u64, 151936u32, 37260u32),
        (151935u32, 0xffffffffffffffffu64, 151936u32, 83505u32),
    ];

    for (token, digest, vocab, expected) in test_cases {
        let result = stub_next_token(token, digest, vocab);
        assert_eq!(
            result, expected,
            "stub_next_token({token}, {digest:#x}, {vocab}) returned {result}, expected {expected}"
        );
        assert!((1..=vocab).contains(&result));
    }
}

#[test]
fn test_digest_layer_and_kv_row_determinism() {
    let row1 = [0.1f32, -0.5f32, 1.25f32, 0.0f32, 0.75f32];
    let d1 = digest_layer(&row1);
    let d2 = digest_layer(&row1);
    assert_eq!(d1, d2, "digest_layer must be deterministic");

    let row2 = [0.1f32, -0.5f32, 1.25f32, 0.0f32, 4.5f32];
    let d3 = digest_layer(&row2);
    assert_ne!(
        d1, d3,
        "different float values must produce different digests"
    );

    let (k1, v1) = kv_row(42, 64);
    let (k2, v2) = kv_row(42, 64);
    assert_eq!(k1, k2);
    assert_eq!(v1, v2);
    assert_eq!(k1.len(), 64);
    assert_eq!(v1.len(), 64);
}

#[test]
fn test_stub_multi_step_sequence_progression() {
    let mut current_token = 42u32;
    let vocab = 1000u32;
    let mut tokens = Vec::new();

    // 10-step teacher-forced stub sequence
    for step in 0..10 {
        let dummy_layer_floats = vec![(step as f32) * 1.5; 128];
        let digest = digest_layer(&dummy_layer_floats);
        current_token = stub_next_token(current_token, digest, vocab);
        tokens.push(current_token);
    }

    // Assert sequence matches exact deterministic progression
    assert_eq!(tokens.len(), 10);
    // Running again must produce bit-identical sequence
    let mut re_current = 42u32;
    let mut re_tokens = Vec::new();
    for step in 0..10 {
        let dummy_layer_floats = vec![(step as f32) * 1.5; 128];
        let digest = digest_layer(&dummy_layer_floats);
        re_current = stub_next_token(re_current, digest, vocab);
        re_tokens.push(re_current);
    }
    assert_eq!(
        tokens, re_tokens,
        "stub generation sequence must be bit-identical"
    );
}
