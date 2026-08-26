//! Speculative Verification & Rejection Sampling Test (Phase 12, Sub-change 12.1).
//!
//! Validates:
//! 1. `SpeculativeVerifier::verify_greedy` exact prefix acceptance and bonus correction.
//! 2. `SpeculativeVerifier::verify_stochastic` distribution-preserving rejection sampling.

use engine_core::sampler::{Sampler, SamplerParams};
use engine_core::speculative::SpeculativeVerifier;

#[test]
fn test_speculative_greedy_full_acceptance() {
    let candidates = vec![10u32, 20u32, 30u32];

    // Target logits predicting exactly [10, 20, 30]
    let mut l0 = vec![0.0f32; 100];
    l0[10] = 10.0;
    let mut l1 = vec![0.0f32; 100];
    l1[20] = 10.0;
    let mut l2 = vec![0.0f32; 100];
    l2[30] = 10.0;

    let target_logits = vec![l0.as_slice(), l1.as_slice(), l2.as_slice()];

    let res = SpeculativeVerifier::verify_greedy(&candidates, &target_logits);

    assert_eq!(res.n_accepted, 3);
    assert_eq!(res.accepted_tokens, vec![10, 20, 30]);
    assert_eq!(res.total_emitted, 3);
}

#[test]
fn test_speculative_greedy_partial_acceptance_and_correction() {
    let candidates = vec![10u32, 20u32, 30u32];

    // Target logits predicting [10, 99 (mismatch!), ...]
    let mut l0 = vec![0.0f32; 100];
    l0[10] = 10.0;
    let mut l1 = vec![0.0f32; 100];
    l1[99] = 10.0; // Correct token is 99, not 20!
    let mut l2 = vec![0.0f32; 100];
    l2[30] = 10.0;

    let target_logits = vec![l0.as_slice(), l1.as_slice(), l2.as_slice()];

    let res = SpeculativeVerifier::verify_greedy(&candidates, &target_logits);

    assert_eq!(res.n_accepted, 1);
    assert_eq!(res.accepted_tokens, vec![10, 99]);
    assert_eq!(res.bonus_token, 99);
    assert_eq!(res.total_emitted, 2);
}

#[test]
fn test_speculative_greedy_immediate_rejection() {
    let candidates = vec![10u32, 20u32];

    // Target logits predicting [77 (mismatch at index 0!), ...]
    let mut l0 = vec![0.0f32; 100];
    l0[77] = 10.0;
    let mut l1 = vec![0.0f32; 100];
    l1[20] = 10.0;

    let target_logits = vec![l0.as_slice(), l1.as_slice()];

    let res = SpeculativeVerifier::verify_greedy(&candidates, &target_logits);

    assert_eq!(res.n_accepted, 0);
    assert_eq!(res.accepted_tokens, vec![77]);
    assert_eq!(res.bonus_token, 77);
    assert_eq!(res.total_emitted, 1);
}

#[test]
fn test_speculative_stochastic_sampling_acceptance() {
    let mut sampler = Sampler::new(12345);
    let params = SamplerParams {
        temperature: 0.7,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        seed: Some(12345),
    };

    let candidates = vec![5u32, 6u32];

    // Target logits strongly favoring 5 and 6
    let mut l0 = vec![0.0f32; 20];
    l0[5] = 20.0;
    let mut l1 = vec![0.0f32; 20];
    l1[6] = 20.0;

    let target_logits = vec![l0.as_slice(), l1.as_slice()];

    let res = SpeculativeVerifier::verify_stochastic(
        &candidates,
        &[],
        &target_logits,
        &mut sampler,
        &params,
        &[],
    );

    assert!(res.n_accepted >= 1);
    assert_eq!(res.accepted_tokens[0], 5);
}
