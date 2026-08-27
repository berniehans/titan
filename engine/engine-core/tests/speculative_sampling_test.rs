//! Speculative Verification & Rejection Sampling Test (Phase 12, Sub-change 12.1).
//!
//! Validates:
//! 1. `SpeculativeVerifier::verify_greedy` exact prefix acceptance and bonus correction.
//! 2. `SpeculativeVerifier::verify_stochastic` distribution-preserving rejection sampling.

use engine_core::sampler::{Sampler, SamplerParams};
use engine_core::speculative::SpeculativeVerifier;

#[test]
fn test_speculative_greedy_full_acceptance() {
    let candidates = vec![20u32, 30u32];

    // l0 predicts token for position 1 (which matches 20).
    // l1 predicts token for position 2 (which matches 30).
    // l2 predicts bonus token for position 3 (bonus token = 40).
    let mut l0 = vec![0.0f32; 100];
    l0[20] = 10.0;
    let mut l1 = vec![0.0f32; 100];
    l1[30] = 10.0;
    let mut l2 = vec![0.0f32; 100];
    l2[40] = 10.0;

    let target_logits = vec![l0.as_slice(), l1.as_slice(), l2.as_slice()];

    let res = SpeculativeVerifier::verify_greedy(&candidates, &target_logits);

    assert_eq!(res.n_accepted, 2);
    assert_eq!(res.emitted_tokens, vec![20, 30, 40]);
    assert_eq!(res.bonus_token, 40);
    assert_eq!(res.total_emitted, 3);
}

#[test]
fn test_speculative_greedy_partial_acceptance_and_correction() {
    let candidates = vec![20u32, 30u32];

    // l0 predicts 20 (matches candidates[0]).
    // l1 predicts 99 (mismatch with candidates[1] = 30!).
    // l2 is unused due to early termination.
    let mut l0 = vec![0.0f32; 100];
    l0[20] = 10.0;
    let mut l1 = vec![0.0f32; 100];
    l1[99] = 10.0;
    let mut l2 = vec![0.0f32; 100];
    l2[30] = 10.0;

    let target_logits = vec![l0.as_slice(), l1.as_slice(), l2.as_slice()];

    let res = SpeculativeVerifier::verify_greedy(&candidates, &target_logits);

    assert_eq!(res.n_accepted, 1);
    assert_eq!(res.emitted_tokens, vec![20, 99]);
    assert_eq!(res.bonus_token, 99);
    assert_eq!(res.total_emitted, 2);
}

#[test]
fn test_speculative_greedy_immediate_mismatch() {
    let candidates = vec![20u32];

    // l0 predicts 77 (mismatch with candidates[0] = 20!).
    let mut l0 = vec![0.0f32; 100];
    l0[77] = 10.0;
    let mut l1 = vec![0.0f32; 100];
    l1[20] = 10.0;

    let target_logits = vec![l0.as_slice(), l1.as_slice()];

    let res = SpeculativeVerifier::verify_greedy(&candidates, &target_logits);

    assert_eq!(res.n_accepted, 0);
    assert_eq!(res.emitted_tokens, vec![77]);
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

    let candidates = vec![6u32];

    // l0 strongly favors 6
    let mut l0 = vec![0.0f32; 20];
    l0[6] = 20.0;
    let mut l1 = vec![0.0f32; 20];
    l1[7] = 20.0;

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
    assert_eq!(res.emitted_tokens[0], 6);
}
