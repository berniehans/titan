//! Speculative verification and multi-token rejection sampling (Phase 12, Sub-change 12.1).
//!
//! Provides mathematically distribution-preserving speculative token verification
//! and deterministic greedy candidate matching.

use crate::sampler::{Sampler, SamplerParams};

/// Result of speculative candidate verification.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeVerificationResult {
    /// Newly emitted tokens starting with `candidates[0]` (or target prediction) through bonus token.
    /// Note: Does not include the already-committed `base_token`.
    pub emitted_tokens: Vec<u32>,
    /// Number of proposed candidate tokens that were verified and accepted (0..=K).
    pub n_accepted: usize,
    /// The final bonus / correction token sampled from the target distribution.
    pub bonus_token: u32,
    /// Total new tokens emitted in this single speculative step (`emitted_tokens.len()`).
    pub total_emitted: usize,
}

/// Speculative verifier evaluating candidate tokens against target logits.
pub struct SpeculativeVerifier;

impl SpeculativeVerifier {
    /// Performs deterministic greedy verification against target logits.
    ///
    /// `candidates`: Slice of $K$ proposed candidate tokens for positions $t+1, \dots, t+K$.
    /// `target_logits`: Array of $K+1$ logit slices:
    ///   - `target_logits[0]`: Prediction for position $t+1$ (evaluated against `candidates[0]`).
    ///   - `target_logits[i]`: Prediction for position $t+i+1$ (evaluated against `candidates[i]`).
    ///   - `target_logits[K]`: Bonus token prediction for position $t+K+1$.
    pub fn verify_greedy(
        candidates: &[u32],
        target_logits: &[&[f32]],
    ) -> SpeculativeVerificationResult {
        assert_eq!(
            target_logits.len(),
            candidates.len() + 1,
            "Target logits must have length candidates.len() + 1"
        );

        let k = candidates.len();
        let mut emitted = Vec::with_capacity(k + 1);
        let mut n_accepted = 0;

        for i in 0..k {
            let cand = candidates[i];
            let logits = target_logits[i];
            let target_tok = Sampler::argmax(logits);

            if target_tok == cand {
                emitted.push(cand);
                n_accepted += 1;
            } else {
                // First mismatch: emit target_tok as correction and terminate verification
                emitted.push(target_tok);
                let bonus_token = target_tok;
                let total_emitted = emitted.len();
                return SpeculativeVerificationResult {
                    emitted_tokens: emitted,
                    n_accepted,
                    bonus_token,
                    total_emitted,
                };
            }
        }

        // All K candidates accepted: sample bonus token from target_logits[K]
        let last_logits = target_logits[k];
        let bonus_token = Sampler::argmax(last_logits);
        emitted.push(bonus_token);
        let total_emitted = emitted.len();

        SpeculativeVerificationResult {
            emitted_tokens: emitted,
            n_accepted,
            bonus_token,
            total_emitted,
        }
    }

    /// Performs distribution-preserving modified rejection sampling for stochastic generation.
    ///
    /// `candidates`: Slice of $K$ proposed candidate tokens for positions $t+1, \dots, t+K$.
    /// `draft_probs`: Slice of $K$ draft probability vectors (or empty if uniform).
    /// `target_logits`: Slice of $K+1$ target logit slices.
    /// `sampler`: The sampler instance for RNG.
    /// `params`: Sampling parameters (temperature, top_p, top_k).
    pub fn verify_stochastic(
        candidates: &[u32],
        draft_probs: &[&[f32]],
        target_logits: &[&[f32]],
        sampler: &mut Sampler,
        params: &SamplerParams,
        context: &[u32],
    ) -> SpeculativeVerificationResult {
        if params.is_greedy() {
            return Self::verify_greedy(candidates, target_logits);
        }

        assert_eq!(
            target_logits.len(),
            candidates.len() + 1,
            "Target logits must have length candidates.len() + 1"
        );

        let k = candidates.len();
        let mut emitted = Vec::with_capacity(k + 1);
        let mut n_accepted = 0;
        let mut current_context = context.to_vec();

        for i in 0..k {
            let cand = candidates[i];
            let logits = target_logits[i];
            let vocab_size = logits.len();

            let target_prob = sampler.compute_distribution(logits, &current_context, params);

            let p_cand = if (cand as usize) < target_prob.len() {
                target_prob[cand as usize]
            } else {
                0.0
            };

            let q_cand = if !draft_probs.is_empty() && i < draft_probs.len() {
                let dp = draft_probs[i];
                if (cand as usize) < dp.len() {
                    dp[cand as usize]
                } else {
                    1.0 / vocab_size as f32
                }
            } else {
                1.0 / vocab_size as f32
            };

            let r = if q_cand > 0.0 {
                (p_cand / q_cand).min(1.0)
            } else {
                1.0
            };
            let u = sampler.sample_uniform();

            if u <= r {
                emitted.push(cand);
                current_context.push(cand);
                n_accepted += 1;
            } else {
                // Rejection: Sample from normalized positive residual max(0, p(x) - q(x))
                let mut residual = vec![0.0f32; vocab_size];
                let mut sum_res = 0.0f32;

                let dp = if !draft_probs.is_empty() && i < draft_probs.len() {
                    Some(draft_probs[i])
                } else {
                    None
                };

                for v in 0..vocab_size {
                    let p_v = if v < target_prob.len() {
                        target_prob[v]
                    } else {
                        0.0
                    };
                    let q_v = if let Some(d) = dp {
                        if v < d.len() {
                            d[v]
                        } else {
                            1.0 / vocab_size as f32
                        }
                    } else {
                        1.0 / vocab_size as f32
                    };
                    let diff = (p_v - q_v).max(0.0);
                    residual[v] = diff;
                    sum_res += diff;
                }

                let bonus_token = if sum_res > 1e-8 {
                    for value in residual.iter_mut().take(vocab_size) {
                        *value /= sum_res;
                    }
                    sampler.sample_from_probs(&residual)
                } else {
                    sampler.sample(logits, &current_context, params)
                };

                emitted.push(bonus_token);
                let total_emitted = emitted.len();
                return SpeculativeVerificationResult {
                    emitted_tokens: emitted,
                    n_accepted,
                    bonus_token,
                    total_emitted,
                };
            }
        }

        // All K accepted: sample bonus token from target logits of last position
        let last_logits = target_logits[k];
        let bonus_token = sampler.sample(last_logits, &current_context, params);
        emitted.push(bonus_token);
        let total_emitted = emitted.len();

        SpeculativeVerificationResult {
            emitted_tokens: emitted,
            n_accepted,
            bonus_token,
            total_emitted,
        }
    }
}
