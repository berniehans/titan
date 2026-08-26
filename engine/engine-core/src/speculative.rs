//! Speculative verification and multi-token rejection sampling (Phase 12, Sub-change 12.1).
//!
//! Provides mathematically distribution-preserving speculative token verification
//! and deterministic greedy candidate matching.

use crate::sampler::{Sampler, SamplerParams};

/// Result of speculative candidate verification.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeVerificationResult {
    /// Sequence of accepted candidate tokens plus one bonus/correction token.
    pub accepted_tokens: Vec<u32>,
    /// Number of candidate tokens that were verified and accepted (0..=K).
    pub n_accepted: usize,
    /// The final bonus token sampled from the target distribution.
    pub bonus_token: u32,
    /// Total tokens emitted in this single speculative step (`n_accepted + 1`).
    pub total_emitted: usize,
}

/// Speculative verifier evaluating candidate tokens against target logits.
pub struct SpeculativeVerifier;

impl SpeculativeVerifier {
    /// Performs deterministic greedy verification against target logits.
    ///
    /// `candidates`: Slice of $K$ proposed candidate tokens.
    /// `target_logits`: Array of $K$ logit slices, each of length `vocab_size`.
    pub fn verify_greedy(
        candidates: &[u32],
        target_logits: &[&[f32]],
    ) -> SpeculativeVerificationResult {
        assert_eq!(
            candidates.len(),
            target_logits.len(),
            "Candidate length must match target logits length"
        );

        let k = candidates.len();
        if k == 0 {
            return SpeculativeVerificationResult {
                accepted_tokens: Vec::new(),
                n_accepted: 0,
                bonus_token: 0,
                total_emitted: 0,
            };
        }

        let mut accepted = Vec::with_capacity(k + 1);
        let mut n_accepted = 0;

        for (_i, (&cand, &logits)) in candidates.iter().zip(target_logits.iter()).enumerate() {
            let target_tok = Sampler::argmax(logits);
            if target_tok == cand {
                accepted.push(cand);
                n_accepted += 1;
            } else {
                // First mismatch: accept target_tok as correction and terminate verification
                accepted.push(target_tok);
                return SpeculativeVerificationResult {
                    accepted_tokens: accepted,
                    n_accepted,
                    bonus_token: target_tok,
                    total_emitted: n_accepted + 1,
                };
            }
        }

        // If all K candidates were accepted, the last token in accepted is candidates[K-1].
        // The bonus token is candidates[K-1] (or sampled from final position logits).
        let bonus_token = candidates[k - 1];
        SpeculativeVerificationResult {
            accepted_tokens: accepted,
            n_accepted,
            bonus_token,
            total_emitted: n_accepted,
        }
    }

    /// Performs distribution-preserving modified rejection sampling for stochastic generation.
    ///
    /// `candidates`: Slice of $K$ proposed candidate tokens.
    /// `draft_probs`: Slice of $K$ draft probability vectors (or empty if uniform).
    /// `target_logits`: Slice of $K$ target logit slices.
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

        let k = candidates.len();
        if k == 0 {
            return SpeculativeVerificationResult {
                accepted_tokens: Vec::new(),
                n_accepted: 0,
                bonus_token: 0,
                total_emitted: 0,
            };
        }

        let mut accepted = Vec::with_capacity(k + 1);
        let mut n_accepted = 0;
        let mut current_context = context.to_vec();

        for i in 0..k {
            let cand = candidates[i];
            let logits = target_logits[i];
            let vocab_size = logits.len();

            // Compute target distribution
            let target_prob = sampler.compute_distribution(logits, &current_context, params);

            let p_cand = if (cand as usize) < target_prob.len() {
                target_prob[cand as usize]
            } else {
                0.0
            };

            let q_cand = if !draft_probs.is_empty() && i < draft_probs.len() {
                let dp = draft_probs[i];
                if (cand as usize) < dp.len() { dp[cand as usize] } else { 1.0 / vocab_size as f32 }
            } else {
                1.0 / vocab_size as f32
            };

            // Acceptance probability r = min(1, p(x) / q(x))
            let r = if q_cand > 0.0 { (p_cand / q_cand).min(1.0) } else { 1.0 };
            let u = sampler.sample_uniform();

            if u <= r {
                accepted.push(cand);
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
                    let p_v = if v < target_prob.len() { target_prob[v] } else { 0.0 };
                    let q_v = if let Some(d) = dp {
                        if v < d.len() { d[v] } else { 1.0 / vocab_size as f32 }
                    } else {
                        1.0 / vocab_size as f32
                    };
                    let diff = (p_v - q_v).max(0.0);
                    residual[v] = diff;
                    sum_res += diff;
                }

                let bonus_token = if sum_res > 1e-8 {
                    for v in 0..vocab_size {
                        residual[v] /= sum_res;
                    }
                    sampler.sample_from_probs(&residual)
                } else {
                    sampler.sample(logits, &current_context, params)
                };

                accepted.push(bonus_token);
                return SpeculativeVerificationResult {
                    accepted_tokens: accepted,
                    n_accepted,
                    bonus_token,
                    total_emitted: n_accepted + 1,
                };
            }
        }

        // All K accepted: sample bonus token from target logits of last position
        let last_logits = target_logits[k - 1];
        let bonus_token = sampler.sample(last_logits, &current_context, params);
        accepted.push(bonus_token);

        SpeculativeVerificationResult {
            accepted_tokens: accepted,
            n_accepted,
            bonus_token,
            total_emitted: n_accepted + 1,
        }
    }
}
