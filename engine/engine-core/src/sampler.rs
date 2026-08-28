//! Advanced production token sampler for Titan inference engine.
//!
//! Supports:
//! 1. Greedy argmax selection (when `temperature <= 1e-4`).
//! 2. Repetition penalty scaling across context tokens.
//! 3. Temperature scaling and softmax probability normalization.
//! 4. Top-K filtering (unbounded if `top_k == 0`).
//! 5. Top-P (nucleus) cumulative mass filtering.
//! 6. Deterministic seedable RNG sampling.

/// Configuration parameters controlling token sampling behavior.
#[derive(Debug, Clone)]
pub struct SamplerParams {
    /// Sampling temperature in [0.0, 2.0]. Values <= 1e-4 perform greedy argmax.
    pub temperature: f32,
    /// Nucleus sampling cumulative probability threshold in (0.0, 1.0].
    pub top_p: f32,
    /// Top-K largest logit candidate filtering (0 disables Top-K).
    pub top_k: usize,
    /// Repetition penalty factor >= 1.0 (1.0 disables penalty).
    pub repetition_penalty: f32,
    /// Optional RNG seed for deterministic reproduction.
    pub seed: Option<u64>,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            seed: None,
        }
    }
}

impl SamplerParams {
    /// Constructs greedy argmax parameters (temperature = 0.0).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: None,
        }
    }

    /// Returns true if temperature <= 1e-4.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 1e-4
    }
}

/// Simple fast 64-bit Xorshift PRNG for reproducible sampling without heavy dependencies.
#[derive(Debug, Clone)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a uniform float in [0.0, 1.0).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Production token sampler.
pub struct Sampler {
    rng: XorShift64,
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new(42)
    }
}

impl Sampler {
    /// Creates a new sampler with initial random seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: XorShift64::new(seed),
        }
    }

    /// Resets the internal PRNG seed.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = XorShift64::new(seed);
    }

    /// Samples the next token ID from raw `logits` given `context_tokens` history and `params`.
    pub fn sample(
        &mut self,
        logits: &[f32],
        context_tokens: &[u32],
        params: &SamplerParams,
    ) -> u32 {
        if logits.is_empty() {
            return 0;
        }



        // 1. Clone logits into candidate working buffer
        let mut filtered: Vec<(usize, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, l)| l.is_finite())
            .collect();

        if filtered.is_empty() {
            return 0;
        }

        // 2. Apply repetition penalty in O(context) with direct indexing:
        if params.repetition_penalty > 1.0 && !context_tokens.is_empty() {
            let pen = params.repetition_penalty;
            let mut seen = std::collections::HashSet::new();
            for &tok in context_tokens {
                let idx = tok as usize;
                if idx < filtered.len() && seen.insert(idx) {
                    if filtered[idx].1 > 0.0 {
                        filtered[idx].1 /= pen;
                    } else {
                        filtered[idx].1 *= pen;
                    }
                }
            }
        }

        // 3. Greedy argmax fast path (when temperature <= 1e-4)
        if params.temperature <= 1e-4 {
            let mut best_idx = 0;
            let mut best_logit = f32::NEG_INFINITY;
            for &(idx, logit) in &filtered {
                if logit > best_logit {
                    best_logit = logit;
                    best_idx = idx;
                }
            }
            return best_idx as u32;
        }

        // 4. Apply temperature scaling:
        let inv_temp = 1.0 / params.temperature;
        for cand in &mut filtered {
            cand.1 *= inv_temp;
        }

        // 5. O(N) top-K partition instead of full 151k sort:
        let k = if params.top_k > 0 && params.top_k < filtered.len() {
            params.top_k
        } else {
            filtered.len().min(1024)
        };
        if k < filtered.len() {
            filtered.select_nth_unstable_by(k, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            filtered.truncate(k);
        }

        // 6. Sort only top-k candidates descending by logit:
        filtered.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 7. Compute Softmax probabilities:
        let max_logit = filtered[0].1;
        let mut sum_exp = 0.0f32;
        let mut probs: Vec<(usize, f32)> = Vec::with_capacity(filtered.len());
        for (idx, logit) in filtered {
            let exp_val = (logit - max_logit).exp();
            probs.push((idx, exp_val));
            sum_exp += exp_val;
        }

        if sum_exp <= 0.0 {
            return probs[0].0 as u32;
        }

        let inv_sum = 1.0 / sum_exp;
        for p in &mut probs {
            p.1 *= inv_sum;
        }

        // 8. Top-P (Nucleus) filtering:
        if params.top_p < 1.0 {
            let mut cumsum = 0.0f32;
            let mut cutoff_idx = probs.len();
            for (i, p) in probs.iter().enumerate() {
                cumsum += p.1;
                if cumsum >= params.top_p {
                    cutoff_idx = (i + 1).min(probs.len());
                    break;
                }
            }
            probs.truncate(cutoff_idx.max(1));

            // Renormalize remaining probabilities
            let mut new_sum = 0.0f32;
            for p in &probs {
                new_sum += p.1;
            }
            if new_sum > 0.0 {
                let inv = 1.0 / new_sum;
                for p in &mut probs {
                    p.1 *= inv;
                }
            }
        }

        // 9. Sample from categorical distribution
        let r = self.rng.next_f32();
        let mut acc = 0.0f32;
        for &(idx, prob) in &probs {
            acc += prob;
            if r <= acc {
                return idx as u32;
            }
        }

        // Fallback to top token
        probs.first().map(|p| p.0 as u32).unwrap_or(0)
    }

    /// Samples the next token with grammar constraint predicate `is_allowed(token_id)`.
    pub fn sample_constrained<F>(
        &mut self,
        logits: &[f32],
        context_tokens: &[u32],
        params: &SamplerParams,
        mut is_allowed: F,
    ) -> u32
    where
        F: FnMut(u32) -> bool,
    {
        if logits.is_empty() {
            return 0;
        }

        // 1. Check candidates in descending order of logit likelihood
        let mut top_candidates: Vec<(usize, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, l)| l.is_finite())
            .collect();

        top_candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut filtered: Vec<(usize, f32)> = Vec::with_capacity(64);
        for (idx, logit) in top_candidates {
            if is_allowed(idx as u32) {
                filtered.push((idx, logit));
                if filtered.len() >= 64 {
                    break;
                }
            }
        }

        if filtered.is_empty() {
            return self.sample(logits, context_tokens, params);
        }

        // 2. Repetition penalty
        if params.repetition_penalty > 1.0 && !context_tokens.is_empty() {
            let pen = params.repetition_penalty;
            let mut seen = std::collections::HashSet::new();
            for &tok in context_tokens {
                let idx = tok as usize;
                if let Some(pos) = filtered.iter().position(|&(i, _)| i == idx) {
                    if seen.insert(idx) {
                        if filtered[pos].1 > 0.0 {
                            filtered[pos].1 /= pen;
                        } else {
                            filtered[pos].1 *= pen;
                        }
                    }
                }
            }
        }

        // 3. Greedy argmax fast path
        if params.temperature <= 1e-4 {
            let mut best_idx = filtered[0].0;
            let mut best_logit = f32::NEG_INFINITY;
            for &(idx, logit) in &filtered {
                if logit > best_logit {
                    best_logit = logit;
                    best_idx = idx;
                }
            }
            return best_idx as u32;
        }

        // 4. Softmax + nucleus sampling
        let inv_temp = 1.0 / params.temperature;
        for cand in &mut filtered {
            cand.1 *= inv_temp;
        }

        filtered.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let max_logit = filtered[0].1;
        let mut sum_exp = 0.0f32;
        let mut probs: Vec<(usize, f32)> = Vec::with_capacity(filtered.len());
        for (idx, logit) in filtered {
            let exp_val = (logit - max_logit).exp();
            probs.push((idx, exp_val));
            sum_exp += exp_val;
        }

        if sum_exp <= 0.0 {
            return probs[0].0 as u32;
        }

        let inv_sum = 1.0 / sum_exp;
        for p in &mut probs {
            p.1 *= inv_sum;
        }

        let r = self.rng.next_f32() * params.top_p;
        let mut accum = 0.0f32;
        for (idx, prob) in probs {
            accum += prob;
            if accum >= r {
                return idx as u32;
            }
        }
        0
    }

    /// Computes greedy argmax of raw logits.
    pub fn argmax(logits: &[f32]) -> u32 {
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        best_idx as u32
    }

    /// Generates a uniform random float in [0.0, 1.0).
    pub fn sample_uniform(&mut self) -> f32 {
        self.rng.next_f32()
    }

    /// Samples a token from an explicit probability distribution vector.
    pub fn sample_from_probs(&mut self, probs: &[f32]) -> u32 {
        let r = self.sample_uniform();
        let mut acc = 0.0f32;
        for (idx, &prob) in probs.iter().enumerate() {
            acc += prob;
            if r <= acc {
                return idx as u32;
            }
        }
        (probs.len().saturating_sub(1)) as u32
    }

    /// Computes full softmax distribution over vocabulary applying temperature, top-k, and top-p.
    pub fn compute_distribution(
        &self,
        logits: &[f32],
        context_tokens: &[u32],
        params: &SamplerParams,
    ) -> Vec<f32> {
        let vocab_size = logits.len();
        if vocab_size == 0 {
            return Vec::new();
        }

        let mut filtered: Vec<(usize, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, l)| l.is_finite())
            .collect();

        if filtered.is_empty() {
            return vec![1.0 / vocab_size as f32; vocab_size];
        }

        // Repetition penalty
        if params.repetition_penalty > 1.0 && !context_tokens.is_empty() {
            let pen = params.repetition_penalty;
            for &tok in context_tokens {
                let idx = tok as usize;
                if idx < vocab_size {
                    for cand in &mut filtered {
                        if cand.0 == idx {
                            if cand.1 > 0.0 {
                                cand.1 /= pen;
                            } else {
                                cand.1 *= pen;
                            }
                        }
                    }
                }
            }
        }

        if params.temperature <= 1e-4 {
            let mut best_idx = 0;
            let mut best_logit = f32::NEG_INFINITY;
            for &(idx, logit) in &filtered {
                if logit > best_logit {
                    best_logit = logit;
                    best_idx = idx;
                }
            }
            let mut one_hot = vec![0.0f32; vocab_size];
            one_hot[best_idx] = 1.0;
            return one_hot;
        }

        let inv_temp = 1.0 / params.temperature;
        for cand in &mut filtered {
            cand.1 *= inv_temp;
        }

        filtered.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if params.top_k > 0 && params.top_k < filtered.len() {
            filtered.truncate(params.top_k);
        }

        let max_logit = filtered[0].1;
        let mut sum_exp = 0.0f32;
        let mut probs: Vec<(usize, f32)> = Vec::with_capacity(filtered.len());
        for (idx, logit) in filtered {
            let exp_val = (logit - max_logit).exp();
            probs.push((idx, exp_val));
            sum_exp += exp_val;
        }

        if sum_exp <= 0.0 {
            let mut one_hot = vec![0.0f32; vocab_size];
            if !probs.is_empty() {
                one_hot[probs[0].0] = 1.0;
            }
            return one_hot;
        }

        let inv_sum = 1.0 / sum_exp;
        for p in &mut probs {
            p.1 *= inv_sum;
        }

        if params.top_p < 1.0 {
            let mut cumsum = 0.0f32;
            let mut cutoff_idx = probs.len();
            for (i, p) in probs.iter().enumerate() {
                cumsum += p.1;
                if cumsum >= params.top_p {
                    cutoff_idx = (i + 1).min(probs.len());
                    break;
                }
            }
            probs.truncate(cutoff_idx.max(1));

            let mut new_sum = 0.0f32;
            for p in &probs {
                new_sum += p.1;
            }
            if new_sum > 0.0 {
                let inv = 1.0 / new_sum;
                for p in &mut probs {
                    p.1 *= inv;
                }
            }
        }

        let mut full_dist = vec![0.0f32; vocab_size];
        for (idx, prob) in probs {
            full_dist[idx] = prob;
        }
        full_dist
    }

    /// Checks if a token ID or generated text piece triggers any stop sequences.
    pub fn is_stop_sequence(
        token_id: u32,
        piece: &str,
        stop_tokens: &[u32],
        stop_strings: &[String],
    ) -> bool {
        if stop_tokens.contains(&token_id) {
            return true;
        }
        for s in stop_strings {
            if !s.is_empty() && (piece.contains(s) || piece == s) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let mut sampler = Sampler::new(42);
        let logits = vec![1.0, 5.0, 2.0, -1.0, 4.5];
        let token = sampler.sample(&logits, &[], &SamplerParams::greedy());
        assert_eq!(token, 1, "must select argmax (index 1 = 5.0)");
    }

    #[test]
    fn test_repetition_penalty() {
        let mut sampler = Sampler::new(42);
        // Candidate 0 has 5.0, Candidate 1 has 4.8
        let logits = vec![5.0, 4.8, 1.0];
        let mut params = SamplerParams::greedy();
        params.repetition_penalty = 1.5;

        // Context contains token 0 -> 5.0 / 1.5 = 3.33 < 4.8
        let token = sampler.sample(&logits, &[0], &params);
        assert_eq!(token, 1, "token 0 should be penalized below token 1");
    }

    #[test]
    fn test_top_k_filtering() {
        let mut sampler = Sampler::new(42);
        let logits = vec![10.0, 9.0, 1.0, 0.5, 0.1];
        let params = SamplerParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 2,
            repetition_penalty: 1.0,
            seed: Some(12345),
        };
        for _ in 0..20 {
            let token = sampler.sample(&logits, &[], &params);
            assert!(token == 0 || token == 1, "token {token} must be in top 2");
        }
    }

    #[test]
    fn test_stop_sequence_detection() {
        let stop_tokens = vec![151645, 151643]; // <|im_end|>, <|endoftext|>
        let stop_strings = vec!["<|im_end|>".to_string(), "Human:".to_string()];

        assert!(Sampler::is_stop_sequence(151645, "", &stop_tokens, &stop_strings));
        assert!(Sampler::is_stop_sequence(100, "<|im_end|>", &stop_tokens, &stop_strings));
        assert!(Sampler::is_stop_sequence(100, "Hello Human: how are you", &stop_tokens, &stop_strings));
        assert!(!Sampler::is_stop_sequence(100, "Hello world", &stop_tokens, &stop_strings));
    }
}
