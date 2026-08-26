/// Decode loop primitives for one generation session.
///
/// A `GenerationSession` owns a single sequence inside a caller-owned
/// [`PagedKvCache`]. Each decode step appends the current token's key/value
/// rows and reads them back, then derives the next token deterministically.
///
/// # Honesty constraint (placeholder forward pass)
///
/// The engine's real "forward pass" (attention / matmul) is not wired yet.
/// Between embedding and vocabulary output the per-layer work is currently a
/// stub: layer bytes are streamed and dequantized, then a **deterministic
/// placeholder** `stub_next_token` collapses `(token, digest_of_dequantized_layer)`
/// into a token id. It is not a trained model and makes no quality claim. Any
/// real decode loop in a later phase replaces `stub_next_token` alone.
use engine_kvcache::{KvCacheError, PagedKvCache};

/// A live sequence inside the shared KV pool.
///
/// The struct is cheap and stateless enough for the scheduler to track many
/// sessions in one `Vec`. The cache itself is owned by the scheduler; a
/// session only stores its `seq` id, the token currently being decoded, and
/// the step budget it has consumed.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenerationSession {
    /// Paged-KV sequence id this session decodes into.
    pub seq: usize,
    /// Token id currently being decoded / appended this step.
    pub current: u32,
    /// Number of tokens generated so far.
    pub steps: u32,
    /// Upper bound exclusive for `next_token` (the stub vocab size).
    pub vocab: u32,
    /// Hard cap on generated completion tokens.
    pub max_tokens: u32,
}

impl GenerationSession {
    /// Allocates a fresh sequence and appends the stub prompt token row.
    #[allow(clippy::too_many_arguments)]
    pub fn begin(
        cache: &mut PagedKvCache,
        seq: usize,
        vocab: u32,
        prompt_token: u32,
        max_tokens: u32,
    ) -> Result<Self, KvCacheError> {
        let row_len = cache.config().row_len();
        let (key, value) = kv_row(prompt_token, row_len);
        cache.append(seq, &key, &value)?;
        Ok(Self {
            seq,
            current: prompt_token,
            steps: 0,
            vocab,
            max_tokens,
        })
    }

    /// Advances the session one decode step.
    ///
    /// Appends the current token's KV rows, reads the freshly-written value
    /// row back, feeds it through the placeholder digest/stub, then returns
    /// the produced next token. Returns `None` only when the session has no
    /// budget left (`steps == max_tokens`) and should be retired.
    pub fn step(&mut self, cache: &mut PagedKvCache) -> Option<u32> {
        if self.is_finished() {
            return None;
        }
        let row_len = cache.config().row_len();
        let index = cache.token_count(self.seq);
        let (key, value) = kv_row(self.current, row_len);
        cache.append(self.seq, &key, &value).expect("append kv");
        let read_back = cache.read_key(self.seq, index).expect("read kv");
        let mut digest = digest_layer(read_back.as_slice());
        // Bind the current token id in so sessions with different histories diverge.
        digest = (digest.wrapping_mul(31)).wrapping_add(self.current as u64);

        let next = stub_next_token(self.current, digest, self.vocab);
        self.current = next;
        self.steps += 1;
        Some(next)
    }

    /// Whether `step` has finished or the session has no budget left.
    pub fn is_finished(&self) -> bool {
        self.max_tokens == 0 || self.steps >= self.max_tokens
    }
}

/// Legacy deterministic placeholder next-token.
///
/// Returns a token id in `1..vocab`. Replaced by `ForwardDriver` in Phase 6.8
/// as the real model generator; retained as a legacy/fallback alias.
pub fn stub_next_token(token: u32, digest: u64, vocab: u32) -> u32 {
    let a = token.wrapping_mul(2654435761u32);
    let b = (digest & 0xffff) as u32;
    // Fold to [1, vocab) deterministically.
    (a ^ b).wrapping_rem(vocab) + 1
}

/// Deterministic 64-bit digest over an array of dequantized floats.
///
/// FNV-1a over the floor-scaled integer encoding of each float. Deterministic
/// on the same input; used to make the placeholder stub reproducible.
pub fn digest_layer(row: &[f32]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for v in row {
        let scaled = ((*v).abs() * 1000.0) as u64;
        h = (h ^ scaled).wrapping_mul(1099511628211);
    }
    h
}

/// Deterministic per-token key/value rows.
///
/// The token content is encoded into the row floats. When the real KV write
/// kernel lands these rows come from the dequant pipeline; today they are a
/// deterministic enumeration over the token id.
pub fn kv_row(token: u32, row_len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut key = Vec::<f32>::with_capacity(row_len);
    let mut value = Vec::<f32>::with_capacity(row_len);
    for i in 0..row_len {
        let ki = (token.wrapping_mul(7)).wrapping_add(i as u32) % 97;
        key.push((ki as f32) * 0.25);
        let vi = (token.wrapping_mul(13)).wrapping_add(i as u32) % 89;
        value.push((vi as f32) * 0.5);
    }
    (key, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> PagedKvCache {
        let cfg = engine_kvcache::PagedKvCacheConfig {
            n_blocks: 8,
            block_tokens: 64,
            heads: 1,
            head_dim: 1,
        };
        PagedKvCache::new(cfg).expect("pool")
    }

    fn run(vocab: u32, prompt: u32, max_tokens: u32) -> Vec<u32> {
        let mut cache = cache();
        let seq = cache.new_sequence();
        let mut session =
            GenerationSession::begin(&mut cache, seq, vocab, prompt, max_tokens).expect("begin");
        let mut out = Vec::new();
        while !session.is_finished() {
            if let Some(tok) = session.step(&mut cache) {
                out.push(tok);
            }
        }
        out
    }

    #[test]
    fn next_token_stub_is_deterministic() {
        let a = run(1000, 42, 7);
        let b = run(1000, 42, 7);
        assert_eq!(a.len(), 7);
        assert_eq!(a, b, "same input must produce same token sequence");
        for t in &a {
            assert!((1..1000).contains(t), "token {t} out of vocab range");
        }
    }

    #[test]
    fn different_prompt_digest_diverges_sequence() {
        let a = run(1000, 1, 7);
        let b = run(1000, 2, 7);
        assert_ne!(a, b, "different prompt must diverge");
    }
}
