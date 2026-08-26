//! Context N-Gram Draft Proposer for Speculative Decoding (Phase 12, Sub-change 12.2).
//!
//! Scans sequence history for matching n-gram suffixes to propose candidate continuation
//! tokens at near-zero CPU latency (< 5 microseconds) and zero extra VRAM footprint.

/// Fast context-based n-gram speculative proposer.
#[derive(Debug, Clone)]
pub struct NgramDraftProposer {
    /// Maximum candidate tokens to propose.
    pub max_draft_k: usize,
    /// Maximum n-gram window size to match (typically 3 or 4).
    pub max_ngram_size: usize,
    /// Minimum n-gram window size to match (typically 2).
    pub min_ngram_size: usize,
}

impl Default for NgramDraftProposer {
    fn default() -> Self {
        Self {
            max_draft_k: 3,
            max_ngram_size: 4,
            min_ngram_size: 2,
        }
    }
}

impl NgramDraftProposer {
    /// Constructs a new proposer with specified draft length and n-gram bounds.
    pub fn new(max_draft_k: usize, max_ngram_size: usize, min_ngram_size: usize) -> Self {
        Self {
            max_draft_k: max_draft_k.max(1),
            max_ngram_size: max_ngram_size.max(2),
            min_ngram_size: min_ngram_size.max(1).min(max_ngram_size),
        }
    }

    /// Proposes candidate tokens by finding the longest matching suffix in `history`.
    ///
    /// `history`: Full sequence of prompt and generated tokens so far.
    /// Returns: Vec of proposed candidate tokens of length up to `max_draft_k`.
    pub fn propose(&self, history: &[u32]) -> Vec<u32> {
        let t = history.len();
        if t < self.min_ngram_size + 1 {
            return Vec::new();
        }

        // Try n-gram lengths from max_ngram_size down to min_ngram_size
        for n in (self.min_ngram_size..=self.max_ngram_size.min(t - 1)).rev() {
            let suffix = &history[t - n..t];

            // Scan historical tokens from right to left to prioritize recency
            // Stop before the current suffix itself (i.e. search in 0..t - n)
            if t < n + 1 {
                continue;
            }
            let search_limit = t - n;

            for i in (0..search_limit).rev() {
                if history[i..i + n] == *suffix {
                    // Match found! Extract continuation that followed this occurrence
                    let cont_start = i + n;
                    let cont_end = (cont_start + self.max_draft_k).min(t);
                    if cont_start < cont_end {
                        let candidates = history[cont_start..cont_end].to_vec();
                        if !candidates.is_empty() {
                            return candidates;
                        }
                    }
                }
            }
        }

        Vec::new()
    }
}
