//! StreamingLLM Attention Sinks & H2O Heavy-Hitter KV Cache Compression.
//!
//! Enables infinite sequence generation within a fixed O(1) VRAM budget by:
//! 1. Pinning initial Attention Sink tokens (0..4) to absorb softmax denominator mass.
//! 2. Maintaining a rolling sliding window over the most recent W tokens.
//! 3. Evicting low-attention intermediate blocks when physical block capacity is exceeded.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingKvConfig {
    /// Number of initial attention sink tokens to pin permanently (default: 4).
    pub sink_tokens: usize,
    /// Number of recent context tokens in the rolling window (e.g. 256).
    pub recent_window_tokens: usize,
    /// Maximum physical token capacity allocated in VRAM.
    pub max_budget_tokens: usize,
    /// Tokens per physical block.
    pub block_tokens: usize,
}

impl Default for StreamingKvConfig {
    fn default() -> Self {
        Self {
            sink_tokens: 4,
            recent_window_tokens: 256,
            max_budget_tokens: 512,
            block_tokens: 16,
        }
    }
}

impl StreamingKvConfig {
    pub fn new(
        sink_tokens: usize,
        recent_window_tokens: usize,
        max_budget_tokens: usize,
        block_tokens: usize,
    ) -> Self {
        let block_tokens = block_tokens.max(1);
        let sink_tokens = sink_tokens.max(1);
        let recent_window_tokens = recent_window_tokens.max(block_tokens);
        let max_budget_tokens = max_budget_tokens.max(sink_tokens + recent_window_tokens);

        Self {
            sink_tokens,
            recent_window_tokens,
            max_budget_tokens,
            block_tokens,
        }
    }

    /// Number of physical blocks allocated for attention sinks.
    pub fn sink_blocks(&self) -> usize {
        (self.sink_tokens + self.block_tokens - 1) / self.block_tokens
    }

    /// Number of physical blocks in the recent rolling window.
    pub fn recent_blocks(&self) -> usize {
        (self.recent_window_tokens + self.block_tokens - 1) / self.block_tokens
    }

    /// Maximum number of active physical blocks allowed for a sequence.
    pub fn max_active_blocks(&self) -> usize {
        (self.max_budget_tokens + self.block_tokens - 1) / self.block_tokens
    }
}

/// Manages block recycling and virtual block table remapping for streaming context.
#[derive(Debug, Clone)]
pub struct StreamingKvManager {
    pub cfg: StreamingKvConfig,
    /// Cumulative attention score per physical block for H2O eviction.
    pub block_scores: Vec<f32>,
}

impl StreamingKvManager {
    pub fn new(cfg: StreamingKvConfig) -> Self {
        let max_blocks = cfg.max_active_blocks() * 4;
        Self {
            cfg,
            block_scores: vec![0.0; max_blocks],
        }
    }

    /// Determines which physical block IDs to evict when physical block table exceeds budget.
    ///
    /// Returns `(retained_blocks, evicted_blocks)`.
    pub fn prune_blocks(&self, block_table: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let max_blocks = self.cfg.max_active_blocks();
        if block_table.len() <= max_blocks {
            return (block_table.to_vec(), Vec::new());
        }

        let sink_count = self.cfg.sink_blocks().min(block_table.len());
        let recent_count = self.cfg.recent_blocks().min(block_table.len() - sink_count);

        let sink_blocks = &block_table[..sink_count];
        let recent_blocks = &block_table[block_table.len() - recent_count..];

        // Middle candidates available for H2O ranking / eviction
        let middle_blocks = &block_table[sink_count..block_table.len() - recent_count];
        let allowable_middle = max_blocks.saturating_sub(sink_count + recent_count);

        let mut scored_middle: Vec<(u32, f32)> = middle_blocks
            .iter()
            .map(|&b| {
                let score = self.block_scores.get(b as usize).copied().unwrap_or(0.0);
                (b, score)
            })
            .collect();

        // Sort descending by score to keep heavy hitters
        scored_middle.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let retained_middle: Vec<u32> = scored_middle.iter().take(allowable_middle).map(|p| p.0).collect();
        let evicted_middle: Vec<u32> = scored_middle.iter().skip(allowable_middle).map(|p| p.0).collect();

        let mut retained = Vec::with_capacity(max_blocks);
        retained.extend_from_slice(sink_blocks);
        retained.extend_from_slice(&retained_middle);
        retained.extend_from_slice(recent_blocks);

        (retained, evicted_middle)
    }

    /// Records observed attention score mass for a block.
    pub fn record_attention_mass(&mut self, block_id: u32, mass: f32) {
        let idx = block_id as usize;
        if idx >= self.block_scores.len() {
            self.block_scores.resize(idx + 64, 0.0);
        }
        self.block_scores[idx] += mass;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_kv_pruning_preserves_sink_and_recent() {
        let cfg = StreamingKvConfig::new(16, 32, 64, 16);
        // sink_blocks = 1 (16 tok), recent_blocks = 2 (32 tok), max_active_blocks = 4 (64 tok)
        let manager = StreamingKvManager::new(cfg);

        // Sequence with 8 allocated blocks: [100, 101, 102, 103, 104, 105, 106, 107]
        let original_table = vec![100, 101, 102, 103, 104, 105, 106, 107];
        let (retained, evicted) = manager.prune_blocks(&original_table);

        assert_eq!(retained.len(), 4);
        assert_eq!(retained[0], 100, "Sink block 100 must be preserved");
        assert!(retained.contains(&106), "Recent block 106 must be preserved");
        assert!(retained.contains(&107), "Recent block 107 must be preserved");

        assert_eq!(evicted.len(), 4);
        println!("Retained blocks: {:?}, Evicted blocks: {:?}", retained, evicted);
    }
}
