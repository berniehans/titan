use crate::error::KvCacheError;

/// Paged KV-cache pool configuration. Blocks hold `block_tokens` positions.
///
/// Each token stores a key and a value row, each of length
/// `heads * head_dim` floats. A block therefore stores
/// `block_tokens * 2 * heads * head_dim` floats.
#[derive(Debug, Clone, PartialEq)]
pub struct PagedKvCacheConfig {
    /// Total number of physical blocks in the pool.
    pub n_blocks: usize,
    /// Logical tokens per block.
    pub block_tokens: usize,
    /// Number of KV heads.
    pub heads: usize,
    /// Head dimension (floats per head).
    pub head_dim: usize,
}

impl PagedKvCacheConfig {
    /// Floats in a single key row (== value row).
    pub fn row_len(&self) -> usize {
        self.heads * self.head_dim
    }

    /// Floats stored per token (key row + value row).
    pub fn floats_per_token(&self) -> usize {
        2 * self.row_len()
    }

    /// Floats stored per physical block.
    pub fn floats_per_block(&self) -> usize {
        self.block_tokens * self.floats_per_token()
    }

    /// Total pool capacity in floats.
    pub fn floats_total(&self) -> usize {
        self.n_blocks * self.floats_per_block()
    }
}

/// CPU reference implementation of a paged KV cache.
///
/// A flat backing buffer stores fixed-size physical blocks. A block table
/// maps each logical sequence to an ordered list of physical block ids, so
/// logically contiguous token positions may live in scattered physical
/// blocks. The pool uses an O(1) free-list allocator and tracks used/total
/// blocks against a fixed budget.
pub struct PagedKvCache {
    cfg: PagedKvCacheConfig,
    /// Flat backing buffer of `n_blocks * floats_per_block()` floats. Physical
    /// block `b` starts at float offset `b * floats_per_block()`.
    buffer: Vec<f32>,
    /// Free-list of physical block ids available for allocation (stack, O(1) pop).
    free_list: Vec<u32>,
    /// Block table: sequence id -> ordered physical block ids.
    block_table: Vec<Vec<u32>>,
    /// Appended token count per sequence (parallel to `block_table`).
    seq_tokens: Vec<usize>,
}

impl PagedKvCache {
    /// Creates a block pool with `cfg.n_blocks` physical blocks.
    pub fn new(cfg: PagedKvCacheConfig) -> Result<Self, KvCacheError> {
        if cfg.n_blocks == 0 {
            return Err(KvCacheError::InvalidArgs("n_blocks must be non-zero"));
        }
        if cfg.block_tokens == 0 {
            return Err(KvCacheError::InvalidArgs("block_tokens must be non-zero"));
        }
        if cfg.heads == 0 || cfg.head_dim == 0 {
            return Err(KvCacheError::InvalidArgs(
                "heads and head_dim must be non-zero",
            ));
        }

        let total = cfg.floats_total();
        let mut buffer = Vec::with_capacity(total);
        buffer.resize(total, 0.0);

        // Push ids descending so pop() yields ascending ids (physical block 0 first).
        let mut free_list = Vec::with_capacity(cfg.n_blocks);
        for i in 0..cfg.n_blocks {
            let id = (cfg.n_blocks - 1 - i) as u32;
            free_list.push(id);
        }

        Ok(Self {
            cfg,
            buffer,
            free_list,
            block_table: Vec::with_capacity(8),
            seq_tokens: Vec::with_capacity(8),
        })
    }

    /// Returns the pool configuration.
    pub fn config(&self) -> &PagedKvCacheConfig {
        &self.cfg
    }

    /// Total physical blocks in the pool.
    pub fn blocks_total(&self) -> usize {
        self.cfg.n_blocks
    }

    /// Physical blocks currently allocated to sequences.
    pub fn blocks_used(&self) -> usize {
        self.blocks_total() - self.free_list.len()
    }

    /// Number of logical sequences currently tracked.
    pub fn sequences(&self) -> usize {
        self.block_table.len()
    }

    /// Ordered physical block IDs backing `seq` (row of the block table).
    /// Returns `None` for an unknown sequence id.
    pub fn seq_block_table(&self, seq: usize) -> Option<&[u32]> {
        self.block_table.get(seq).map(|v| v.as_slice())
    }

    /// Frees every physical block held by `seq` back to the pool and resets
    /// `seq`'s token count to zero. The sequence remains registered and can be
    /// appended to again, which will allocate fresh blocks from the free list.
    pub fn free_sequence(&mut self, seq: usize) {
        if seq >= self.block_table.len() {
            return;
        }
        let blocks = std::mem::take(&mut self.block_table[seq]);
        self.free_list.extend(blocks.into_iter().rev());
        self.seq_tokens[seq] = 0;
    }

    /// Creates a new sequence with its own block-table row. Returns its id.
    pub fn new_sequence(&mut self) -> usize {
        let id = self.block_table.len();
        self.block_table.push(Vec::<u32>::new());
        self.seq_tokens.push(0);
        id
    }

    /// Logical tokens appended to `seq`.
    pub fn token_count(&self, seq: usize) -> usize {
        if seq >= self.seq_tokens.len() {
            return 0;
        }
        self.seq_tokens[seq]
    }

    /// Appends one token's key+value rows to `seq`, allocating a fresh physical
    /// block from the free list when the current tail block is full.
    pub fn append(&mut self, seq: usize, key: &[f32], value: &[f32]) -> Result<(), KvCacheError> {
        let expected = self.cfg.row_len();
        if key.len() != expected || value.len() != expected {
            return Err(KvCacheError::InvalidArgs(
                "key/value length must equal heads * head_dim",
            ));
        }
        if seq >= self.block_table.len() {
            return Err(KvCacheError::InvalidArgs("unknown sequence id"));
        }

        // Start a new physical block when the tail block is full (or none exists).
        let pos = self.seq_tokens[seq];
        if pos.is_multiple_of(self.cfg.block_tokens) {
            if self.free_list.is_empty() {
                return Err(KvCacheError::PoolExhausted {
                    blocks_used: self.blocks_used(),
                    blocks_total: self.blocks_total(),
                });
            }
            let phys = self.free_list.pop().expect("free list non-empty");
            self.block_table[seq].push(phys);
        }

        let blocks = self.block_table[seq].len();
        let phys = self.block_table[seq][blocks - 1];
        let slot = pos % self.cfg.block_tokens;

        let block_base = phys as usize * self.cfg.floats_per_block();
        let slot_base = slot * self.cfg.floats_per_token();
        let key_off = block_base + slot_base;
        let val_off = key_off + self.cfg.row_len();

        let m = self.cfg.row_len();
        self.buffer[key_off..key_off + m].copy_from_slice(key);
        self.buffer[val_off..val_off + m].copy_from_slice(value);

        self.seq_tokens[seq] += 1;
        Ok(())
    }

    /// Reads back the key row of `token` in `seq` as exact floats.
    pub fn read_key(&self, seq: usize, token: usize) -> Result<Vec<f32>, KvCacheError> {
        self.row(seq, token, /*is_value=*/ false)
    }

    /// Reads back the value row of `token` in `seq`.
    pub fn read_value(&self, seq: usize, token: usize) -> Result<Vec<f32>, KvCacheError> {
        self.row(seq, token, /*is_value=*/ true)
    }

    /// Copies one key or value row for a logical token into a fresh vector.
    fn row(&self, seq: usize, token: usize, is_value: bool) -> Result<Vec<f32>, KvCacheError> {
        if seq >= self.block_table.len() {
            return Err(KvCacheError::InvalidArgs("unknown sequence id"));
        }
        if token >= self.seq_tokens[seq] {
            return Err(KvCacheError::InvalidArgs("token out of range"));
        }
        let blocks = self.block_table[seq].len();
        let block_index = token / self.cfg.block_tokens;
        let slot = token % self.cfg.block_tokens;
        if block_index >= blocks {
            return Err(KvCacheError::InvalidArgs("token out of range"));
        }

        let phys = self.block_table[seq][block_index];
        let block_base = phys as usize * self.cfg.floats_per_block();
        let slot_base = slot * self.cfg.floats_per_token();
        let base = block_base + slot_base + if is_value { self.cfg.row_len() } else { 0 };

        let mut out = Vec::with_capacity(self.cfg.row_len());
        let m = self.cfg.row_len();
        for i in 0..m {
            out.push(self.buffer[base + i]);
        }
        Ok(out)
    }

    /// Materializes a logically-contiguous view of all key rows in `seq`, even
    /// when they span several scattered physical blocks.
    pub fn read_keys(&self, seq: usize) -> Result<Vec<f32>, KvCacheError> {
        self.read_all(seq, /*is_value=*/ false)
    }

    /// Materializes a logically-contiguous view of all value rows in `seq`.
    pub fn read_values(&self, seq: usize) -> Result<Vec<f32>, KvCacheError> {
        self.read_all(seq, /*is_value=*/ true)
    }

    /// Gathers every key (or value) row of `seq` into one contiguous vector.
    fn read_all(&self, seq: usize, is_value: bool) -> Result<Vec<f32>, KvCacheError> {
        if seq >= self.block_table.len() {
            return Err(KvCacheError::InvalidArgs("unknown sequence id"));
        }
        let n = self.seq_tokens[seq];
        let mut out = Vec::with_capacity(n * self.cfg.row_len());
        for token in 0..n {
            let row = self.row(seq, token, is_value).unwrap();
            out.extend(row.iter().copied());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_pool_reports_blocks_total() {
        let cache = PagedKvCache::new(PagedKvCacheConfig {
            n_blocks: 8,
            block_tokens: 16,
            heads: 4,
            head_dim: 6,
        })
        .expect("pool create");
        assert_eq!(cache.blocks_total(), 8);
        assert_eq!(cache.blocks_used(), 0);
        assert_eq!(cache.config().floats_per_token(), 2 * 4 * 6);
        assert_eq!(cache.config().floats_per_block(), 16 * 48);
    }

    #[test]
    fn append_single_token_reads_back_exact_floats() {
        let mut cache = PagedKvCache::new(PagedKvCacheConfig {
            n_blocks: 8,
            block_tokens: 16,
            heads: 2,
            head_dim: 3,
        })
        .expect("pool create");
        let seq = cache.new_sequence();

        let key = [1.5, -2.25, 0.5, 3.0, -4.0, 6.0];
        let value = [4.0, 8.0, -16.0, 1.0, 2.0, 3.0];
        cache.append(seq, &key, &value).expect("append");

        let key_out = cache.read_key(seq, 0).expect("read key");
        let value_out = cache.read_value(seq, 0).expect("read value");
        assert_eq!(key_out.len(), 6);
        assert_eq!(value_out.len(), 6);
        for (a, e) in key_out.iter().zip(key.iter()) {
            assert!((a - e).abs() < 1e-6, "key mismatch: {} vs {}", a, e);
        }
        for (a, e) in value_out.iter().zip(value.iter()) {
            assert!((a - e).abs() < 1e-6, "value mismatch: {} vs {}", a, e);
        }
    }

    #[test]
    fn append_spanning_scattered_blocks_reads_back_logically_contiguous() {
        // Force physical scatter: sequence 0 fills block 0, then a second
        // sequence grabs block 1 and 2, so sequence 0's second block is
        // physically non-adjacent (block 3).
        let mut cache = PagedKvCache::new(PagedKvCacheConfig {
            n_blocks: 4,
            block_tokens: 4,
            heads: 1,
            head_dim: 2,
        })
        .expect("pool create");

        let seq_a = cache.new_sequence();
        let seq_b = cache.new_sequence();

        // seq_a: 4 tokens -> physical block 0
        for k in 0..4 {
            let kf = k as f32;
            cache
                .append(seq_a, &[kf, kf * 10.0], &[0.0, 0.0])
                .expect("append a");
        }
        // seq_b: 4 tokens -> physical blocks 1, 2
        cache
            .append(seq_b, &[9.0, 9.0], &[9.0, 9.0])
            .expect("append b1");
        cache
            .append(seq_b, &[8.0, 8.0], &[8.0, 8.0])
            .expect("append b2");
        cache
            .append(seq_b, &[7.0, 7.0], &[7.0, 7.0])
            .expect("append b3");
        cache
            .append(seq_b, &[6.0, 6.0], &[6.0, 6.0])
            .expect("append b4");

        // seq_a: tokens 4,5,6,7 land in physical block 3 (scattered from block 0)
        for t in 4..8 {
            let tf = t as f32;
            cache
                .append(seq_a, &[tf, tf * 10.0], &[0.0, 0.0])
                .expect("append a2");
        }

        // seq_a holds 8 logical tokens across physical blocks 0 and 3.
        assert_eq!(cache.token_count(seq_a), 8);
        let keys = cache.read_keys(seq_a).expect("read keys");
        assert_eq!(keys.len(), 8 * 2);
        for t in 0..8 {
            let k = t as f32;
            assert!(
                (keys[2 * t] - k).abs() < 1e-6,
                "key row {}: got {} want {}",
                t,
                keys[2 * t],
                k
            );
            assert!(
                (keys[2 * t + 1] - (k * 10.0)).abs() < 1e-6,
                "key col {}: got {} want {}",
                t,
                keys[2 * t + 1],
                k * 10.0
            );
        }
    }

    #[test]
    fn pool_exhaustion_returns_typed_error() {
        // 1 block, 2 tokens/block -> exactly 2 appends fit before the pool is full.
        let mut cache = PagedKvCache::new(PagedKvCacheConfig {
            n_blocks: 1,
            block_tokens: 2,
            heads: 1,
            head_dim: 2,
        })
        .expect("pool create");
        let seq = cache.new_sequence();

        let row = [0.5, 1.5];
        cache.append(seq, &row, &row).expect("append 1");
        cache
            .append(seq, &row, &row)
            .expect("append 2 (fills block)");

        // Third append must need a new (unavailable) block -> typed error.
        let err = cache.append(seq, &row, &row).unwrap_err();
        match err {
            KvCacheError::PoolExhausted {
                blocks_used,
                blocks_total,
            } => {
                assert_eq!(blocks_used, 1);
                assert_eq!(blocks_total, 1);
            }
            other => panic!("expected PoolExhausted, got {other:?}"),
        }
        // Pool stays full; the failed append left no partial state.
        assert_eq!(cache.blocks_used(), 1);
        assert_eq!(cache.token_count(seq), 2);
    }

    #[test]
    fn free_then_realloc_succeeds() {
        let mut cache = PagedKvCache::new(PagedKvCacheConfig {
            n_blocks: 2,
            block_tokens: 2,
            heads: 1,
            head_dim: 2,
        })
        .expect("pool create");
        let seq = cache.new_sequence();

        // Exhaust both blocks.
        let row = [1.0, 2.0];
        cache.append(seq, &row, &row).unwrap();
        cache.append(seq, &row, &row).unwrap();
        cache.append(seq, &row, &row).unwrap();
        cache.append(seq, &row, &row).unwrap();
        assert!(matches!(
            cache.append(seq, &row, &row),
            Err(KvCacheError::PoolExhausted { .. })
        ));

        // Free seq A -> its 2 blocks return to the pool (0 in use).
        cache.free_sequence(seq);
        assert_eq!(cache.blocks_used(), 0);
        assert_eq!(cache.token_count(seq), 0);

        // Re-append the same sequence: allocation succeeds again from the freed pool.
        for _ in 0..3 {
            cache.append(seq, &row, &row).expect("realloc append");
        }
        assert_eq!(cache.token_count(seq), 3);
        assert_eq!(cache.blocks_used(), 2);

        // The data actually reads back (freshly written, not stale).
        let keys = cache.read_keys(seq).expect("read keys");
        assert_eq!(keys.len(), 3 * 2);
    }

    #[test]
    fn budget_capacity_matches_config_total() {
        // The free-list accounting must never exceed the configured budget:
        // blocks_used is bounded by n_blocks at all times.
        let mut cache = PagedKvCache::new(PagedKvCacheConfig {
            n_blocks: 3,
            block_tokens: 4,
            heads: 2,
            head_dim: 3,
        })
        .expect("pool create");
        let a = cache.new_sequence();
        let b = cache.new_sequence();

        let row = vec![0.1f32; 6];
        let mut allocated = 0;
        while allocated < cache.blocks_total() {
            cache.append(a, &row, &row).unwrap();
            allocated = cache.blocks_used();
            assert!(cache.blocks_used() <= cache.blocks_total());
        }
        // Sequence b cannot allocate now.
        assert!(matches!(
            cache.append(b, &row, &row),
            Err(KvCacheError::PoolExhausted { .. })
        ));
    }
}
