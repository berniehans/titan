//! Copy-on-Write (CoW) Virtual Block Table for Zero-Copy Sequence Branching.
//!
//! Enables atomic $O(1)$ branching for reasoning trees, subagents, and speculative rollouts
//! without duplicating physical GPU memory. Physical page copies are deferred until a branch
//! mutates or appends new tokens to a shared block.

use crate::radix::PhysicalBlockId;
use std::sync::Arc;

/// A physical KV-cache block descriptor tracked by reference-counted pointers.
#[derive(Debug, PartialEq, Eq)]
pub struct SharedBlock {
    pub id: PhysicalBlockId,
    pub is_dirty: bool,
}

/// Virtual Block Table with Copy-on-Write semantics for branching agent rollouts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CowBlockTable {
    pub blocks: Vec<Arc<SharedBlock>>,
    pub seq_len: usize,
}

impl Default for CowBlockTable {
    fn default() -> Self {
        Self::new()
    }
}

impl CowBlockTable {
    /// Creates an empty sequence block table.
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            seq_len: 0,
        }
    }

    /// Forks the sequence table in $O(1)$ time by cloning atomic pointers.
    pub fn fork(&self) -> Self {
        Self {
            blocks: self.blocks.clone(),
            seq_len: self.seq_len,
        }
    }

    /// Appends a new physical block to this sequence table.
    pub fn append_block(&mut self, block_id: PhysicalBlockId) {
        self.blocks.push(Arc::new(SharedBlock {
            id: block_id,
            is_dirty: false,
        }));
    }

    /// Returns true if the block at `block_index` is currently shared with another branch.
    pub fn is_shared(&self, block_index: usize) -> bool {
        self.blocks
            .get(block_index)
            .map(|b| Arc::strong_count(b) > 1)
            .unwrap_or(false)
    }

    /// Returns true if the active tail block is shared and requires a Copy-on-Write duplication.
    pub fn tail_needs_cow(&self) -> bool {
        if let Some(last) = self.blocks.last() {
            Arc::strong_count(last) > 1
        } else {
            false
        }
    }

    /// Replaces the active tail block with a freshly allocated physical block copy.
    pub fn perform_tail_cow(&mut self, new_block_id: PhysicalBlockId) {
        if !self.blocks.is_empty() {
            let last_idx = self.blocks.len() - 1;
            self.blocks[last_idx] = Arc::new(SharedBlock {
                id: new_block_id,
                is_dirty: true,
            });
        }
    }

    /// Returns the linear slice of physical block IDs backing this sequence.
    pub fn physical_block_ids(&self) -> Vec<PhysicalBlockId> {
        self.blocks.iter().map(|b| b.id).collect()
    }

    /// Number of blocks in this sequence table.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Returns true if the table contains no blocks.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}
