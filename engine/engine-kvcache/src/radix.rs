//! Automatic Prefix Caching (APC) via Radix Tree (Trie) over token sequences.
//!
//! Enables zero-latency prompt prefill bypassing for multi-turn agent conversations,
//! system prompts, and tool-calling loops by matching the Longest Common Prefix (LCP)
//! and reusing cached physical KV blocks.

use std::collections::BTreeMap;
use std::time::Instant;

pub type PhysicalBlockId = u32;

/// A node in the token-prefix Radix Tree.
#[derive(Debug)]
pub struct RadixNode {
    /// Token subsequence stored in this edge/node.
    pub tokens: Vec<u32>,
    /// Physical KV blocks allocated for this token sequence.
    pub blocks: Vec<PhysicalBlockId>,
    /// Child branches keyed by the first token of the child branch.
    pub children: BTreeMap<u32, Box<RadixNode>>,
    /// Timestamp of last access for LRU eviction.
    pub last_accessed: Instant,
    /// If true, this node (e.g. system prompt / tool schemas) is immune to LRU eviction.
    pub is_pinned: bool,
    /// Number of active references currently attached to this node.
    pub ref_count: usize,
}

impl RadixNode {
    pub fn new(tokens: Vec<u32>, blocks: Vec<PhysicalBlockId>, is_pinned: bool) -> Self {
        Self {
            tokens,
            blocks,
            children: BTreeMap::new(),
            last_accessed: Instant::now(),
            is_pinned,
            ref_count: 0,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// Result of matching a token sequence against the Radix Tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RadixMatchResult {
    /// Total number of consecutive tokens matched in the prefix tree.
    pub matched_tokens: usize,
    /// Ordered list of physical block IDs backing the matched prefix.
    pub matched_blocks: Vec<PhysicalBlockId>,
}

/// Thread-safe Radix Tree prefix cache managing KV block reuse and LRU reclamation.
#[derive(Debug)]
pub struct RadixTree {
    pub root: RadixNode,
    pub block_tokens: usize,
}

impl RadixTree {
    /// Creates an empty Radix Tree with the specified block token capacity.
    pub fn new(block_tokens: usize) -> Self {
        Self {
            root: RadixNode::new(Vec::new(), Vec::new(), true),
            block_tokens,
        }
    }

    /// Finds the Longest Common Prefix (LCP) for `tokens` in the Radix Tree.
    /// Updates `last_accessed` on all matched nodes.
    pub fn match_prefix(&mut self, tokens: &[u32]) -> RadixMatchResult {
        let mut matched_tokens = 0;
        let mut matched_blocks = Vec::new();
        let mut current = &mut self.root;
        let mut remaining = tokens;

        current.last_accessed = Instant::now();

        while !remaining.is_empty() {
            let first_tok = remaining[0];
            if let Some(child) = current.children.get_mut(&first_tok) {
                child.last_accessed = Instant::now();
                let edge_len = child.tokens.len();
                let common_len = remaining
                    .iter()
                    .zip(&child.tokens)
                    .take_while(|&(a, b)| a == b)
                    .count();

                if common_len == edge_len {
                    // Full edge matched, advance deeper into the trie
                    matched_tokens += common_len;
                    matched_blocks.extend_from_slice(&child.blocks);
                    remaining = &remaining[common_len..];
                    current = child;
                } else {
                    // Partial edge matched
                    matched_tokens += common_len;
                    // Calculate complete blocks covered by common_len
                    let full_blocks = (common_len / self.block_tokens).min(child.blocks.len());
                    matched_blocks.extend_from_slice(&child.blocks[..full_blocks]);
                    break;
                }
            } else {
                break;
            }
        }

        RadixMatchResult {
            matched_tokens,
            matched_blocks,
        }
    }

    /// Inserts a sequence of tokens and their associated physical blocks into the Radix Tree.
    pub fn insert(&mut self, tokens: &[u32], blocks: &[PhysicalBlockId], is_pinned: bool) {
        if tokens.is_empty() {
            return;
        }

        let mut current = &mut self.root;
        let mut rem_tokens = tokens;
        let mut rem_blocks = blocks;

        while !rem_tokens.is_empty() {
            let first_tok = rem_tokens[0];

            if !current.children.contains_key(&first_tok) {
                // Branch does not exist, create new leaf node
                current.children.insert(
                    first_tok,
                    Box::new(RadixNode::new(
                        rem_tokens.to_vec(),
                        rem_blocks.to_vec(),
                        is_pinned,
                    )),
                );
                return;
            }

            // Existing branch found, check common prefix length
            let mut child = current.children.remove(&first_tok).unwrap();
            let common_len = rem_tokens
                .iter()
                .zip(&child.tokens)
                .take_while(|&(a, b)| a == b)
                .count();

            if common_len == child.tokens.len() {
                // Whole child edge matched, descend into child
                let num_blocks = child.blocks.len();
                let next_blocks = if rem_blocks.len() >= num_blocks {
                    &rem_blocks[num_blocks..]
                } else {
                    &[]
                };
                rem_tokens = &rem_tokens[common_len..];
                rem_blocks = next_blocks;
                current.children.insert(first_tok, child);
                current = current.children.get_mut(&first_tok).unwrap();
            } else {
                // Split edge: create an intermediate parent node and two children
                let split_tokens = child.tokens[..common_len].to_vec();
                let child_remaining_tokens = child.tokens[common_len..].to_vec();
                let new_first_tok_child = child_remaining_tokens[0];

                let split_blocks_count = (common_len / self.block_tokens).min(child.blocks.len());
                let split_blocks = child.blocks[..split_blocks_count].to_vec();
                let child_rem_blocks = child.blocks[split_blocks_count..].to_vec();

                let mut split_node = Box::new(RadixNode::new(split_tokens, split_blocks, is_pinned));

                // Re-attach old child with remaining suffix
                child.tokens = child_remaining_tokens;
                child.blocks = child_rem_blocks;
                split_node.children.insert(new_first_tok_child, child);

                // Insert new branch for remainder of inserted sequence (if any)
                let rem_insert_tokens = &rem_tokens[common_len..];
                let rem_insert_blocks = if rem_blocks.len() >= split_blocks_count {
                    &rem_blocks[split_blocks_count..]
                } else {
                    &[]
                };

                if !rem_insert_tokens.is_empty() {
                    let new_branch_first = rem_insert_tokens[0];
                    split_node.children.insert(
                        new_branch_first,
                        Box::new(RadixNode::new(
                            rem_insert_tokens.to_vec(),
                            rem_insert_blocks.to_vec(),
                            is_pinned,
                        )),
                    );
                }

                current.children.insert(first_tok, split_node);
                return;
            }
        }
    }

    /// Evicts unpinned nodes according to LRU policy to free up physical blocks.
    /// Returns the list of freed `PhysicalBlockId`s.
    pub fn evict_lru(&mut self, target_blocks_to_free: usize) -> Vec<PhysicalBlockId> {
        let mut freed_blocks = Vec::new();
        while freed_blocks.len() < target_blocks_to_free {
            // Find the oldest unpinned leaf node in the tree
            if let Some((_first_tok, evicted_blocks)) = Self::find_and_remove_oldest_leaf(&mut self.root) {
                freed_blocks.extend(evicted_blocks);
            } else {
                // No more unpinned nodes eligible for eviction
                break;
            }
        }
        freed_blocks
    }

    fn find_and_remove_oldest_leaf(
        current: &mut RadixNode,
    ) -> Option<(u32, Vec<PhysicalBlockId>)> {
        let mut oldest_key: Option<u32> = None;
        let mut oldest_time = Instant::now();

        // 1. Check direct children that are unpinned leaves
        for (&key, child) in &current.children {
            if !child.is_pinned && child.ref_count == 0 {
                if child.is_leaf() {
                    if oldest_key.is_none() || child.last_accessed < oldest_time {
                        oldest_time = child.last_accessed;
                        oldest_key = Some(key);
                    }
                }
            }
        }

        if let Some(key) = oldest_key {
            let removed = current.children.remove(&key).unwrap();
            return Some((key, removed.blocks));
        }

        // 2. Recursively search non-leaf children
        for child in current.children.values_mut() {
            if let Some(res) = Self::find_and_remove_oldest_leaf(child) {
                return Some(res);
            }
        }

        None
    }

    /// Computes the total number of physical KV blocks cached across the entire Radix Tree.
    pub fn total_cached_blocks(&self) -> usize {
        fn count_blocks(node: &RadixNode) -> usize {
            let mut sum = node.blocks.len();
            for child in node.children.values() {
                sum += count_blocks(child);
            }
            sum
        }
        count_blocks(&self.root)
    }

    /// Invalidates and prunes any trie branches referencing any of the specified physical blocks.
    /// Prevents stale/dangling physical block pointer reuse when blocks are recycled or pruned.
    pub fn invalidate_physical_blocks(&mut self, freed_blocks: &[PhysicalBlockId]) {
        if freed_blocks.is_empty() {
            return;
        }
        let block_set: std::collections::HashSet<PhysicalBlockId> = freed_blocks.iter().copied().collect();
        Self::prune_nodes_with_blocks(&mut self.root, &block_set);
    }

    fn prune_nodes_with_blocks(
        node: &mut RadixNode,
        block_set: &std::collections::HashSet<PhysicalBlockId>,
    ) {
        node.children.retain(|_, child| {
            let contains_freed = child.blocks.iter().any(|b| block_set.contains(b));
            !contains_freed
        });
        for child in node.children.values_mut() {
            Self::prune_nodes_with_blocks(child, block_set);
        }
    }
}
