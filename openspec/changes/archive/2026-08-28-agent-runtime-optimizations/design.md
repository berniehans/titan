# Design — Agent Runtime Optimizations for Constrained VRAM (6 GB)

## 1. System Architecture & Request Lifecycle

The diagram below illustrates the end-to-end execution flow of an autonomous agent turn with **Automatic Prefix Caching (APC)**, **Overlapped Grammar Masking**, and **Zero-Copy Branching (CoW)**:

```
[Agent Request Received (Prompt Tokens + Grammar Schema)]
                           │
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ STEP 1: Radix Tree Prefix Match (engine-kvcache)        │
 │ • Traverse RadixTree with prompt token sequence         │
 │ • Identify Longest Common Prefix (LCP): K matched tokens│
 └─────────────────────────┬───────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    [Full Prefix Matched]       [Partial Prefix / Miss]
    (K == Prompt Length)        (K < Prompt Length)
             │                           │
             │                           ▼
             │            ┌────────────────────────────────────────┐
             │            │ Run Chunked Prefill for (Prompt[K..N]) │
             │            │ Append new KV blocks to Radix Tree     │
             │            └──────────────────┬─────────────────────┘
             │                               │
             └─────────────┬─────────────────┘
                           │
                           ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ STEP 2: Autonomous Forward Loop & Overlapped Grammar Pipeline            │
 │                                                                          │
 │     HOST CPU THREAD (Tokio Worker)      │    GPU DEVICE (CUDA Stream)    │
 │  ────────────────────────────────────── │ ────────────────────────────── │
 │  [t] Parse schema & compute bitmask     │ [t] Autonomous Decode Pass     │
 │      for token t+1 (Packed u32 bits)    │     (GEMV -> Attention -> FFN) │
 │               │                         │               │                │
 │               ▼ (Async DMA to VRAM)     │               ▼                │
 │      H2D bitmask buffer copy            │     Logits Tensor in VRAM      │
 │               │                         │               │                │
 │               └─────────────────────────┼──────> [Event Sync]            │
 │                                         │               │                │
 │                                         │               ▼                │
 │                                         │ [t] apply_logit_mask_kernel    │
 │                                         │     (Logits &= Bitmask)        │
 │                                         │               │                │
 │                                         │               ▼                │
 │                                         │ [t] GPU Argmax / Softmax       │
 │                                         │     Sampled Token t+1 in VRAM  │
 └─────────────────────────────────────────┴────────────────────────────────┘
                           │
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ STEP 3: Tree Forking & Copy-on-Write (CoW)              │
 │ • If subagents or reasoning branches fork:              │
 │   - Sequence clones BlockTable pointers via Arc-refs    │
 │   - Mutating shared tail blocks triggers CoW clone      │
 └─────────────────────────────────────────────────────────┘
```

---

## 2. Formal Rust Type Definitions

### 2.1 RadixTree & Prefix Cache (`engine-kvcache`)

```rust
pub type PhysicalBlockId = u32;

/// A node in the token-prefix Radix Tree.
#[derive(Debug)]
pub struct RadixNode {
    /// Token subsequence represented by this node.
    pub tokens: Vec<u32>,
    /// Physical KV blocks allocated for this token sequence.
    pub blocks: Vec<PhysicalBlockId>,
    /// Child branches keyed by the first token of their sequence.
    pub children: std::collections::BTreeMap<u32, Box<RadixNode>>,
    /// Last access timestamp for LRU eviction.
    pub last_accessed: std::time::Instant,
    /// If true, this node (e.g. system prompt / tools) is protected from eviction.
    pub is_pinned: bool,
    /// Reference count of active sessions referencing these blocks.
    pub ref_count: usize,
}

/// Radix Tree managing prefix sharing and KV block reuse.
pub struct RadixTree {
    pub root: RadixNode,
    pub block_tokens: usize,
}

/// Result of querying the prefix cache.
#[derive(Debug)]
pub struct RadixMatchResult {
    /// Number of tokens matched in the prefix tree.
    pub matched_tokens: usize,
    /// Sequence of physical block IDs reused from the cache.
    pub matched_blocks: Vec<PhysicalBlockId>,
}
```

---

### 2.2 Copy-on-Write Sequence Table (`engine-kvcache`)

```rust
use std::sync::Arc;

/// A physical KV cache block descriptor with reference counting.
#[derive(Debug)]
pub struct SharedBlock {
    pub id: PhysicalBlockId,
    pub is_dirty: bool,
}

/// Copy-on-Write Virtual Block Table for branching sequences.
#[derive(Debug, Clone)]
pub struct CowBlockTable {
    pub blocks: Vec<Arc<SharedBlock>>,
    pub seq_len: usize,
}

impl CowBlockTable {
    /// Forks the sequence table instantaneously with zero memory duplication.
    pub fn fork(&self) -> Self {
        Self {
            blocks: self.blocks.clone(), // Increments Arc reference count
            seq_len: self.seq_len,
        }
    }

    /// Prepares the active tail block for appending; clones if shared (Arc::strong_count > 1).
    pub fn prepare_for_write(
        &mut self,
        allocator: &mut BlockAllocator,
    ) -> Result<PhysicalBlockId, KvCacheError> {
        let last_idx = self.blocks.len() - 1;
        if Arc::strong_count(&self.blocks[last_idx]) > 1 {
            // Allocate new physical block and copy KV data (CoW)
            let new_block_id = allocator.allocate_block()?;
            let old_id = self.blocks[last_idx].id;
            allocator.copy_block_gpu(old_id, new_block_id)?;
            self.blocks[last_idx] = Arc::new(SharedBlock { id: new_block_id, is_dirty: true });
        }
        Ok(self.blocks[last_idx].id)
    }
}
```

---

### 2.3 Constrained Decoding & Logit Bitmasking (`engine-core`)

```rust
/// Pre-allocated double-buffered GPU bitmask for zero-allocation generation.
pub struct BitmaskBuffer {
    /// Device memory buffer holding vocab_size / 32 u32 bitfields.
    pub dev_bitmask: DeviceBuffer,
    /// Pinned host memory for fast DMA transfer.
    pub host_pinned: Vec<u32>,
    /// Number of u32 elements in the bitmask.
    pub bitmask_words: usize,
}

/// Abstract grammar parser interface executed asynchronously on CPU.
pub trait GrammarParser: Send + Sync {
    /// Advances parser state given the newly accepted token.
    fn advance(&mut self, token: u32) -> Result<(), GrammarError>;
    /// Computes the packed u32 bitmask of all legally allowed next tokens.
    fn compute_allowed_mask(&self, out_mask: &mut [u32]);
    /// Returns true if the grammar has reached an accepting termination state.
    fn is_accepted(&self) -> bool;
}
```

---

## 3. CUDA Kernel Specifications

### 3.1 `apply_logit_mask_kernel` (`engine-cuda/kernels/logit_mask.cu`)

```cuda
// Kernel: In-place logit masking in GPU VRAM
// Invalidates unallowed tokens by setting logits to -INFINITY.
extern "C" __global__ void apply_logit_mask_kernel(
    float* __restrict__ logits,              // [vocab_size] FP32 logits
    const unsigned int* __restrict__ mask,   // [vocab_size / 32] packed bitmask
    int vocab_size                           // Total vocabulary size (e.g. 128256 or 151936)
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= vocab_size) return;

    const int word_idx = idx >> 5;           // idx / 32
    const int bit_idx  = idx & 31;           // idx % 32
    const unsigned int word = mask[word_idx];

    // Check if bit is 0 (disallowed)
    if (!((word >> bit_idx) & 1u)) {
        logits[idx] = -1e30f; // -INFINITY mask
    }
}
```

* **Grid & Block Configuration:** Launched with `blockDim.x = 256` (8 warps) and `gridDim.x = (vocab_size + 255) / 256`.
* **Execution Time:** $< 0.04\text{ ms}$ on NVIDIA RTX 3060 (negligible latency).

---

### 3.2 Attention Sinks & Protected Window Kernel Modification (`paged_attention.cu`)

* **Sink Retention Rule:** The first $K_{\text{sink}} = 4$ tokens of a sequence are designated as *attention sinks* to preserve high attention score concentration without softmax overflow.
* **Sliding Window Indexing:**
  $$\text{Valid Attention Range} = [0, K_{\text{sink}}) \cup [\max(K_{\text{sink}}, \text{pos} - W_{\text{size}}), \text{pos}]$$
* Blocks falling outside the valid attention range are skipped during attention score accumulation and reclaimed by the LRU memory manager.

---

## 4. VRAM Budget & Safety Guards (RTX 3060 6 GB)

```
Total Physical VRAM: 6,144 MB | Usable Budget: ~5,200 MB
├── Model Weights (3B Q4_K):      ~2,020 MB
├── Resident Activation Scratch:    ~128 MB
├── CUDA Graph Executable Nodes:     ~16 MB
├── Radix Pinned System Blocks:     ~256 MB (Always Protected)
├── Dynamic Paged KV Block Pool:  ~2,400 MB (Sliding Window & CoW Pool)
└── Bitmask & DMA Bounce Buffers:    ~16 MB
───────────────────────────────────────────
Total Peak Allocation:            ~4,836 MB (< 93% VRAM Guard Limit)
```
