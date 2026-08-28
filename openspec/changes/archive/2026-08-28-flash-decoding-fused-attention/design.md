# Architecture Design: FlashDecoding & Split-KV PagedAttention

## 1. Algorithm Overview

FlashDecoding splits the sequence dimension $N$ (KV cache tokens) into $S$ independent chunks of size $B = 256$ tokens:
$$S = \left\lceil \frac{N}{B} \right\rceil$$

```
                         FLASH-DECODING PIPELINE
                         
  Query Vector Q [n_heads, d_head]       Paged KV Pool (Global VRAM)
            │                                      │
            ├───────────────┬──────────────────────┤
            ▼               ▼                      ▼
      [ Split 0 ]      [ Split 1 ]          [ Split S-1 ]
    (Tokens 0..255)  (Tokens 256..511)    (Tokens .. N-1)
            │               │                      │
            ▼               ▼                      ▼
     (m_0, l_0, acc_0) (m_1, l_1, acc_1)   (m_S, l_S, acc_S)  <- Partial Softmax in VRAM
            │               │                      │
            └───────────────┼──────────────────────┘
                            ▼
              [ FlashDecoding Merge Kernel ]
         (Log-Sum-Exp Online Reduction across S splits)
                            │
                            ▼
               Final Attention Output Vector O
```

### Stage 1: Parallel Split-KV Attention (`flash_decoding_split_kernel`)
- **Grid Shape:** `(n_heads, S)` where $S = \min(\lceil N / 256 \rceil, 32)$.
- **Thread Block:** 32 threads (1 warp for head_dim=128 using `float4` vectorized loads).
- **Execution:**
  Each thread block $(h, s)$ iterates only through block indices $b \in [s \cdot \text{blocks\_per\_split}, (s+1) \cdot \text{blocks\_per\_split})$ in the BlockTable.
  It performs standard online softmax over its local chunk and writes:
  - `partial_m[h, s]` = local max score $m_s$
  - `partial_l[h, s]` = local denominator $l_s$
  - `partial_acc[h, s, 0..d_head-1]` = local unnormalized weighted value vector $acc_s$

### Stage 2: Fused Online Softmax Reduction (`flash_decoding_reduce_kernel`)
- **Grid Shape:** `(n_heads)`
- **Thread Block:** 32 threads (1 warp for head_dim=128).
- **Execution:**
  Loads $(m_s, l_s, acc_s)$ for all $s \in 0..S-1$ across shared memory registers.
  Computes the global maximum:
  $$m_{\text{global}} = \max_{s \in 0..S-1} m_s$$
  Computes the global normalizer:
  $$l_{\text{global}} = \sum_{s=0}^{S-1} e^{m_s - m_{\text{global}}} l_s$$
  Computes the final output:
  $$O_i = \frac{1}{l_{\text{global}}} \sum_{s=0}^{S-1} e^{m_s - m_{\text{global}}} acc_{s, i}$$
  Writes directly to `attn_out[h, 0..d_head-1]`.

## 2. Dynamic Adaptive Dispatch

If $N \le 256$ ($S = 1$), the engine skips the reduction stage entirely and runs single-pass fused PagedAttention, ensuring zero latency regression on short prompts.
