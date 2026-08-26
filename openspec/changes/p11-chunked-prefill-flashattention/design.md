# Design: Phase 11 — Chunked Prefill & FlashAttention-2 GPU Kernel

## 1. Architecture & Memory Flow

```
Prompt Tokens (len = S)
       │
       ▼ Chunk into slices S_chunk <= 128
 ┌────────────────────────────────────────────────────────────┐
 │ Stage 1: Batched Embedding & Residual Lookup               │
 │ • x_chunk = (S_chunk, h) in VRAM                           │
 └─────────────────────────────┬──────────────────────────────┘
                               │
 ┌─────────────────────────────┴──────────────────────────────┐
 │ Stage 2: Batched Quantized GEMM (M=S_chunk, K=h, N=qdim/kvd)│
 │ • Uses gemm_q4k / gemm_q6k tiled shared-memory kernels      │
 │ • Produces Q (S_chunk, nh, hd), K, V in single pass        │
 └─────────────────────────────┬──────────────────────────────┘
                               │
 ┌─────────────────────────────┴──────────────────────────────┐
 │ Stage 3: Batched RMSNorm + RoPE Fused Launch               │
 │ • Applies rotary positional embeddings across rows 0..S-1  │
 └─────────────────────────────┬──────────────────────────────┘
                               │
 ┌─────────────────────────────┴──────────────────────────────┐
 │ Stage 4: Batched KV Cache Append                           │
 │ • Appends (S_chunk, kvd) keys and values into resident     │
 │   paged KV blocks in a single grid launch                  │
 └─────────────────────────────┬──────────────────────────────┘
                               │
 ┌─────────────────────────────┴──────────────────────────────┐
 │ Stage 5: FlashAttention-2 Causal Kernel                    │
 │ • Tiled Q blocks (B_r) against K/V blocks (B_c)            │
 │ • Online softmax renormalization with running row max/sum  │
 │ • Reads resident paged KV blocks (historical + current)    │
 │ • Writes Attn Out (S_chunk, nh * hd) directly to VRAM      │
 └─────────────────────────────┬──────────────────────────────┘
                               │
 ┌─────────────────────────────┴──────────────────────────────┐
 │ Stage 6: Batched MLP (WO -> Gate/Up -> SwiGLU -> Down)    │
 │ • Batched GEMM across all intermediate feed-forward layers │
 └────────────────────────────────────────────────────────────┘
```

---

## 2. FlashAttention-2 Algorithm Details

For each attention head $h \in [0, nh)$ and query block $i \in [0, \lceil S / B_r \rceil)$:
1. Load $Q_i \in \mathbb{R}^{B_r \times hd}$ into shared memory.
2. Initialize running max vector $m_i \leftarrow -\infty$ and running sum vector $l_i \leftarrow 0$, output accumulator $O_i \leftarrow 0$.
3. For each key/value block $j \in [0, \min(i + 1, \lceil S_{\text{total}} / B_c \rceil))$:
   a. Load $K_j, V_j \in \mathbb{R}^{B_c \times hd}$ from paged KV pool into shared memory.
   b. Compute block score matrix $S_{ij} = \frac{Q_i K_j^T}{\sqrt{hd}}$.
   c. If $i == j$ (diagonal block), apply causal mask $S_{ij}[r, c] \leftarrow -\infty$ for $c > r$.
   d. Compute new row max $\tilde{m}_i = \max(m_i, \text{rowmax}(S_{ij}))$.
   e. Compute unnormalized weights $P_{ij} = \exp(S_{ij} - \tilde{m}_i)$.
   f. Update running sum $l_i \leftarrow l_i \cdot \exp(m_i - \tilde{m}_i) + \text{rowsum}(P_{ij})$.
   g. Update output $O_i \leftarrow O_i \cdot \text{diag}(\exp(m_i - \tilde{m}_i)) + P_{ij} V_j$.
   h. Update $m_i \leftarrow \tilde{m}_i$.
4. Finalize output $O_i \leftarrow O_i / l_i$ and store to global VRAM.

---

## 3. Batched Quantized GEMM Implementation

To compute $Y = X W^T$ where $X \in \mathbb{R}^{M \times K}$ is FP32 activations and $W$ is quantized super-blocks (`Q4_K` / `Q6_K` / `Q8_0`):
- Grid dimensions: `grid_x = div_ceil(N, TILE_N)`, `grid_y = div_ceil(M, TILE_M)`.
- Threads load a $TILE_M \times TILE_K$ tile of $X$ and a $TILE_N \times TILE_K$ tile of $W$ into shared memory.
- Dequantization happens in registers while iterating across the $K$ dimension.
- Results accumulated in 32-bit registers and written out in coalesced memory transactions.
