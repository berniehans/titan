# Delta Specification: Phase 11 — Chunked Prefill & FlashAttention-2 GPU Kernel

## ADDED Requirements

### Requirement: Batched Quantized GEMM Execution
The system SHALL provide native CUDA matrix multiplication kernels (`gemm_q4k`, `gemm_q6k`, `gemm_q80`) computing $Y = X W^T$ for batch sizes $M \ge 1$ without materializing uncompressed weights in VRAM.

#### Scenario: Batched GEMM numerical parity
- **WHEN** computing $Y = X W^T$ for activation matrix $X \in \mathbb{R}^{M \times K}$ ($M \in \{16, 64, 128, 256\}$) against quantized weight tensor $W$
- **THEN** output floats SHALL match CPU reference batched GEMM with maximum relative error $< 10^{-4}$
- **AND** cosine similarity SHALL exceed 0.9999

### Requirement: FlashAttention-2 Causal Prefill
The system SHALL provide a tiled CUDA FlashAttention-2 kernel computing causal self-attention and cross-attention over resident paged KV blocks with $O(S)$ intermediate VRAM usage.

#### Scenario: FlashAttention-2 causal parity
- **WHEN** executing causal prefill attention on sequence length $S \in [1, 2048]$
- **THEN** output attention vectors SHALL match CPU reference causal attention with cosine similarity $\ge 0.9999$
- **AND** peak temporary attention memory consumption SHALL remain bounded by shared-memory tile sizes ($B_r \times B_c$) without allocating $S \times S$ global memory matrices

### Requirement: Chunked Prefill Forward Pipeline
The `ForwardDriver` SHALL support chunked prefill evaluation, splitting long prompts into bounded token slices ($S_{\text{chunk}} \le \text{CHUNK\_SIZE}$) and computing full forward passes in parallel batches.

#### Scenario: Multi-token prompt evaluation
- **WHEN** evaluating a multi-token prompt of length $S$
- **THEN** prefill SHALL execute via batched GEMM and FlashAttention-2 chunks
- **AND** final output logits SHALL match single-token serial prefill with cosine similarity $\ge 0.997$
- **AND** Time To First Token (TTFT) SHALL achieve at least $5\times$ speedup on prompts $\ge 128$ tokens
