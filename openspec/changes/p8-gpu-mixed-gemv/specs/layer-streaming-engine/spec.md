# Delta Specification: Phase 8 — Native GPU Mixed-Quant GEMV

## ADDED Requirements

### Requirement: Native GPU Q6_K dequantization and GEMV
The system SHALL provide native CUDA kernels for dequantizing `Q6_K` super-blocks (256 weights in 210 bytes) and executing fused matrix-vector products directly on GPU registers/shared memory without CPU fallback or intermediate VRAM allocations.

#### Scenario: Q6_K GPU block-level parity
- **WHEN** GPU `dequant_q6k` unpacks raw Q6_K byte buffers
- **THEN** output floats SHALL match CPU reference floats with maximum relative error $< 10^{-4}$
- **AND** cosine similarity SHALL exceed 0.9999

#### Scenario: Full-layer GPU execution without host synchronization
- **WHEN** `ForwardDriver` executes decode steps across layers containing mixed Q4_K_M and Q6_K tensors
- **THEN** all matrix multiplications SHALL execute on CUDA streams
- **AND** no intermediate layer activations SHALL synchronize or transfer to host CPU memory

### Requirement: GPU Accelerated Throughput Target
The system SHALL achieve accelerated autoregressive decoding when operating on mixed-quantization models.

#### Scenario: Qwen3-0.6B decode throughput
- **WHEN** executing multi-step autoregressive generation on the Qwen3-0.6B Q4_K_M fixture on RTX 3060
- **THEN** steady-state generation speed SHALL exceed 15.0 tokens/second
