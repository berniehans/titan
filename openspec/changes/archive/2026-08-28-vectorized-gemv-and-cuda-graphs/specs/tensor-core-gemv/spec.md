## ADDED Requirements

### Requirement: 128-bit Vectorized Coalesced Quantized Loads
The quantized GEMV kernel suite SHALL load super-block quantized byte arrays (qs) into registers using 128-bit vectorized transactions (uint4) to maximize DRAM memory bus saturation.

#### Scenario: Vectorized Q4_K super-block load
- **WHEN** reading the 128-byte payload of a Q4_K super-block in gemm_q4k_mma_kernel
- **THEN** threads issue 128-bit uint4 memory operations, achieving $\ge 160\text{ GB/s}$ effective memory bandwidth on a 192-bit bus.

### Requirement: End-to-End Fused FFN Pipeline
The compute engine SHALL execute FFN RMSNorm, Gate projection, Up projection, SwiGLU activation, and Down projection in an integrated pipeline without intermediate global VRAM roundtrips.

#### Scenario: Single-pass FFN execution
- **WHEN** computing the feed-forward network layer during decode
- **THEN** intermediate SwiGLU activations are produced directly into registers/shared memory for immediate consumption by the Down projection kernel.
