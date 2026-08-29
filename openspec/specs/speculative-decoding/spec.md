# Speculative Decoding Core Specification

## Purpose
Provides dual-model GPU-resident speculative acceleration and context n-gram speculation, enabling high-capacity target models (3B/7B) to achieve draft model generation speeds (120 - 160 tok/s) with 100% mathematical output distribution equivalence.

## Requirements

### Requirement: GPU-Resident Dual Model Execution
The engine SHALL host both Draft ($M_{\text{draft}}$) and Target ($M_{\text{target}}$) model weights in physical GPU VRAM concurrently without exceeding total VRAM allocation limits.

#### Scenario: Llama 3.2 1B draft with 3B target
- **WHEN** loading both Llama 3.2 1B Q4_K_M (~807 MB) and Llama 3.2 3B Q4_K_M (~2,020 MB)
- **THEN** total resident VRAM allocation does not exceed 3.5 GB on a 6 GB device
- **AND** both models execute forward passes without host memory eviction.

### Requirement: Parallel $K$-Candidate Forward Verification
The Target engine SHALL evaluate all $K$ candidate tokens proposed by the Draft engine in a single parallel GPU forward pass using DP4A vectorized multi-row kernels.

#### Scenario: Verification pass for K=3 candidates
- **WHEN** receiving $K=3$ proposed tokens from the Draft engine
- **THEN** the Target engine computes logits for all $K+1=4$ positions simultaneously in $\le 35\text{ ms}$
- **AND** emits all consecutively matching candidate tokens plus one bonus token with $\ge 2.7\times$ speedup per token over serial decode.

### Requirement: Zero-Copy BlockTable Index Rollback
The KV-cache subsystem SHALL adjust sequence position and allocated blocks via virtual block table index manipulations upon candidate rejection without data copies.

#### Scenario: Partial acceptance of candidate tokens
- **WHEN** only $M < K$ candidate tokens are accepted by the speculative sampler
- **THEN** the target sequence position is set to $\text{pos} + M$
- **AND** non-accepted KV blocks are recycled without re-allocating memory buffers.
