# speculative-decoding Delta Specification

## Purpose

Provides dual-model GPU-resident speculative acceleration enabling high-capacity target models (3B/7B) to achieve draft model speeds (120 - 160 tok/s).

## Requirements

### Requirement: GPU-Resident Dual Model Execution
The engine SHALL host both Draft ($) and Target ($) model weights in physical GPU VRAM concurrently without exceeding total VRAM allocation limits.

#### Scenario: Llama 3.2 1B draft with 3B target
- **WHEN** loading both Llama 3.2 1B Q4_K_M and Llama 3.2 3B Q4_K_M
- **THEN** total resident VRAM allocation does not exceed 3.5 GB on a 6 GB device.

### Requirement: Parallel $-Candidate Forward Verification
The Target engine SHALL evaluate all $ candidate tokens proposed by the Draft engine in a single parallel GPU forward pass.

#### Scenario: Verification pass for K=4 tokens
- **WHEN** receiving =4$ proposed tokens from the Draft engine
- **THEN** the Target engine computes logits for all $ positions simultaneously in $\le 16\text{ ms}$, emitting  \in [1, K+1]$ validated tokens.

### Requirement: Zero-Copy BlockTable Index Rollback
The KV-cache subsystem SHALL adjust sequence position and allocated blocks via virtual block table index manipulations upon candidate rejection without data copies.

#### Scenario: Partial acceptance of candidate tokens
- **WHEN** only  < K$ candidate tokens are accepted by the speculative sampler
- **THEN** the target sequence position is set to  + M$ and non-accepted KV blocks are recycled without re-allocating memory buffers.
