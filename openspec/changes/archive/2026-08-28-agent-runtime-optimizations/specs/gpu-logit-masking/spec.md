# gpu-logit-masking Delta Specification

## Purpose

Provides high-performance, overlapped logit bitmasking for grammar-constrained decoding (JSON Schema, tool calling, regex) directly in GPU VRAM without full-vocabulary host-device synchronizations.

## Requirements

### Requirement: Asynchronous Host Grammar Bitmask Computation
The engine orchestration layer SHALL compute the allowed-token bitmask for token generation step $t+1$ on host CPU worker threads concurrently with the GPU forward execution of token step $t$.

#### Scenario: Tool call JSON generation step
- **WHEN** generating structured tool arguments constrained by a JSON schema
- **THEN** the CPU advances the grammar state and writes the packed `u32` token bitmask to pinned host memory while the GPU computes the transformer forward pass.

### Requirement: In-Place GPU Logit Bitmasking Kernel
The CUDA execution layer SHALL provide `apply_logit_mask_kernel` that applies the pre-computed bitmask directly to the logits buffer in GPU VRAM prior to softmax and sampling.

#### Scenario: Disallowed token suppression
- **WHEN** applying the bitmask on the vocabulary logits
- **THEN** logits corresponding to zero bits in the bitmask are set to $-1\times 10^{30}$ ($-\infty$) in less than $0.05\text{ ms}$, ensuring zero probability during argmax or stochastic sampling.

### Requirement: 100% Syntactic Grammar Conformance
The sampler subsystem SHALL enforce that only grammatically valid tokens are sampled, guaranteeing 100% syntactic compliance for structured tool invocations without retry loops.

#### Scenario: Terminal JSON property closing
- **WHEN** the grammar permits only closing brackets or commas
- **THEN** all other vocabulary tokens are suppressed, preventing hallucinated syntax or parse errors in downstream tool executors.
