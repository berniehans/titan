# kv-compression Delta Specification

## Purpose
Quantizes paged KV cache blocks to 8-bit floating point format (FP8), doubling resident context token capacity.

## Requirements

### Requirement: FP8 Key-Value Storage Format
The KV-cache subsystem SHALL support storing paged Key and Value buffers in FP8 format (`E4M3` or `E5M2`) with dynamic per-head scaling factors.

#### Scenario: 4096-token sequence on 6 GB GPU
- **WHEN** allocating KV cache for a sequence of 4096 tokens on Llama 3.2 3B
- **THEN** physical KV pool memory consumption does not exceed 300 MB
- **AND** attention output cosine similarity against FP16 reference is $\ge 0.999$.
