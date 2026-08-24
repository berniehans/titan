# Change: PagedAttention decode kernel (`paged_attention.cu`) (Phase 6.5)

## Why
Attention over the resident paged KV cache (built in 6.4) needs a decode kernel that walks the scattered block table with online softmax in one pass — no intermediate allocations. This is a proven production design (vLLM PagedAttention); Titan ports it rather than inventing it.

## What Changes
- `paged_attention.cu` (new `engine-cuda` module): single-pass decode over the block table with online (streaming) softmax — softmax running-sum accumulated per block so the kernel touches physical blocks in block-table order.
- GQA head mapping: n_head groups map onto n_head_kv KV heads via head-group index arithmetic.
- Zero intermediate allocations: no runtime cudaMalloc; all scratch is register/local-memory per block.
- Causal handling for the prefill path (masked/halving the block-local causal region).
- CPU SDPA reference (engine-core) for parity across scattered multi-block sequences.

## Traceability gate (port of vLLM)
- Top-of-source comment REQUIRED in `paged_attention.cu`:
  `// Port of vLLM PagedAttention decode kernel (csrc/pos_encoding_kernels.cu, paged attention v1/v2) (Apache-2.0)`.

## Gate
Parity vs CPU SDPA reference across scattered multi-block sequences (1..2048 tokens) cos-sim ≥ 0.9999; no runtime cudaMalloc (asserted by test that the launch path makes no allocator calls during the kernel).

## Non-goals
- No eviction/swapping to host RAM (matches 4.4 policy).
- No batching concurrency beyond head-map correctness; no fused causal/prefill beyond the block-local restriction.

## Impact
- **Affected code:** `engine-cuda/src/`, `paged_attention.cu`, `paged_attention.rs` wrapper, `engine-core` SDPA reference
- **Gate:** parity ≥ 0.9999 scattered (1..2048), zero runtime cudaMalloc, all suites green

## Tasks (summary — details in tasks.md)
1. CPU SDPA reference + RED parity (1..2048, scattered multi-block)
2. Implement `paged_attention.cu` + wrapper (online softmax, GQA, zero alloc)
3. Parity + no-cudaMalloc assertion
4. Gate

## Environment notes
- NVRTC via `%LOCALAPPDATA%/Temp` PATH trick; GPU tests `#[ignore]`. vLLM reference (Apache-2.0).