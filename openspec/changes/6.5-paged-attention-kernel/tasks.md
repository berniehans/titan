# Tasks: 6.5-paged-attention-kernel

> Execute via bot coder. Strict TDD. One commit per task group. GPU tests: `#[ignore]` + `%LOCALAPPDATA%/Temp` PATH trick for NVRTC.

## 1. CPU SDPA reference + RED
- [ ] 1.1 Failing test: CPU SDPA reference over sequential blocks (attention matrix) matches hand-computed scores (cos-sim ≥ 0.9999)
- [ ] 1.2 Failing test: CPU SDPA over SCATTERED multi-block table (block order 3,0,2,1) same logical attention as contiguous
- [ ] 1.3 Failing test: CPU SDPA with causal mask (prefill path) applies upper-triangular restriction

## 2. Kernel — GREEN (engine-cuda)
- [ ] 2.1 Implement `paged_attention.cu` with top-of-source port comment:
  `// Port of vLLM PagedAttention decode kernel (csrc/pos_encoding_kernels.cu, paged attention v1/v2, Apache-2.0)`
- [ ] 2.2 Online-softmax single pass over block table in block order; GQA head-group mapping (n_head → n_head_kv)
- [ ] 2.3 Zero intermediate allocations: no cudaMalloc in the launch path (static scratch only)
- [ ] 2.4 Causal handling for prefill path
- [ ] 2.5 Verify PASS on local GPU (multiple seq/gqa configs, 1..2048 tokens)

## 3. Parity + zero-alloc assertion
- [ ] 3.1 Failing test (`#[ignore]`): GPU paged vs CPU SDPA across scattered multi-block sequences 1..2048 tokens, cos-sim ≥ 0.9999
- [ ] 3.2 Failing test: launch path performs ZERO runtime cudaMalloc (instrumented allocator counter)
- [ ] 3.3 Record numbers (cos-sim, max seq, budget); verify PASS

## 4. Gate
- [ ] 4.1 Full suite green (CPU + GPU `--ignored`, clippy -D warnings)
- [ ] 4.2 Gate sealed: parity ≥ 0.9999 across scattered 1–2048, no cudaMalloc at runtime