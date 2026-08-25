# Tasks: 6.5-paged-attention-kernel

> Execute via bot coder. Strict TDD. One commit per task group. GPU tests: `#[ignore]` + `%LOCALAPPDATA%/Temp` PATH trick for NVRTC.

## 1. CPU SDPA reference + RED
- [x] 1.1 Failing test: CPU SDPA reference over sequential blocks (attention matrix) matches hand-computed scores (cos-sim ≥ 0.9999)
- [x] 1.2 Failing test: CPU SDPA over SCATTERED multi-block table (block order 3,0,2,1) same logical attention as contiguous
- [x] 1.3 Failing test: CPU SDPA with causal mask (prefill path) applies upper-triangular restriction

## 2. Kernel — GREEN (engine-cuda)
- [x] 2.1 Implement `paged_attention.cu` with top-of-source port comment:
  `// Port of vLLM PagedAttention decode kernel (csrc/pos_encoding_kernels.cu, paged attention v1/v2, Apache-2.0)`
- [x] 2.2 Online-softmax single pass over block table in block order; GQA head-group mapping (n_head → n_head_kv)
- [x] 2.3 Zero intermediate allocations: no cudaMalloc in the launch path (static scratch only)
- [x] 2.4 Causal handling for prefill path
- [x] 2.5 Verify PASS on local GPU (multiple seq/gqa configs, 1..2048 tokens)

## 3. Parity + zero-alloc assertion
- [x] 3.1 Failing test (`#[ignore]`): GPU paged vs CPU SDPA across scattered multi-block sequences 1..2048 tokens, cos-sim ≥ 0.9999
- [x] 3.2 Failing test: launch path performs ZERO runtime cudaMalloc (instrumented allocator counter)
- [x] 3.3 Record numbers (cos-sim, max seq, budget); verify PASS — cos-sim=0.99999999999956 (cfg A, min over 1..2048), cos-sim=0.99999999999964 (cfg B, min over 1..1024); max seq=2048; zero runtime cudaMalloc (live allocs before=4 after=4); verified PASS on local RTX 3060.

## 4. Gate
- [x] 4.1 Full suite green (CPU + GPU `--ignored`, clippy -D warnings)
- [x] 4.2 Gate sealed: parity ≥ 0.9999 across scattered 1–2048, no cudaMalloc at runtime — REAL: cos-sim=0.99999999999956 (cfg A 1..2048) & 0.99999999999964 (cfg B 1..1024); zero runtime cudaMalloc (live allocs before=4 after=4); fmt --check clean; clippy --workspace --all-targets -- -D warnings clean (exit 0, 0 errors); CPU suite 80 passed 0 failed; GPU --ignored single-threaded 34 passed 0 failed.