# Implementation Tasks: Phase 11 — Chunked Prefill & FlashAttention-2 GPU Kernel

## 1. Batched Quantized GEMM Kernels (sub-change 11.1)
- [x] 1.1 Implement `gemm_q4k_kernel`, `gemm_q6k_kernel`, and `gemm_q80_kernel` in `engine-cuda/kernels/gemm_quant.cu` for arbitrary batch size $M \ge 1$.
- [x] 1.2 Implement Rust RAII wrapper `BatchedQuantGEMV` / `BatchedGEMM` in `engine-cuda/src/batched_gemm.rs`.
- [x] 1.3 Create TDD parity test `engine-cuda/tests/gemm_batched_parity.rs` testing $M \in \{16, 64, 128, 256\}$ against CPU reference.
- Gate PASS: `cargo test -p engine-cuda --test gemm_batched_parity` PASS (`cos-sim >= 0.9999`).

## 2. FlashAttention-2 Causal GPU Kernel (sub-change 11.2)
- [x] 2.1 Implement `flash_attention_2_kernel` in `engine-cuda/kernels/flash_attention_2.cu` with online softmax scaling and paged KV pool reading.
- [x] 2.2 Implement Rust RAII wrapper `FlashAttention2` in `engine-cuda/src/flash_attention.rs`.
- [x] 2.3 Create TDD parity test `engine-cuda/tests/flash_attention_parity.rs` verifying causal attention against CPU reference.
- Gate PASS: `cargo test -p engine-cuda --test flash_attention_parity` PASS (`cos-sim >= 0.9999`).

## 3. ForwardDriver Chunked Prefill Integration (sub-change 11.3)
- [ ] 3.1 Implement batched RoPE and batched KV cache append in `engine-cuda`.
- [ ] 3.2 Implement `ForwardDriver::prefill_chunked` in `engine-core/src/forward_driver.rs`.
- [ ] 3.3 Create parity test `engine-core/tests/chunked_prefill_parity.rs` asserting bit-identical logits (`cos-sim >= 0.997`) against serial prefill across all prompt fixtures.
- Gate: `cargo test -p engine-core --test chunked_prefill_parity` PASS.

## 4. TTFT Speedup Benchmarks & Phase 11 Seal (sub-change 11.4)
- [ ] 4.1 Benchmark TTFT in `engine-server/tests/ttft_benchmark_gate.rs` across prompt lengths ($S \in \{16, 64, 128, 256, 512, 1024\}$).
- [ ] 4.2 Record measured speedups and TTFT latencies in `docs/BENCHMARKS.md`.
- [ ] 4.3 Verify full workspace test suite `cargo test --workspace` with 0 regressions.
- [ ] 4.4 Sync delta spec to main spec and archive change.
- Gate: Throughput / TTFT speedup verified, all tests green, Phase 11 sealed.
