## 1. CUDA Q6_K Dequantization Kernel (sub-change 8.1)
- [x] 1.1 Implement raw CUDA kernel `dequant_q6k_kernel` in `engine-cuda/kernels/dequant_q6k.cu` (unpack 128B `ql`, 64B `qh`, 16B `scales`, 2B `d`).
- [x] 1.2 Add Rust wrapper `Q6KDequantizer` in `engine-cuda/src/dequant_q6k.rs`.
- [x] 1.3 Create TDD parity test `engine-cuda/tests/dequant_q6k_parity.rs` verifying bit-exact/rel-L2 parity (< 1e-4) against `engine_core::dequant_q6k_cpu`.
- Gate PASS: Single-block & multi-block Q6_K cos-sim = 1.000000, max diff = 0.000000e0 vs CPU reference.

## 2. Fused Q6_K GEMV Kernel (sub-change 8.2)
- [ ] 2.1 Implement fused `gemv_q6_k` CUDA kernel in `engine-cuda/src/kernels/gemv_q6k.cu` using warp shuffle reductions.
- [ ] 2.2 Extend `MultiFormatGEMV` in `engine-cuda/src/multiformat_gemv.rs` with `GgmlType::Q6_K` dispatch.
- [ ] 2.3 Create TDD parity test `engine-cuda/tests/gemv_q6k_parity.rs` verifying matrix-vector output against CPU reference GEMV.
- Gate: `cargo test -p engine-cuda --test gemv_q6k_parity` PASS.

## 3. Full GPU Forward Driver Layer Loop (sub-change 8.3)
- [ ] 3.1 Implement GPU embedding lookup for Q6_K token embeddings in `engine-cuda/src/norm_rope.rs` / `forward_driver.rs`.
- [ ] 3.2 Update `ForwardDriver::step_one` in `engine-core/src/forward_driver.rs` to route all Q6_K projections (`attn_v`, `ffn_down`) to GPU `MultiFormatGEMV`.
- [ ] 3.3 Verify zero CPU syncs during layer decode loop and assert cumulative drift parity against `logits_00.bin`.
- Gate: `cargo test -p engine-core --test driver_cumulative_drift` PASS with cos-sim >= 0.997.

## 4. Throughput Benchmarks & Seal (sub-change 8.4)
- [ ] 4.1 Benchmark multi-step autoregressive generation throughput in `engine-server/tests/real_throughput_gate.rs`.
- [ ] 4.2 Record measured tok/s and speedup in `docs/BENCHMARKS.md`.
- [ ] 4.3 Verify full workspace test suite `cargo test --workspace` with 0 regressions.
- Gate: Throughput >= 15.0 tok/s, all gates green, benchmarks sealed.
