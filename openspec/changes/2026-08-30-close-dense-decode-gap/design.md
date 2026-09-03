# Design: Close the Dense Decode Performance Gap

## Baseline and Measurement Contract

The benchmark uses the existing five-model head-to-head harness on an RTX 3060 Laptop GPU with the same GGUF files, CUDA-enabled `llama.cpp`, greedy decoding, batch size one, two prompts, and a fixed generated-token count. Each run records:

- GPU, driver, runtime/NVRTC, Rust and `llama.cpp` build identity.
- Model file SHA-256 and quantization type.
- Prefill and decode throughput separately.
- Cold and warm prompt-cache conditions separately.
- Titan per-stage timings and synchronization points.
- `llama.cpp` timings from its server response/log.

The canonical metric is decode tok/s. Prefill and total request latency are secondary metrics and must not be mixed into the decode gate.

## Optimization Loop

Each optimization iteration follows this order:

1. Run the baseline benchmark and a targeted correctness gate.
2. Capture stage timings and GPU metrics for the same model/configuration.
3. Select one bottleneck hypothesis, with a predicted measurable effect.
4. Implement one focused change behind an explicit path or configuration when practical.
5. Run the narrow correctness/parity test first.
6. Run the targeted benchmark and compare confidence intervals/repeated runs.
7. Keep the change only if it improves the target without violating parity, VRAM, or regression gates.

No optimization is accepted from a single noisy run. The target model set is Llama 3.2 3B first, then Llama 3.2 1B and DeepSeek 1.5B, because those currently have the largest deficits.

## Workstream A: Benchmark and Profiling Truth

- Add repeated-run support and machine-readable JSON/CSV output to the benchmark harness.
- Separate cold-start, warm-cache, prefill, decode, and end-to-end measurements.
- Add per-layer/per-kernel timing aggregation for Titan.
- Capture Nsight Compute/Systems evidence when available; otherwise retain CUDA event timings and launch counts.
- Record model hashes and exact command lines in artifacts.

## Workstream B: Decode Bottleneck Optimization

Investigate in measured order:

1. Q4_K/Q6_K GEMV/GEMM occupancy, tile shape, split-K policy, and register/shared-memory pressure.
2. 128-bit load alignment/coalescing and redundant weight/activation traffic.
3. Gate/up/SwiGLU/down fusion and intermediate materialization.
4. QKV, output projection, and LM-head launch shape and memory residency.
5. CUDA Graph parameter updates, host synchronization, and graph replay boundaries.
6. Any residual per-token allocations, copies, or stream barriers.

The MMA path is not assumed to be faster merely because it uses Tensor Cores; it must beat the current DP4A path in the actual model shapes and remain within the parity tolerance.

## Workstream C: Reference-Backed Kernel Correctness

Every kernel change must have:

- An independent CPU formula/reference test.
- A GPU parity test covering the relevant dimensions and edge cases.
- A pinned upstream traceability comment (repository, commit, file/function) when arithmetic or layout is ported from `llama.cpp` or vLLM.
- A regression artifact for any previously fixed issue, including SwiGLU.

## Workstream D: Release Evidence

The change is gated by the existing constitution rules and the new dense-decode spec. Documentation records fresh incomplete checkpoints when useful, but release claims and archival status remain blocked until the complete gate passes. Historical numbers remain labelled historical.

## Rollback

Keep each optimization as a reviewable atomic commit/change unit. If a targeted benchmark regresses or parity fails, disable/revert only that optimization path and retain the measurement artifact explaining why it was rejected.
