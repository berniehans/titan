# Proposal: Close the Dense Decode Performance Gap

## Why

Titan's current controlled benchmark remains behind CUDA-enabled `llama.cpp` on three of five models, and does not yet satisfy the release gate. The latest reproduced ratios (three repetitions per model, averaged across cold/warm report statistics) are:

- Qwen 2.5 1.5B: 1.009x
- Llama 3.2 1B: 0.896x
- Llama 3.2 3B: 0.735x
- DeepSeek-R1-Distill 1.5B: 0.864x
- Qwen3 0.6B: 1.057x

The fresh artifact is `local-artifacts/benchmarks/rerun-20260901-085229.json`, with raw output in the matching `.log` file. The test completed successfully, but the performance gate remains open.

The present implementation has already introduced vectorized loads, fused FFN paths, CUDA Graph replay, and an Ampere MMA path, but the measured gap shows that the critical decode path is not yet competitive for larger dense models. We need an evidence-driven optimization change rather than another broad kernel rewrite.

## What Changes

- Establish a reproducible, apples-to-apples performance gate with per-model and per-stage measurements.
- Profile Titan and `llama.cpp` on the same RTX 3060 workload to identify the dominant gap before changing kernels.
- Optimize the dominant decode bottlenecks in descending measured impact: quantized GEMV/GEMM, FFN fusion, memory traffic, graph replay/dispatch, and synchronization.
- Port proven arithmetic and tiling patterns from pinned `llama.cpp`/vLLM references where appropriate, with source commit/hash traceability and independent CPU goldens.
- Preserve a correct DP4A fallback and validate every optimized kernel against the existing CPU/reference gates.
- Rebaseline benchmark documentation only from fresh measured runs.

## First-Instance Success Criteria

The first instance is complete only when all criteria below are met on the reference RTX 3060 Laptop setup:

1. Titan reaches at least 0.95x of `llama.cpp` decode throughput for each of the five benchmark models.
2. Aggregate Titan/`llama.cpp` throughput ratio is at least 0.95x.
3. No model regresses by more than 5% from the current baseline.
4. All CPU, GPU, correctness, formatting, Clippy, and workspace test gates pass.
5. The benchmark log, profiler evidence, environment, model hashes, and command are archived as reproducible artifacts.

A later change may target sustained parity above 1.00x; this change does not claim that goal prematurely.

## Non-Goals

- No MoE expert streaming or Phase 7 work.
- No new quantization format beyond the currently supported GGUF paths.
- No redesign of HTTP, CLI, KV-cache, or public server APIs unless profiling proves an API boundary is the bottleneck.
- No removal of numerical parity gates for performance reasons.
- No comparison against undocumented or differently configured engines.

## Impact

- `engine-cuda`: kernels, launch configuration, profiling hooks, and reference traceability.
- `engine-core`: decode scheduling and synchronization only where measured necessary.
- `engine-server/tests`: benchmark harness, repeatability, and regression gates.
- `docs/BENCHMARKS.md`: current measured results only.
- `openspec/specs`: performance requirements and scenarios will be updated only after the corresponding gates are measured.
