# Dense Decode Performance Specification

## Purpose

Define a reproducible, correctness-preserving path for bringing Titan's dense-model decode throughput to the level of CUDA-enabled `llama.cpp` on the reference RTX 3060 Laptop GPU.

## ADDED Requirements

### Requirement: Reproducible Decode Benchmark

The benchmark harness SHALL record model identity, hardware/runtime identity, exact command, prompt/cache condition, generated-token count, prefill throughput, decode throughput, and raw timing data for every measured model.

#### Scenario: Repeated five-model comparison

- **WHEN** the five-model benchmark is run with the pinned environment and at least three repetitions
- **THEN** it produces machine-readable results and raw logs that can be independently aggregated without hand transcription.

### Requirement: Per-Stage Decode Attribution

Titan SHALL expose enough timing and launch information to attribute decode time to major execution stages and synchronization/copy overhead.

#### Scenario: Llama 3.2 3B diagnosis

- **WHEN** the 3B decode path is profiled
- **THEN** the report identifies the dominant measured contributors and records the optimization hypothesis selected from that evidence.

### Requirement: Reference-Preserving Kernel Optimization

Every optimized quantized or fused kernel SHALL retain numerical parity with an independent CPU/reference implementation and SHALL preserve a tested fallback path where architecture compatibility requires it.

#### Scenario: Optimized Q4_K/Q6_K path

- **WHEN** an optimized kernel is compared against its independent CPU golden across representative model dimensions
- **THEN** all declared error thresholds pass, CUDA Graph execution remains valid, and no existing parity regression appears.

### Requirement: Dense Decode Parity Gate

The first performance milestone SHALL require Titan decode throughput to reach at least 0.95x CUDA-enabled `llama.cpp` for every model in the five-model benchmark and at least 0.95x in aggregate.

#### Scenario: First-instance release gate

- **WHEN** the final repeated benchmark is executed on the reference RTX 3060 setup
- **THEN** every model ratio is >= 0.95x, aggregate ratio is >= 0.95x, and no model is more than 5% below the frozen Titan baseline.

### Requirement: Quality Gates

The change SHALL pass formatting, strict Clippy, workspace compilation, workspace tests, targeted GPU tests, and the relevant benchmark/correctness gates before it is considered complete.

#### Scenario: Pre-commit validation

- **WHEN** all optimization tasks are complete
- **THEN** `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo check --workspace`, `cargo test --workspace`, and the ignored GPU parity suite pass.
