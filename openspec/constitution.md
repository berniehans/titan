# Constitution — LLM Inference Engine in Rust

Immutable project rules. Every spec and change references this document. Any change here requires Bernie's explicit approval.

## 1. Purpose
Local LLM inference engine in Rust for constrained hardware (RTX 3060 Laptop, 6 GB VRAM, PCIe 4.0 x8). Priority: verifiable correctness > throughput > features. The high-value use case is MoE expert-streaming; the baseline is a resident dense model.

## 2. Stack and conventions (non-negotiable)
- **Rust stable** with `rust-toolchain.toml`. Auxiliary Python repos (tools/): UV required.
- Cargo workspace: `engine-api`, `engine-core`, `engine-io`, `engine-cuda`, `engine-kvcache`. No circular dependencies between crates.
- `cargo clippy -- -D warnings` clean on every commit.
- Weight format: GGUF only for MVP.
- Errors with `thiserror` in libs, `anyhow` in bins. No `unwrap()` outside tests.
- `unsafe` only in `engine-cuda`/`engine-io`, with a mandatory `// SAFETY:` comment and a test that exercises it.

## 3. Development process
- **TDD**: test before implementation on every task. Numerical parity against reference (llama.cpp) for every kernel.
- **Per-phase gates**: each phase has a measurable criterion; do NOT advance without a green gate. Risky gates (new kernels, CUDA pipeline changes) require human sign-off before running on GPU.
- **Codegen**: implementation delegated to agy (gemini-3.7-flash-high) via harness; independent review (a model different from the writer) before merge. The orchestrator never writes code directly.
- **Honest verification**: "done" = tests run + output read. Reporting success without evidence is forbidden.
- **Git**: frequent commits per task. NEVER `git push` without Bernie's explicit order.

## 4. Hardware constraints (fixed assumptions)
- Usable VRAM: ~5.2 GB of 6 GB. Budget: buffers ~0.9 GB, activations/driver ~1.3 GB, remainder KV-cache.
- Bus: PCIe 4.0 x8 ≈ 12 GB/s effective.
- Weights load from NVMe to pinned RAM ONCE only; streaming is ALWAYS RAM→VRAM.
- Every throughput estimate is validated with a real benchmark before being used in specs.

## 5. Spec quality
- Requirements in SHALL format with concrete WHEN/THEN scenarios.
- Every numeric spec cites its source (verified own calculation or measured benchmark, never an unlabeled estimate).
- Explicit non-goals in every change to prevent scope creep.
