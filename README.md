# Titan

Rust + CUDA LLM inference engine for GGUF models whose weights **do not fit in VRAM**: tensors live in pinned RAM (NVMe → host once) and stream layer-by-layer to the GPU through a double-buffered pipeline.

## Current state — Phase 0-2 done, Phase 3 in progress

| Component | Status |
|---|---|
| Cargo workspace (5 crates) + stable toolchain + CI | ✅ |
| GGUF v3 parser (header, metadata KV, tensor infos, layer index) | ✅ TDD |
| Error-path hardening: malformed headers, truncation matrix, allocation-bomb guards (`MetadataTooLarge`) | ✅ 13 tests |
| Pinned RAM RAII (`cuMemAllocHost`/`cuMemFreeHost`, 4096 B aligned) | ✅ TDD, GPU |
| Single-pass NVMe → pinned loader with GB/s metric | ✅ (~400 MB in <1 s) |
| CUDA stream / event / VRAM `DeviceBuffer` RAII primitives | ✅ GPU-tested |
| **Double-buffered pipeline** (ping-pong slots, event-gated compute, no CPU busy-wait) | ✅ 8-layer ordering test |
| Benchmark: pipelined vs sequential | ✅ 10.43 ms < 10.77 ms on RTX 3060 |
| Q4_K_M layout + CPU reference dequantizer | ✅ TDD |
| GPU dequant kernel + parity gate (<0.01/elem, measured bit-exact 0.0) | ✅ done; Nsight overlap trace ⏳ pending |
| KV cache, SSE server, batching | ⏳ later phases |

**Test suite:** 20 CPU suites green in CI; 9+ GPU integration tests run locally (`#[ignore]`).

**Reference hardware:** RTX 3060 6 GB · target: dense 14B Q4_K_M (~8.5 GB in RAM) at ≈1.4 tok/s measured over PCIe x8.

## Architecture

```
engine/
├── engine-api        # public engine contracts
├── engine-core       # orchestration and generation loop
├── engine-io         # GGUF v3 parser + pinned-memory loader (error-path hardened)
├── engine-cuda       # CUDA FFI: pinned host, streams, events, VRAM buffers RAII
└── engine-kvcache    # KV cache for attention
```

Core principle ([spec](openspec/specs/layer-streaming-engine/spec.md)): weights are read from disk **once** at startup; there is no `read()` during generation. The H2D copy of layer N+1 overlaps layer N's compute via two event-synchronized streams. Q4_K_M weights are dequantized inside the GPU kernels, without materializing FP16 copies in VRAM.

## Usage

```bash
# 1. Download the test fixture (Qwen3-0.6B Q4_K_M, ~400 MB, idempotent, SHA256-verified)
bash tools/download_fixture.sh

# 2. Build + lint
cd engine
cargo build --workspace
cargo clippy --workspace -- -D warnings

# 3. CPU tests
cargo test --workspace

# 4. GPU tests (require a local CUDA device; marked #[ignore])
cargo test --workspace -- --ignored
```

Tests that depend on the GGUF fixture skip automatically when the file is absent (e.g. in CI); locally they run in full.

## CI

GitHub Actions: `cargo fmt --check`, `clippy -D warnings`, `cargo test` (CPU). GPU tests run on local hardware.

## Development

Spec-driven with [OpenSpec](openspec/constitution.md): each phase is a change under `openspec/changes/` with a proposal, tasks, and a verifiable gate before marking it done.
