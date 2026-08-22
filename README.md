# Titan

Rust + CUDA LLM inference engine for GGUF models whose weights **do not fit in VRAM**: tensors live in pinned RAM (NVMe → host once) and stream layer-by-layer to the GPU through a double-buffered pipeline.

## Current state — Phase 0-1 (bootstrap)

| Component | Status |
|---|---|
| Cargo workspace (5 crates) + stable toolchain + CI | ✅ |
| GGUF v3 parser (header, metadata KV, tensor infos, layer index) | ✅ TDD |
| Pinned RAM RAII (`cuMemAllocHost`/`cuMemFreeHost`, 4096 B aligned) | ✅ TDD, GPU |
| Single-pass NVMe → pinned loader with GB/s metric | ✅ (~400 MB in <1 s) |
| Per-layer streaming, CUDA double buffering, dequant kernels, KV cache | ⏳ next phases |

**Reference hardware:** RTX 3060 6 GB · target: dense 14B Q4_K_M (~8.5 GB in RAM) at ≈1.4 tok/s measured over PCIe x8.

## Architecture

```
engine/
├── engine-api        # public engine contracts
├── engine-core       # orchestration and generation loop
├── engine-io         # GGUF v3 parser + pinned-memory loader
├── engine-cuda       # CUDA FFI: pinned host RAII (next: streams, kernels)
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
