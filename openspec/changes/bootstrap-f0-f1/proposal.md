# Change: Workspace bootstrap (Phase 0-1)

## Why
Engine starting point: build infrastructure, CI, and the I/O primitives (GGUF parser + pinned memory) everything else rests on. This is the smallest change that produces verifiable value.

## What Changes
- Create Cargo workspace with 5 empty crates (`engine-api`, `engine-core`, `engine-io`, `engine-cuda`, `engine-kvcache`) + rust-toolchain.toml + GitHub Actions CI.
- Implement GGUF v3 parser (metadata + tensor infos) in engine-io with tests against a fixture.
- Implement RAII pinned RAM allocation aligned to 4096 B via cudarc/FFI in engine-cuda.
- Load Qwen3-0.6B Q4_K_M fixture fully into pinned memory with GB/s metric.

## Non-goals
- No kernels, forward pass, streaming, or HTTP server (later changes).
- No formats other than GGUF.

## Impact
- **Affected specs:** layer-streaming-engine (requirement "Single weight load into pinned RAM")
- **Affected code:** new workspace under `engine/`, fixtures in `testdata/`
- **Gate:** `cargo test` green + fixture loaded <5 s + clippy clean

## Tasks (summary — details in tasks.md)
1. Workspace + toolchain + CI
2. Downloadable fixture with checksums
3. GGUF parser (TDD)
4. Pinned memory RAII (TDD)
5. Full loader with metrics
