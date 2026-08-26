# Tasks: bootstrap-f0-f1

> Execute with harness (agy gemini-3.7-flash-high via coder profile). Strict TDD. One commit per task.

## 1. Workspace
- [x] 1.1 Create `engine/Cargo.toml` workspace with members = the 5 crates; shared deps in workspace.dependencies (cudarc 0.12 cuda-12000, tokio, axum, anyhow, thiserror, tracing)
- [x] 1.2 Create empty crates with src/lib.rs + rust-toolchain.toml (stable)
- [x] 1.3 GitHub Actions CI: fmt --check, clippy -D warnings, test (GPU tests with #[ignore])
- [x] 1.4 Verify: `cargo build && cargo clippy -- -D warnings && cargo test` green

## 2. Fixture
- [x] 2.1 Script tools/download_fixture.sh: download Qwen3-0.6B Q4_K_M GGUF (~400 MB) to testdata/, idempotent, record SHA256 in testdata/CHECKSUMS.md
- [x] 2.2 Verify download and checksum

## 3. GGUF parser (engine-io)
- [x] 3.1 Failing test: parse fixture header → magic "GGUF", version 3
- [x] 3.2 Implement header reading + metadata KV (types u8..f64, string, array)
- [x] 3.3 Failing test: tensor infos → name/dims/type/offset of all fixture tensors
- [x] 3.4 Implement tensor infos; validate against reference gguf-dump
- [x] 3.5 Failing test: map tensors by name pattern (blk.N.*, token_embd, output)
- [x] 3.6 Implement layer indexing for subsequent streaming load
- [x] 3.7 Verify: cargo test -p engine-io green

## 4. Pinned memory (engine-cuda)
- [x] 4.1 Failing test (#[ignore] without GPU): allocate 256 MB pinned, write pattern, read back equal, Drop frees (debug counter)
- [x] 4.2 Implement RAII wrapper cudaMallocHost/cudaFreeHost aligned to 4096 B with // SAFETY:
- [x] 4.3 Verify test PASS on local GPU

## 5. Full loader
- [x] 5.1 Failing test: load full fixture → byte sum == file size, tensors in contiguous per-layer regions
- [x] 5.2 Implement loader: read GGUF once, write tensors to pinned memory, log GB/s
- [x] 5.3 Gate F0-F1: fixture loaded <5 s, clippy clean, all tests green
