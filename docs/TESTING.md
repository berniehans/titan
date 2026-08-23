# Titan — Testing

> Canonical requirements: [`openspec/specs/layer-streaming-engine/spec.md`](../openspec/specs/layer-streaming-engine/spec.md),
> constitution [`openspec/constitution.md`](../openspec/constitution.md) (TDD + per-phase gates).

This document describes how Titan is tested and how to reproduce the verification steps
exactly. It narrates the strategy; the constitution and phase gates are the authority.

## Test strategy

- **CPU suites run everywhere, including CI.** README curates this as **20 CPU suites
  green in CI**. The storage/reference dequant, parser error-paths, and layout logic are
  all covered without a GPU.
- **GPU integration tests are marked `#[ignore]` and run only locally** on a machine
  with a CUDA device (`cargo test --workspace -- --ignored`). These exercise pinned
  allocation, streams, events, device buffers, and the double-buffered pipeline against
  the real CUDA driver via cudarc (0.12.1, `cuda-12000`).
- Counts are re-measured at each gate; the workspace currently has 30 CPU test functions
  and 10 `#[ignore]`d GPU tests across the `engine-core`, `engine-io`, and `engine-cuda`
  crates. `engine-api` and `engine-kvcache` are placeholders. Keep the README counts in
  sync when suites change.

CI (`.github/workflows`, ubuntu-latest): `cargo fmt --check` → `cargo clippy -- -D warnings`
→ `cargo test` (CPU only). No GPU is available in CI, so GPU coverage is a local step.

## Fixture behaviour

- Tests that depend on the GGUF fixture **skip gracefully** when
  `testdata/Qwen3-0.6B-Q4_K_M.gguf` is absent — e.g. in CI, where the ~400 MB fixture is
  not checked in. They do not fail; they are reported as skipped.
- Locally, fetch the fixture in full with **`bash tools/download_fixture.sh`**
  ([`../tools/download_fixture.sh`](../tools/download_fixture.sh)). It is idempotent
  (exits 0 if a valid file is already present), pinned to the unsloth mirror, and
  verifies **size (396,705,472 bytes) + SHA256
  (`ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a`)** before it
  accepts any file (CHECKSUMS: [`../testdata/CHECKSUMS.md`](../testdata/CHECKSUMS.md)).
  A custom mirror can be supplied via `FIXTURE_URL`.

## Error-path coverage (engine-io)

The parser/loader fail loudly and without unbounded allocations on bad input
(13 error-path tests; proposal
[`openspec/changes/hardening-error-paths/proposal.md`](../openspec/changes/hardening-error-paths/proposal.md)).
Error variants in `GgufError` (`engine/engine-io/src/error.rs`):

| Scenario | Guard / variant |
|---|---|
| Malformed header | `InvalidMagic` (file does not start with `GGUF`), `UnsupportedVersion` (only v3 accepted) |
| Malformed metadata | `InvalidUtf8`, `InvalidValueType`, `InvalidTensorType`, `InvalidAlignment` |
| Truncation at header / metadata / tensor-info boundaries | `UnexpectedEof`, `Io` |
| **Allocation-bomb guards** — declared string/array/count length exceeds the safe bound | `MetadataTooLarge { what, len }` (bounded error, no OOM) |
| Tensor offsets past EOF / invalid shapes | `TensorOutOfBounds { name, offset, size, file_size }`, `InvalidTensorShape` |
| Loader layout sum > file size | `InvalidTensorShape` (clear error, no panic) |
| Non-contiguous layer tensors (malformed interleaving) | `InvalidTensorShape` (contiguity precondition in `LoadedLayout::from_reader`) |

`engine-core` and `engine-cuda` have their own typed errors (`EngineError`,
`CudaError`) with explicit allocation/size guards (e.g. `MetadataTooLarge`-style bounds
become `CudaError::InvalidSize`, `EngineError::InvalidLayerSize` for layers larger than
the pipeline's `max_layer_bytes`).

## Per-phase verification gates

Every phase is only "done" when its gate is green **and** evidenced here / in its spec
(constitution §3: per-phase gates; risky GPU gates require human sign-off before running
on GPU).

| Layer | Command / criterion |
|---|---|
| Build | `cargo build --workspace` |
| Format | `cargo fmt --check` |
| Lint | `cargo clippy --workspace -- -D warnings` |
| CPU tests | `cargo test --workspace` (runs everywhere incl. CI; fixture tests skip when absent) |
| GPU tests (local) | `cargo test --workspace -- --ignored` (requires CUDA device) |
| Benchmark comparison | run `engine-core/tests/pipeline_bench.rs` on reference hardware and update [`BENCHMARKS.md`](./BENCHMARKS.md) (see the "Rule" there) |
| Numerical parity (Phase 3) | GPU dequant vs CPU reference, < 0.01/elem, block-by-block |

## Exact reproduction (same as README usage)

```bash
# 1. Download the test fixture (Qwen3-0.6B Q4_K_M, ~400 MB; idempotent, SHA256-verified)
bash tools/download_fixture.sh

# 2. Build + lint
cd engine
cargo build --workspace
cargo clippy --workspace -- -D warnings

# 3. CPU tests (runs everywhere, incl. CI)
cargo test --workspace

# 4. GPU tests (require a local CUDA device; marked #[ignore])
cargo test --workspace -- --ignored
```

## Related

- Canonical spec: [`openspec/specs/layer-streaming-engine/spec.md`](../openspec/specs/layer-streaming-engine/spec.md)
- Constitution (TDD, gate sign-off): [`openspec/constitution.md`](../openspec/constitution.md)
- Benchmarks & methodology: [`BENCHMARKS.md`](./BENCHMARKS.md)
- Architecture: [`ARCHITECTURE.md`](./ARCHITECTURE.md)