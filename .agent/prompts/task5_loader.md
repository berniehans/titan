You are implementing Task 5 (full loader to pinned memory + GB/s metric) of OpenSpec "bootstrap-f0-f1" in crate **engine-io**. Repo root: C:/Users/niber/AppData/Local/hermes/workspace/motor-llm-rust-cuda. Work only inside engine/engine-io/ plus its Cargo.toml.

This machine has an NVIDIA RTX 3060 Laptop GPU (present) so CUDA-backed tests should PASS here, but mark them #[ignore] so CI without GPU skips them (same pattern as engine-cuda task 4).

## Constitution rules (MANDATORY)
- TDD: write the failing test FIRST (RED), verify it fails, then implement (GREEN).
- thiserror for lib errors. NO unwrap() outside tests.
- cargo clippy -- -D warnings clean.
- The unsafe code lives in engine-cuda's PinnedHost (already written, verified). The loader in engine-io should use PinnedHost's SAFE API — it should NOT need new unsafe.

## Existing building blocks (AUTHORITATIVE — read these files)
1. **engine-cuda** exposes `PinnedHost` (crate engine-cuda, src/lib.rs re-exports it):
   - `PinnedHost::alloc(size_bytes: usize) -> Result<PinnedHost, CudaError>` — 4096-aligned pinned host memory via cudaMallocHost.
   - `PinnedHost::bytes() -> usize`, `as_ptr() -> *mut u8`, `as_slice() -> &[u8]`, `as_mut_slice() -> &mut [u8]`, `live_allocations() -> usize`.
   - CudaError via thiserror.
2. **engine-io** parser (`GgufReader`, src/reader.rs) — read it. It exposes:
   - `GgufReader::open<P: AsRef<Path>>(path) -> Result<Self, GgufError>`
   - `header() -> &GgufHeader` where GgufHeader has `magic, version, tensor_count, metadata_kv_count`.
   - `metadata() -> &HashMap<String, GgufValue>`
   - `tensor_infos() -> &[TensorInfo]` where TensorInfo has `name, dims, ggml_type, offset, size_bytes`.
   - `layer_index() -> &LayerIndex` exposing `layers() -> Vec<usize>`, `by_layer(idx) -> Option<&[TensorInfo]>`, `non_layer_tensors()`, `tensors()`.
   - `tensor_data_offset() -> u64`, `alignment() -> u64`, `get_tensor(name) -> Option<&TensorInfo>`.
   IMPORTANT: GgufReader::open reads only the METADATA (header + KV + tensor infos). It does NOT read the tensor data blob bytes. The File handle used internally is closed after open.

## Task deliverable — a loader/layout struct + a GPU pinned-load function

Design TWO parts:

### Part A (CPU-only, no CUDA): `LoadedLayout` (src/loader.rs)
Pure layout/accounting computed from a GgufReader — works WITHOUT GPU, deterministic, unit-testable. Contract:
- `LoadedLayout::from_reader(reader: &GgufReader) -> Result<Self, GgufError>`:
  - Computes `data_blob_size = file data area size` = sum of tensor spans ENDING at the last tensor (i.e. max over tensors of `offset + size_bytes`). This is the total pinned bytes to allocate. Note this equals `file_size - tensor_data_offset` for a well-formed GGUF.
  - Builds a per-tensor view that maps each tensor to the (layer, offset_into_blob, size) so consumers can place data contiguously.
- Public accessors:
  - `total_size_bytes() -> u64` (the pinned blob size = sum of tensor spans max end).
  - `tensor_span(name) -> Option<(u64 offset_into_blob, u64 size)>`.
  - `layer_spans(layer) -> Option<&[(u64 offset_into_blob, u64 size)]>` (contiguous regions per layer).
  - keep it minimal and clean.
- No unsafe, no CUDA, no file I/O beyond the reader's data.

### Part B (GPU-backed): `load_fixture_to_pinned` in src/loader.rs
`pub fn load_to_pinned(reader: &GgufReader, path: &Path) -> Result<LoadedPinned, GgufError>` where:
- Reads the GGUF tensor data blob from disk ONCE (not per-tensor): open the file (or reuse GgufReader), seek to `reader.tensor_data_offset()`, read exactly `data_blob_size` bytes into a temp Vec.
- Allocate ONE `PinnedHost` of `data_blob_size` bytes (via engine_cuda::PinnedHost::alloc).
- Copy the blob bytes into the pinned buffer (via as_mut_slice().copy_from_slice).
- Return a `LoadedPinned` struct that owns the PinnedHost plus the layout:
  - `pub struct LoadedPinned { host: PinnedHost, layout: LoadedLayout }`
  - Accessors: `total_size_bytes()`, `as_slice() -> &[u8]` (borrows the pinned buffer), `tensor(name) -> Option<&[u8]>` (returns the slice of that tensor's bytes in the pinned buffer), `layer(layer) -> Option<&[u8]>` (contiguous bytes of that layer).
- METRIC: measure elapsed time of the disk read + pinned copy using std::time::Instant; compute GB/s = bytes / seconds; log via `tracing::info!` with fields `bytes`, `seconds`, `gb_per_second`. DO use tracing (the crate is already a workspace dependency; add `tracing.workspace = true` to engine-io Cargo.toml).
- Errors propagate via GgufError (add variants if needed, e.g. an Io passthrough — GgufError already has Io(#[from] std::io::Error) and can carry CudaError? Add `#[error("CUDA: {0}")] Cuda(#[from] engine_cuda::CudaError)` — but watch for circular dep: engine-cuda does NOT depend on engine-io (one-way OK). engine-io CAN depend on engine-cuda.

## Dependencies (engine-io/Cargo.toml)
Add:
```toml
engine-cuda = { path = "../engine-cuda" }
tracing.workspace = true
thiserror.workspace = true
```
(move thiserror from `"2.0"` to `.workspace = true` for consistency). NOTE: this makes engine-io depend on engine-cuda. Verify no circular dep.

## TDD sequence (follow strictly — failing test FIRST)

**RED 5.1 test** (`tests/loader_synthetic.rs`, CPU-only, NOT #[ignore] — runs on any machine):
- Build a tiny synthetic GGUF in-memory is complex; instead test via the REAL fixture accounting without needing the file: Actually the loader needs the reader which needs the file. Approach: write ONE test that opens the real fixture and asserts layout accounting:
  - `let reader = GgufReader::open(&fixture)?;`
  - `let layout = LoadedLayout::from_reader(&reader)?;`
  - assert `layout.total_size_bytes() > 0`.
  - assert the accounting is consistent: sum over all tensors of size_bytes == layout.total_size_bytes() (contiguous, no gaps in data area for this fixture — verify with a real computed check: for the fixture, tensor spans are contiguous so total_size_bytes == sum of all tensor size_bytes, AND == file_size - tensor_data_offset).
  - **KEY assertion**: `layout.total_size_bytes() + reader.tensor_data_offset() == file_size` (this is "sum bytes == file size" — the loaded data + metadata == whole file). Use std::fs::metadata for file_size.
  Make this test pass using ONLY Part A (no PinnedHost, no GPU). This is the 5.1 gate → GREEN after Part A.

**RED 5.2 test** (`tests/loader_pinned.rs`, #[ignore] — GPU/CI-skip):
- `#[ignore] fn test_load_fixture_to_pinned()`:
  - open reader, `load_to_pinned(&reader, &fixture)`.
  - assert `loaded.total_size_bytes() == file_size - tensor_data_offset`.
  - assert `loaded.as_slice().len() == total_size_bytes`.
  - assert `loaded.tensor("token_embd.weight")` returns Some slice whose `.len() == token_embd.size_bytes`.
  - assert `loaded.tensor("output_norm.weight")` returns Some slice with correct length.
  - assert `loaded.layer(0)` returns Some bytes with len == sum of that layer's tensor sizes.
  - (this exercises GB/s log path via tracing, though metric is just log — optionally expose `last_gb_per_second()` for assert > 0).
- GREEN after Part B. Run with `cargo test -p engine-io -- --include-ignored` on this GPU machine.

## Verification you MUST run and quote real output
- `cd engine && cargo build` (workspace)
- `cd engine && cargo test -p engine-io` (CPU tests — 5.1 layout must pass without GPU)
- `cd engine && cargo test -p engine-io -- --include-ignored` (GPU pinned loader test passes here)
- `cd engine && cargo clippy -p engine-io --all-targets -- -D warnings` clean.
- Also confirm engine-cuda tests still pass (no regressions).

Wire modules: add `pub mod loader;` and re-export `LoadedLayout, LoadedPinned` in src/lib.rs. Do NOT break existing API. Do NOT modify other crates. Do NOT commit — just implement + run + report with real output.