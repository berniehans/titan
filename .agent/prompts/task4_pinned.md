You are implementing Task 4 (pinned host memory RAII) of OpenSpec "bootstrap-f0-f1" in crate **engine-cuda**. Repo root: C:/Users/niber/AppData/Local/hermes/workspace/motor-llm-rust-cuda. Work only inside engine/engine-cuda/.

Machine has an NVIDIA RTX 3060 Laptop GPU (present in this environment) and the CUDA 12 runtime DLL, but NO CUDA toolkit (nvcc) — that's fine, cudarc loads the driver dynamically. GPU tests should PASS here, but mark them with #[ignore] so CI (no GPU) skips them.

## Constitution rules (MANDATORY)
- TDD: write the failing test FIRST, confirm it fails, then implement (RED→GREEN).
- unsafe ONLY in engine-cuda/engine-io and ALWAYS with a `// SAFETY:` comment explaining the contract.
- thiserror for lib error types. NO unwrap() outside tests (use expect/assert in tests).
- cargo clippy -- -D warnings clean.

## cudarc 0.12.1 FFI API (AUTHORITATIVE — inspect /c/Users/niber/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cudarc-0.12.1 to confirm details)
The driver exposes a raw sys Lib handle via `sys::lib()` (a static global). Use its raw functions in an unsafe block. Relevant FFI (from src/driver/sys/sys_12000.rs — the 12000 build):
- `pub unsafe fn cuMemAllocHost_v2(&self, pp: *mut *mut c_void, bytesize: usize) -> CUresult`
- `pub unsafe fn cuMemFreeHost(&self, p: *mut c_void) -> CUresult`
- `pub unsafe fn cuGetResultString / errors via CUresult`

Import path: `use cudarc::driver::sys;` then `let lib = unsafe { sys::lib() };` — but careful: cuMemAllocHost_v2/free must run bound to a valid CUDA context. The simplest robust way is to create a [`CudaDevice`] via `CudaDevice::new(0)` which sets up the primary context automatically, then call the raw sys lib functions inside an unsafe block to allocate/free HOST pinned memory. The device stays alive for the lifetime of the allocation to keep the driver + context loaded.

Look at cudarc's own examples (in the crate: examples/ folder) and src/driver/safe/alloc.rs + core.rs for the exact public name of the type (`CudaDevice::new(0)` returns Result<Arc<CudaDevice>, ...>). Prefer calling host alloc/free via `cudarc::driver::sys::lib().cuMemAllocHost_v2(&pp, size)` and verify the CUresult (0 == CUDA_SUCCESS). Use the CUDA_SUCCESS constant correctly for the build (from cudarc sys enum, e.g. sys::CUresult::CUDA_SUCCESS or compare == 0).

## The deliverable — a RAII pinned memory wrapper

Create src/pinned_host.rs (and wire it in lib.rs, re-export `PinnedHost`). Contract:

- `PinnedHost` RAII struct:
  - `pub fn alloc(size_bytes: usize) -> Result<PinnedHost, CudaError>` — allocates HOST pinned memory of at least `size_bytes`, ALIGNED to 4096 bytes.
  - Since cudaMallocHost doesn't guarantee 4096 alignment on every driver, allocate `size + 4096` bytes and compute an aligned usable window at the low end: scan the returned block for the first 4096-aligned offset within `[0,4096]`, return a pointer `aligned + 4096*0` offset such that data starts 4096-aligned. Keep the original base pointer internally so `Drop` calls cuMemFreeHost only on the RAW base pointer (never on the aligned offset).
  - Holds the base pointer + actual usable span metadata. On `Drop` (destructor), calls cuMemFreeHost(base) exactly once — RAII. Non-copyable (disable clone). Provide `Clone::disable`-style or just don't mark Clone-able.
  - `bytes() -> usize` returns usable allocation size.
  - `as_ptr() -> *mut u8` returns the aligned usable pointer (host pointer).
  - Optional `write_pattern(seed)` / read-back helpers for tests, or just let the test mem::write directly through as_ptr.
- Error type `CudaError` via thiserror, with cases: `AllocFailed(fn_name)` and `FreeFailed(...)`. On alloc failure return error, do NOT leave dangling.
- Must NOT crash the process if CUDA driver is missing at runtime — if `CudaDevice::new(0)` fails (no driver), propagate as a typed error.
- All unsafe in this module with `// SAFETY:` comment per constitution.

## TDD sequence (strict order — write failing test BEFORE impl)

4.1 RED test: `#[ignore] fn test_pinned_256mb_alloc_write_read_free()`:
   - alloc 256 MiB pinned via PinnedHost::alloc(256*MiB, 4096)
   - assert bytes() >= 256 MiB
   - write a pattern (e.g. fill with value computed from index: `b[i] = (i*7+13)&0xFF`) through as_ptr
   - read back and assert equal
   - ensure Drop frees: use a debug counter — e.g. keep a static `is_allocated` flag or a counter of live allocations; simplest: after the test scope ends (Drop), you may expose a test-only counter (e.g. an internal `AtomicUInt` of OutstandingAllocs) and assert it returns to the prior value after the wrapper goes out of scope. Implement a debug counter inside PinnedHost (static), inc on alloc, dec on drop, and the test asserts it dropped.
   - #[ignore] so CI without GPU skips it.
4.2 Implement PinnedHost per contract. GREEN.
4.3 VERIFY: run on THIS machine (has GPU) — test PASS. With #[ignore], `cargo test` on CI/low machines skips; here it runs.

## engine-cuda current state
The crate currently has src/lib.rs with `pub fn version()`. Its Cargo.toml declares `[dependencies] cudarc.workspace = true`. You will ADD the pinned_host module + tests. Do not change other crates. You may add src/pinned_host.rs and update lib.rs re-exports. Do NOT remove the existing version fn.

## Verification requirements
- Run `cd engine && cargo test -p engine-cuda` — quote the REAL output (which tests ran, passed).
- Run `cargo clippy -p engine-cuda --all-targets -- -D warnings` — must be clean (0 warnings).
- Ensure the GPU-containing environment will execute the pinned test (not crash with driver-missing), since this machine has the GPU.
- Do NOT commit. Just implement + run + report.