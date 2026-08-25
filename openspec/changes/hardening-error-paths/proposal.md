# Change: Parser/loader error-path hardening tests

## Why
The current suite validates the happy path almost exclusively. Corrupt, truncated, and malformed GGUF inputs are untested — a streaming engine must fail loudly and without unbounded allocations on bad input.

## What Changes
- Add negative/error-path tests to `engine-io`: truncated file at header/metadata/tensor-info boundaries, bad magic, bad version (1/2), oversized declared string/array lengths (allocation-bomb guard), tensor offset past EOF.
- Add a loader-level test: file shorter than layout sum → clear error, no panic.
- No production code changes unless a test exposes a real bug (fix root cause if so).

## Non-goals
- No new formats, no GPU changes.

## Impact
- **Affected code:** `engine/engine-io/tests/*` (+ small fixes in src only if bugs surface)
- **Gate:** all new tests pass; existing suite stays green

## Tasks
- [x] 1. Malformed-header tests: bad magic, version 1/2, empty file → GgufError variants
- [x] 2. Truncation matrix: cut fixture/synthetic buffer at header end, mid-metadata, mid-tensor-infos → UnexpectedEof errors
- [x] 3. Allocation guards: declared string len > MAX_STRING_LEN and array len absurd → bounded error, no OOM
- [x] 4. Loader mismatch: layout sum > file size → InvalidTensorShape-style error
- [x] 5. Full suite green + clippy clean; fix any real bug found at root cause
