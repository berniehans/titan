You are implementing Task 3 (GGUF v3 parser) of OpenSpec "bootstrap-f0-f1" in crate **engine-io**. Repo root: C:/Users/niber/AppData/Local/hermes/workspace/motor-llm-rust-cuda. Work only inside engine/engine-io/.

Constitution rules: TDD (test-first, RED→GREEN→REFACTOR), no unwrap() outside tests, errors via thiserror (it's a lib crate), cargo clippy -- -D warnings clean, don't touch other crates.

## Fixture
The test fixture is the real Qwen3-0.6B-Q4_K_M.gguf at C:/Users/niber/AppData/Local/hermes/workspace/motor-llm-rust-cuda/testdata/Qwen3-0.6B-Q4_K_M.gguf (396,705,472 bytes). Tests MUST NOT hardcode machine-specific absolute paths into non-test code — expose the fixture path in tests via an env var override (e.g. env ENGINE_TESTDATA, default to a relative path) OR resolve relative to the crate source via canonical_path. Pick one clean approach. Tests may skip gracefully with a clear message if the fixture is not present, but on this machine it IS present so tests must actually run and pass.

## GGUF v3 format spec (AUTHORITATIVE, from llama.cpp gguf.h)

File layout:
1. magic: 4 bytes ASCII "GGUF"
2. version: uint32 (must be 3)
3. n_tensors: int64
4. n_kv: int64
5. For each KV: key (string), value_type (int32 enum gguf_type), then the value:
   - if value_type == GGUF_TYPE_ARRAY(9): read array_type (int32), array_count (uint64), then array_count elements of array_type.
   - else: read the scalar value of that type.
6. For each tensor (tensor info): name (string), n_dims (int32/uint32), dims[n_dims] (int64 each), tensor_type (int32 enum ggml_type), data_offset (uint64).
   NOTE: llama.cpp writes n_dims as int32 per recent code (the comment says uint32; some writers used int32 field width 4 bytes — read a uint32_le for n_dims). dims are int64_le. Data offset is uint64_le.
7. Tensor data blob follows (aligned to general.alignment or 32).

Strings: uint64 length, then that many UTF-8 bytes, no null terminator.

Int enums stored as int32. bool values as int8. All integers little-endian.

gguf_type enum (KV value types):
UINT8=0, INT8=1, UINT16=2, INT16=3, UINT32=4, INT32=5, FLOAT32=6, BOOL=7, STRING=8, ARRAY=9, UINT64=10, INT64=11, FLOAT64=12.

ggml_type enum (tensor types) — Qwen3 Q4_K_M uses these:
F32=0,F16=1,Q4_0=2,Q4_1=3,Q5_0=6,Q5_1=7,Q8_0=8,Q8_1=9,Q2_K=10,Q3_K=11,Q4_K=12,Q5_K=13,Q6_K=14,Q8_K=15,IQ2_XXS=16,IQ2_XS=17,IQ3_XXS=18,IQ1_S=19,IQ4_NL=20,IQ3_S=21,IQ2_S=22,IQ4_XS=23,I8=24,I16=25,I32=26,I64=27,F64=28,IQ1_M=29,BF16=30,TQ1_0=34,TQ2_0=35,MXFP4=39,NVFP4=40,Q1_0=41,Q2_0=42.

## TDD sequence (follow this exact order; write the failing test BEFORE implementation)

3.1 RED test: parse header of the fixture file → magic == "GGUF", version == 3. Assert n_tensors > 0 and n_kv > 0 too.
3.2 Implement: a reader that opens the file, reads the fixed header (magic, version, n_tensors, n_kv), plus generic KV metadata reader supporting all scalar types (u8, i8, u16, i16, u32, i32, f32, bool, string, u64, i64, f64) and arrays. Store metadata in a typed structure (e.g. a map of key → a tagged value enum). GREEN.
3.3 RED test: iterate ALL tensor infos from the real fixture → each has name, dims[], type, and a data_offset/total span; the sum of tensor byte spans must equal the tensor-data area; the FULL span of tensor data must fit within file size. Assert a known tensor exists (e.g. "token_embd.weight") and that total tensor count is sane (>100 for a 0.6B Q4_K_M).
3.4 Implement: the tensor-info reader producing a Vec<TensorInfo{name, dims, ggml_type, offset, size_bytes}>. Compute size_bytes from ggml_type block size and dims (use llama.cpp block sizes: Q4_K block=256, block_size=144 bytes; Q8_0: 32,34; F16: 1,2; F32:1,4; Q5_K:256,176; etc.). Validate against the real file so tensor sum + aligned header == file size.
3.5 RED test: layer pattern mapping — tensors match name patterns: "blk.N.*", "token_embd.weight", "output.weight". Build a helper classify_layer(name) → Option<int> layer index (parse blk.N.), and group by layer.
3.6 Implement the layer index: Map<layer, Vec<TensorInfo>> plus a flat list; expose funcs `layers()`, `tensors()`, `by_layer(idx)`.
3.7 VERIFY: `cargo test` in engine-io green (run via `cd engine && cargo test -p engine-io`), clippy clean.

## Design constraints
- Public API in lib.rs (re-export from a module). e.g. `GgufReader::open(path)`, `header()`, `metadata()`, `tensor_infos()`, `layer_index()`.
- Errors: define `GgufError` via thiserror. No unwrap outside `#[test]` (use expect/assert in tests).
- Keep it dependency-light. If a std-only approach works for binary parsing, use std; it may depend on the workspace cudarc already only declares on engine-cuda. engine-io currently has NO deps declared — only add what you actually use (, maybe nothing). Prefer std.
- clippy-clean: no hidden warnings, no dead code, docs on public API.
- This is a fresh lib; write modules under src/ (e.g. src/gguf.rs) or inline in lib.rs — choose clean structure. Expose only the documented public surface.
- Commit is NOT done by you.

## After implementation
- Confirm `cargo clippy -- -D warnings` clean on engine-io specifically.
- Confirm engine-io tests pass. Report the list of test names + actual pass output (paste the test result block into your reply) and the parsed header facts (magic, version, tensor count) as evidence.
- Do not modify other crates, CI, .gitignore, or openspec files.
- IMPORTANT: do not run agy's own verification while editing — run cargo after edits and quote the real output.