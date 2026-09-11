# Titan — Testing

> Canonical requirements: [`openspec/specs/layer-streaming-engine/spec.md`](../openspec/specs/layer-streaming-engine/spec.md),
> constitution [`openspec/constitution.md`](../openspec/constitution.md) (TDD + per-phase gates),
> Stable Functional MVP contract [`docs/MVP.md`](MVP.md),
> Workspace state [`docs/WORKSPACE_STATE.md`](WORKSPACE_STATE.md).

This document describes how Titan is tested and how to reproduce the verification steps
exactly. It narrates the strategy; the constitution, OpenSpec specifications, and phase gates are the authority.

---

## 🛑 Current Status & Testing Boundary

| Dimension | Current Value | Authority / Artifact |
| :--- | :--- | :--- |
| **Functional MVP Status** | **`mvp_blocked_evidence`** (NOT accepted) | [`local-artifacts/reviews/mvp-decision-20260911T162326Z.json`](MVP.md) |
| **Strict Production Release Gate** | **`not_accepted`** / **`rejected`** | `local-artifacts/reviews/p5-release-gate-20260912T130000Z.json` |
| **Release Approval** | **`not_granted`** | [`docs/RELEASE_ENVELOPE_V1.json`](RELEASE_ENVELOPE_V1.json) |
| **Production Promotion** | **`promotion_authorized = false`** | No candidate promoted to default |
| **Default Runtime Dispatch** | **`production_dispatch_changed = false`** | Verified default F32 resident baseline |
| **Release Candidate Generation** | **`rc_generated = false`** | No release candidate generated |
| **MVP Gate Unit Suite** | **`25 passed, 0 failed`** (100%) | `uv run --project tools --no-sync pytest tools/test_mvp_gate.py` |
| **Full Python Tooling Suite** | **`88 passed, 0 failed; 8 subtests passed`** (100%) | `uv run --project tools --no-sync pytest tools` |
| **OpenSpec Live State** | **`40 passed, 0 failed`** (40 items) | `openspec validate --all` |

> [!IMPORTANT]
> **Functional MVP vs. Strict Production Gate:**
> * A throughput ratio of $\ge 0.95\times$ compared to `llama.cpp` is **NOT** an acceptance criterion for the Stable Functional MVP. The $0.95\times$ comparison remains a separate strict production-performance diagnostic.
> * In the functional MVP evaluation, throughput ratios (aggregate median `0.5286115177544821`) and baseline regressions are advisory warnings (`ratio_advisory: advisory_warning`, `regression_advisory: failed`).
> * Conversely, **generation correctness, operational E2E verification, and clean benchmark execution safety remain strictly blocking**.
>
> **Exact MVP Blockers:**
> 1. **`build_binding_unverified`:** The F8 correctness execution (5/5 models verified) does not record the Titan binary or source identity matching the clean benchmark's `engine_identity.titan` build hash (`sha256:bc92211356bb0eb87ddd59e252d9d4497f8669d7257ac0133459040ce4eb131c`).
> 2. **`model_hash_missing`:** The clean benchmark artifact (`local-artifacts/benchmarks/p5-final-clean-20260911T151844Z.json`, 30 rows) records model names but lacks per-result GGUF weight file hashes required for exact binding to F8.
>
> **Evidence State Classification:**
> * **Verified:** F8.R1.P generation correctness (5 models, F32, batch 1, resident KV, Graph=false, candidate selectors absent); operational E2E (Resident JSON, Streaming SSE with N-gram, grammar JSON parsed as `{"city": "Tokyo"}`); clean benchmark structure (30 rows); clean telemetry; selector provenance; MVP gate tests (25); full Python suite (88 tests + 8 subtests; the earlier MVP execution record reported 84); OpenSpec (40/40).
> * **Blocked:** F8-to-Titan-build binding; benchmark per-result model hashes; MVP terminal decision (`mvp_blocked_evidence`); strict release gate (`rejected`); Nsight profiling (`ERR_NVGPUCTRPERM`).
> * **Advisory:** Titan/llama.cpp throughput ratios (aggregate `0.5286115177544821`); per-model ratios; baseline regression status.
> * **Deferred:** `ffn_sync` full-loop attribution (`missing_real_frontier`); asymmetric reference stage coverage; candidate kernel optimizations; Q8 production integration; RC generation.
> * **Not Exercised:** Fresh GPU benchmark rerun (not justified; existing blockers are schema/binding gaps, not transient execution failures).
>
> **Local-Only Evidence & Publishing Boundary:**
> Directories such as `local-artifacts/`, `.hermes/`, `models/`, and `target/` are local runtime assets and not publishable repository content.

---

## Current Workspace Execution Checkpoint — 2026-09-11

The current organization and verification status is maintained in [`WORKSPACE_STATE.md`](WORKSPACE_STATE.md) and [`docs/MVP.md`](MVP.md):
- **F8.R1.P Correctness:** Verified 5/5 models on F32, batch 1, resident KV, Graph=false, candidate selectors absent (`local-artifacts/reviews/f32-five-model-correctness-matrix-1789097311038-15004.json`).
- **Operational E2E:** Verified across Resident HTTP JSON, Streaming SSE with N-gram speculative acceleration (`local-artifacts/reviews/p5o-e2e-unified-modes-20260912T090000Z.log`), and grammar-constrained JSON decoding with serde verification producing `{"city": "Tokyo"}` (`local-artifacts/reviews/p5o-grammar-json-20260912T100000Z.log`).
- **Clean Throughput Benchmark:** 30 valid rows across 5 models, 3 repetitions, cold/warm conditions, 41 tokens/sample, observed Graph disabled, and clean telemetry (`local-artifacts/benchmarks/p5-final-clean-20260911T151844Z.json`).
- **Strict Release Gate:** Status `rejected` (aggregate ratio `0.5286115177544821` vs required $\ge 0.95\times$; regression failed, `local-artifacts/reviews/p5-release-gate-20260912T130000Z.json`).
- **MVP Status:** `mvp_blocked_evidence` due to missing build binding and per-result model hashes (`local-artifacts/reviews/mvp-decision-20260911T162326Z.json`).

## F3 causal execution checkpoint — 2026-09-11

The fresh diagnostic matrix is linked through
`../local-artifacts/reviews/f3-full-forward-causal-bound-20260911T041000Z-evidence.json`
to its JSON and raw log. It executed five exact models, three repetitions, two
prompt/cache conditions, Graph=false, reference Graph reuse disabled, and no
candidate selectors. The parsed JSON has 30/30 finite/non-negative, non-overlapping
interval rows with residual/tolerance reconciliation, exact token contracts, and
observed Graph disabled. Since telemetry was enabled, provenance is explicitly
`diagnostic_attribution`, not clean throughput. The verdict is
`verified_diagnostic_incomplete` because `ffn_sync=not_available/missing_real_frontier`;
F4 must remain closed.

## Candidate control/candidate gate — 2026-09-07

For the current candidate, the verified commands were:

```bash
# control
env -u TITAN_F32_FFN_GATE_UP_VARIANT TITAN_BENCHMARK_SKIP_GRAPH=1 TITAN_BENCHMARK_REPETITIONS=3 TITAN_BENCHMARK_JSON=local-artifacts/benchmarks/phase9-five-model-control-f32-nograph-20260907.json cargo test --release -p engine-server --test multi_model_comparison_bench -- --ignored --nocapture

# candidate
TITAN_F32_FFN_GATE_UP_VARIANT=q4k_single_row TITAN_BENCHMARK_SKIP_GRAPH=1 TITAN_BENCHMARK_REPETITIONS=3 TITAN_BENCHMARK_JSON=local-artifacts/benchmarks/phase9-five-model-candidate-q4k-single-row-f32-nograph-20260907.json cargo test --release -p engine-server --test multi_model_comparison_bench -- --ignored --nocapture
```

Both completed 5 models × 3 repetitions. The programmatic decision is
`rejected_regression` because Qwen3 warm regressed `5.112480663017738%`; the candidate remains opt-in. Do not use the instrumented FFN smoke or this candidate gate to imply release readiness.

The causal RCA then verified that Qwen3 did not dispatch the candidate kernel: its Gate/Up shape is `1024 x 3072`, while `q4k_single_row` requires `3072 x 8192`. Any further causal comparison must be paired/interleaved on the eligible Llama 3B shape; do not relax the formal gate from this observation.

The paired follow-up passed with three alternating Llama 3B pairs, identical control/candidate token sequences, candidate dispatch observed on all candidate arms, and mean decode delta `-5.822233333333315 ms`. It is diagnostic evidence only; the formal five-model candidate gate remains `rejected_regression`.

The Qwen3-shape follow-up then passed real parity, dispatch smoke and a fresh five-model candidate-relative gate. Qwen3 warm improved `1.726895992021782%`; the candidate is `accepted` as opt-in only. The absolute candidate/reference mean ratio is `0.4637685580716343`, so release remains rejected and no default dispatch change is allowed.

The multi-shape follow-up passed strict parity and dispatch across all five fixtures. The fresh 5×3 control/candidate matrix passed the relative gate in every cell, with minimum improvement `+2.612874495571482%` and median improvements `+7.505232877200196%` cold / `+10.185411006978251%` warm. The candidate is `accepted_opt_in`; its absolute candidate/reference means (`0.5151486550565579x` cold, `0.4729248466771503x` warm) still fail the `0.95x` release gate.

## Candidate QKV multi-row batch=1 gate — 2026-09-08

For the QKV multi-row candidate, verified artifacts and test harnesses are:

- Contract test: `engine/engine-core/tests/qkv_multi_row_batch1_candidate_contract.rs` (verified RED then GREEN; asserts selector `TITAN_F32_QKV_VARIANT=q4k_multi_row_batch1`, allowlist shapes, formats, and fallback).
- Real parity test: `engine/engine-core/tests/real_f32_qkv_multi_row_batch1_parity.rs` ([`real-f32-qkv-multi-row-batch1-parity-1788845218784-10980-0.json`](../local-artifacts/reviews/real-f32-qkv-multi-row-batch1-parity-1788845218784-10980-0.json)); **5/5 strict Q/K/V parity** verified against independent CPU reference calculations (`forward_cpu::matmul`), with `relative_l2 < 1e-4` and `cosine > 0.9999` across Q, K, and V on all five fixtures.
- Dispatch smoke test: `engine/engine-server/tests/real_f32_qkv_multi_row_batch1_dispatch_smoke.rs` ([`real-f32-qkv-multi-row-batch1-dispatch-smoke-1788846368939-35648-0.json`](../local-artifacts/reviews/real-f32-qkv-multi-row-batch1-dispatch-smoke-1788846368939-35648-0.json)); **5/5 actual `gemm_fused_qkv_multi_row_kernel` observations** under candidate arm with `"default"` under control arm; fused Gate/Up, Q6K single-row, and Q8 verified absent.
- Fresh five-model benchmark runs:
  - Control: [`phase18-five-model-control-qkv-batch1-f32-nograph-20260908.json`](../local-artifacts/benchmarks/phase18-five-model-control-qkv-batch1-f32-nograph-20260908.json)
  - Candidate: [`phase18-five-model-candidate-qkv-batch1-f32-nograph-20260908.json`](../local-artifacts/benchmarks/phase18-five-model-candidate-qkv-batch1-f32-nograph-20260908.json)
  - Execution parameters: 5 models, 3 repetitions, 41 generated tokens, temperature 0, `cuda_graphs = false`.
  - **Pairing Status**: Separate non-paired runs; same-run pairing was NOT satisfied. Do not call them a clean benchmark or performance acceptance.
- Authoritative Decision: [`phase18-qkv-multi-row-batch1-decision-20260908.json`](../local-artifacts/reviews/phase18-qkv-multi-row-batch1-decision-20260908.json)
  - `candidate_status`: `rejected_regression`
  - `production_status`: `not_promoted`
  - `release_gate_status`: `rejected`
  - `production_dispatch_changed`: `false`
  - `ffn_sync`: `not_available/missing_real_frontier`
  - `nsight`: `blocked/ERR_NVGPUCTRPERM`
- Gate metrics:
  - Relative regression gate: FAILED (min delta `-55.634076991989666%` on DeepSeek cold, median cold `-9.080238424479903%`, median warm `+45.24988926281488%`).
  - Absolute release gate: FAILED (candidate/reference mean `0.44780836339879965x` cold, `0.5849578722483578x` warm vs required `0.95x`).
- Decisive Rejection: Regressions observed on multiple models far exceed the -5% threshold. The phase18 decision did not authorize a release rescue; the later paired diagnostic is recorded below and is not acceptance. Production dispatch remains strictly unchanged (`production_dispatch_changed = false`).

### QKV same-invocation paired diagnostic follow-up

- Command: `cargo test -p engine-server --test real_f32_qkv_multi_row_batch1_paired_diagnostic -- --ignored --nocapture`; exit code `101`.
- The same-invocation paired harness was attempted but **executed_failed/incomplete** before the five-model matrix.
- Artifact: `../local-artifacts/benchmarks/real-f32-qkv-multi-row-batch1-paired-diagnostic-1788879532145-3500-0.json`, with `status = "token_sequence_mismatch"`, `diagnostic_only = true`, `acceptance_claimed = false`, `release_acceptance_evaluated = false`, and `fixtures = []` because it stopped at the first fixture.
- Raw log: `../local-artifacts/reviews/real-f32-qkv-multi-row-batch1-paired-diagnostic-1788879532145-3500-0.log` records Qwen 2.5 1.5B Pair 1 control tokens `[100905, 3433, 2780, 28339, 22379]` and candidate tokens `[100905, 99545, 147367, 64689, 138136]`; the first token is equal, then the sequences diverge.
- This diagnostic is not a passing acceptance benchmark and does not alter the prior `rejected_regression` / `not_promoted` / release `rejected` decision. Do not infer whether the mismatch is an arithmetic bug or expected drift; RCA remains unresolved and requires a same-domain first-decode differential before any fix or promotion.

### QKV first-decode RCA diagnostic coverage — 2026-09-09

The test-only metrics helper was exercised through RED then GREEN pure tests. The
recorded GREEN command was:

```bash
cargo test --manifest-path engine/Cargo.toml -p engine-server --test real_f32_qkv_multi_row_batch1_paired_diagnostic qkv_first_decode_metrics::tests -- --nocapture
```

It passed 5 helper tests. The scoped compile, format, and lint commands were:

```bash
cargo test --manifest-path engine/Cargo.toml -p engine-server --test real_f32_qkv_multi_row_batch1_paired_diagnostic --no-run
cargo fmt --manifest-path engine/Cargo.toml --all -- --check
cargo clippy --manifest-path engine/Cargo.toml -p engine-server --test real_f32_qkv_multi_row_batch1_paired_diagnostic -- -D warnings
```

All three commands exited 0 in the recorded evidence. The real GPU diagnostic
was run once with the exact command below and exited 101:

```bash
cargo test --manifest-path engine/Cargo.toml -p engine-server --test real_f32_qkv_multi_row_batch1_paired_diagnostic real_f32_qkv_multi_row_batch1_paired_diagnostic -- --exact --ignored --nocapture --test-threads=1
```

Read-only validation used the exact emitted artifact
`../local-artifacts/benchmarks/real-f32-qkv-multi-row-batch1-paired-diagnostic-1788956770585-6596-0.json`.
It verified schema `1.1.0`, `status = "token_sequence_mismatch"`,
`fixtures = []`, `failed_pair` presence, diagnostic-only flags, equal
`first_decode_input = 100905`, finite prefill and first-decode logits of length
`151936`, and the persisted metrics. The raw log and decision are
`../local-artifacts/reviews/real-f32-qkv-multi-row-batch1-paired-diagnostic-1788956770585-6596-0.log`
and `../local-artifacts/reviews/qkv-first-decode-rca-decision-20260909T122609Z.json`.

This is a fail-closed distinction: diagnostic capture is verified, while engine
correctness remains failed/unresolved because the first-pair token sequences
mismatch. The run did not evaluate release acceptance, and the evidence does not
prove a kernel root cause or release readiness.

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
[`openspec/changes/archive/2026-08-26-hardening-error-paths/proposal.md`](../openspec/changes/archive/2026-08-26-hardening-error-paths/proposal.md)).
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
| Benchmark comparison | `cargo test --release -p engine-server --test multi_model_comparison_bench -- --ignored --nocapture`; update [`BENCHMARKS.md`](./BENCHMARKS.md) only with captured output |
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

## Reproduced Titan vs. llama.cpp benchmark

The current comparison harness is:

```bash
cargo test --release -p engine-server --test multi_model_comparison_bench -- --ignored --nocapture
```

It runs two prompts and 41 generated tokens per available model, starts the local CUDA-enabled `llama-server.exe`, then runs Titan on the same GGUF. Missing model files are reported as `SKIP`; a benchmark report must never fill those rows with historical values. On Windows, Titan's NVRTC DLL directory must be visible to the test process.

The historical recorded run (`2026-09-01`) completed with `1 passed, 0 failed` in 100.76 seconds and measured all five configured models with three repetitions each. Its output is `../local-artifacts/benchmarks/rerun-20260901-085229.json` with raw log `../local-artifacts/benchmarks/rerun-20260901-085229.log`. The newer 2026-09-02 checkpoint is recorded separately in `../local-artifacts/benchmarks/final-head-to-head-20260902_165459.json`; performance remains separate from numerical correctness and the release gate is rejected.

## Stable Functional MVP Reproduction Commands

All Python tooling commands require Python 3.12+ via UV with `--project tools --no-sync`:

```powershell
# 1. Run MVP Gate Unit Tests (25 passed)
uv run --project tools --no-sync pytest tools/test_mvp_gate.py -v

# 2. Run Full Python Tooling Suite (88 tests + 8 subtests)
uv run --project tools --no-sync pytest tools -v

# 3. Evaluate the MVP Gate with Verified Evidence Artifacts
uv run --project tools --no-sync python -B tools/mvp_gate.py `
  --benchmark C:/Users/niber/AppData/Local/hermes/workspace/titan/local-artifacts/benchmarks/p5-final-clean-20260911T151844Z.json `
  --correctness-evidence C:/Users/niber/AppData/Local/hermes/workspace/titan/local-artifacts/reviews/f32-five-model-correctness-matrix-1789097311038-15004.json `
  --correctness-sha256 4e1d5571181ca782d32f5fe20c39f1bc8ad06c275c9009496927320cb833d111 `
  --correctness-binding C:/Users/niber/AppData/Local/hermes/workspace/titan/local-artifacts/reviews/mvp-correctness-binding-20260911T162326Z.json `
  --operational-evidence C:/Users/niber/AppData/Local/hermes/workspace/titan/local-artifacts/reviews/mvp-operational-evidence-20260911T162326Z.json `
  --regression-baseline C:/Users/niber/AppData/Local/hermes/workspace/titan/local-artifacts/benchmarks/p2-clean-baseline-20260911T150447Z-aligned.json `
  --output C:/Users/niber/AppData/Local/hermes/workspace/titan/local-artifacts/reviews/mvp-decision-20260911T162326Z.json

# 4. OpenSpec Structural Validation (40 items)
openspec validate --all
```

## Related

- Stable Functional MVP Contract: [`MVP.md`](./MVP.md)
- Canonical spec: [`openspec/specs/layer-streaming-engine/spec.md`](../openspec/specs/layer-streaming-engine/spec.md)
- Constitution (TDD, gate sign-off): [`openspec/constitution.md`](../openspec/constitution.md)
- Benchmarks & methodology: [`BENCHMARKS.md`](./BENCHMARKS.md)
- Architecture: [`ARCHITECTURE.md`](./ARCHITECTURE.md)