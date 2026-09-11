# Titan — Complete Project Diagnosis and Roadmap

**Document status:** Feasibility-reviewed engineering diagnosis and gated roadmap (revision 3)
**Audit timestamp:** 2026-09-11 16:23:26 UTC
**Repository:** `berniehans/titan`
**Workspace:** `C:/Users/niber/AppData/Local/hermes/workspace/titan`
**Branch:** `master`
**Audited HEAD:** `e15fa907f0ccad0aee75b4ac85d0ec08411fb675`
**Stable Functional MVP Contract:** [`docs/MVP.md`](MVP.md)
**Workspace State & Evidence:** [`docs/WORKSPACE_STATE.md`](WORKSPACE_STATE.md)

---

## 🛑 Current Status & Execution Boundary

| Dimension | Current Value | Authority / Artifact |
| :--- | :--- | :--- |
| **Functional MVP Status** | **`mvp_blocked_evidence`** (NOT accepted) | [`local-artifacts/reviews/mvp-decision-20260911T162326Z.json`](MVP.md) |
| **Strict Production Release** | **`not_accepted`** / **`rejected`** | `local-artifacts/reviews/p5-release-gate-20260912T130000Z.json` |
| **Release Approval** | **`not_granted`** | [`docs/RELEASE_ENVELOPE_V1.json`](RELEASE_ENVELOPE_V1.json) |
| **Production Promotion** | **`promotion_authorized = false`** | No candidate promoted to default |
| **Default Runtime Dispatch** | **`production_dispatch_changed = false`** | Verified default F32 resident baseline |
| **Release Candidate Generation** | **`rc_generated = false`** | No release candidate generated |
| **MVP Gate Unit Suite** | **`25 passed, 0 failed`** (100%) | `uv run --project tools --no-sync pytest tools/test_mvp_gate.py` |
| **Full Python Tooling Suite** | **`88 passed, 0 failed; 8 subtests passed`** (100%) | `uv run --project tools --no-sync pytest tools` |
| **OpenSpec Live State** | **`40 passed, 0 failed`** (40 items) | `openspec validate --all` |

> [!IMPORTANT]
> **Functional MVP vs. Strict Production Release:**
> * A throughput ratio of $\ge 0.95\times$ compared to `llama.cpp` is **NOT** an acceptance criterion for the Stable Functional MVP. The $0.95\times$ comparison is exclusively a strict production-performance diagnostic.
> * In the functional MVP evaluation, throughput ratios (aggregate median `0.5286115177544821`) and baseline regressions are advisory warnings (`ratio_advisory: advisory_warning`, `regression_advisory: failed`).
> * Conversely, **generation correctness, operational E2E execution, and clean benchmark safety remain strictly blocking**.
>
> **Exact MVP Blockers:**
> 1. **`build_binding_unverified`:** The independent F32 correctness run (F8.R1.P) verified model accuracy across 5 models, but does not record the Titan binary or source identity matching the clean benchmark's `engine_identity.titan` build hash (`sha256:bc92211356bb0eb87ddd59e252d9d4497f8669d7257ac0133459040ce4eb131c`).
> 2. **`model_hash_missing`:** The final clean benchmark artifact (`local-artifacts/benchmarks/p5-final-clean-20260911T151844Z.json`, 30 rows) records model names but lacks per-result GGUF weight file hashes required for exact binding to F8.
>
> **Local-Only Evidence & Publishing Boundary:**
> Directories such as `local-artifacts/`, `.hermes/`, `models/`, and `target/` contain local execution logs, private model weights, and compiler targets. They are **not** publishable repository content. Advanced features (speculative decoding, layer streaming, APC, attention sinks) exist in the codebase but are **not** all production-integrated or certified under the release envelope.

---

## 1. Executive Diagnosis

Titan is a technically viable experimental LLM inference engine with a substantial working foundation, but it is not currently a release candidate.

The correct overall classification is:

> **Functional experimental F32 inference engine. Stable Functional MVP is currently `mvp_blocked_evidence` due to unproven F8-to-build binding and missing per-result model hashes in the clean benchmark artifact; strict competitive release remains `rejected` on the strict performance gate (aggregate ratio `0.5286115177544821` vs required $\ge 0.95\times$; regression failed). Generation correctness (5 models) and operational E2E behavior (Resident JSON, Streaming SSE with N-gram, grammar JSON parsed as `{"city": "Tokyo"}`) are independently verified, but no production promotion or release candidate is authorized.**

The project is not blocked because the architecture is missing. The base engine, CUDA runtime, model ingestion, serving path, KV management, advanced runtime features, and observability primitives already exist. The project is blocked because the current evidence does not yet bind the verified correctness execution to the final clean benchmark build, and throughput remains below the separate strict production threshold. Implementation presence is not proof that every advanced feature is integrated or release-verified.

### Current Decisions

- **Stable Functional MVP:** Follows the separate contract in [`docs/MVP.md`](MVP.md) where throughput ratio is advisory and correctness/operations are blocking. Current status is `mvp_blocked_evidence`.
- **F32 Baseline:** F32 remains the sole production correctness and default route.
- **Q8 Status:** Q8 remains experimental and benchmark-only.
- **Candidate Kernels:** QKV multi-row batch 1 remains rejected and not promoted; multi-shape Gate/Up remains opt-in only.
- **Production Dispatch:** No experimental selector changes production defaults (`production_dispatch_changed = false`).
- **Attribution Frontiers:** `ffn_sync` remains explicitly unavailable when no independently measurable frontier exists.
- **Nsight:** Blocked on Windows by `ERR_NVGPUCTRPERM`.
- **Release Authority:** `promotion_authorized = false`, `rc_generated = false`, `release_approval = not_granted`. No commit or push is authorized without Bernie's explicit approval.

---

## 2. Evidence and repository state

### 2.1 Git state

The audited checkout is intentionally dirty and must be preserved.

- Branch: `master`, tracking `origin/master`.
- HEAD: `e15fa907f0ccad0aee75b4ac85d0ec08411fb675`.
- Tracked modified paths: 13.
- Untracked Git status entries: 37 at feasibility review (directory entries are not individual file counts).
- `git diff --check`: no whitespace errors; only LF/CRLF conversion warnings.
- No reset, restore, clean, stash, merge, commit, or push was performed.
- At the original audit, no active coder job or benchmark worker was detected (not re-probed during this planning review):
  - `agy-worker-pool/queue`: 0 entries.
  - `agy-worker-pool/running`: 0 entries.

The dirty tree contains production-path edits, test-only diagnostics, OpenSpec changes, documentation, and local planning material. It must not be treated as a single homogeneous change. Any future commit must be assembled from reviewed paths and hunks only.

### 2.2 OpenSpec state

Live structural validation:

```text
openspec validate --all
Totals: 40 passed, 0 failed (40 items)
```

The repository currently contains 24 active changes and specs validating cleanly. Note that `openspec validate --all` proves document structure only. It does not prove runtime correctness, performance, or release readiness.

### 2.3 Last Recorded Software Gates & Tooling Suites

The repository software gates have been reconciled and verified:

```bash
# Cargo Workspace Gates
cargo fmt --manifest-path engine/Cargo.toml --all -- --check
cargo check --manifest-path engine/Cargo.toml --workspace
cargo clippy --manifest-path engine/Cargo.toml --workspace --all-targets -- -D warnings
cargo test --manifest-path engine/Cargo.toml --workspace
```

* **Cargo Test Note:** An initial false-positive in `engine/engine-server/tests/full_forward_causal_bound.rs` was corrected in the test file only (no runtime/producer modification); subsequent workspace tests exited 0 (`local-artifacts/reviews/m5-final-cargo-gates-after-test-fix-20260911T162326Z.raw.log`).
* **MVP Gate Unit Suite:** 25 passed, 0 failed (`uv run --project tools --no-sync pytest tools/test_mvp_gate.py`).
* **Full Python Tooling Suite:** 88 passed, 0 failed; 8 subtests passed (`uv run --project tools --no-sync pytest tools`).
* **MVP Decision:** Evaluated via `tools/mvp_gate.py` yielding `mvp_blocked_evidence` (`local-artifacts/reviews/mvp-decision-20260911T162326Z.json`).
* **Strict Production Release Gate:** Evaluated via `tools/release_gate.py` yielding `status: rejected` (`local-artifacts/reviews/p5-release-gate-20260912T130000Z.json`).

---

## 3. Architecture diagnosis by subsystem

Titan is organized around five architectural planes. They must retain separate correctness, performance, and release gates.

### 3.1 Ingestion and representation

**Status: implemented; ongoing contract hardening.**

Capabilities include:

- GGUF loading and metadata inspection.
- Tokenizer and model configuration resolution.
- Tensor dimensions, layouts, formats, and size inspection.
- Q4_K, Q6_K, Q8_1, F16, and F32 representation paths.
- Five real reference model fixtures used by diagnostic and benchmark work.

Primary risk:

- The F32 driver label does not always mean pure F32 arithmetic inside a candidate kernel. The QKV candidate path quantizes activations to Q8_1 before dispatch.
- Every future parity document must name the actual numerical domain: original F32 activations, CPU Q8_1 reconstruction, or GPU Q8_1 execution.

Required invariant:

> A Q8_1-domain parity result must never be relabeled as strict original-F32 end-to-end equivalence.

### 3.2 GPU execution, CPU reference, and memory

**Status: substantial implementation; production F32 path retained; experimental candidates gated.**

Implemented or verified areas include:

- Independent CPU reference operations.
- CUDA/NVRTC runtime compilation and manual dispatch.
- Q4_K and Q6_K dequantized GEMV/GEMM paths.
- Q8_1 activation handling.
- Q6_K dispatch correction for the Q8 path.
- Paged KV-cache and attention kernels.
- CUDA Graph execution.
- VRAM-aware execution and layer streaming.

Important correctness facts:

- The Q6_K Q8-dispatch repair was verified.
- Q8 activation quantizer bytes were verified against the CPU representation.
- Full-driver Q8 still exhibits distributed numerical drift and remains unaccepted.
- The QKV candidate passed isolated Q/K/V parity against its declared CPU Q8_1-domain reference, but failed the paired first-decode diagnostic.

### 3.3 Forward driver, generation, and scheduling

**Status: operational F32 path; candidate paths isolated.**

Capabilities include:

- Real forward execution.
- Prefill and decode.
- Resident KV-cache.
- CUDA Graph and non-Graph paths.
- Layer streaming and double buffering.
- Continuous batching and multi-slot scheduling.
- Greedy sampling and diagnostic token comparisons.
- Speculative decoding and experimental MoE paths.

Current boundary:

- Production defaults remain unchanged.
- Candidate selectors are opt-in only.
- The QKV candidate has not been accepted into the production dispatch.
- `ffn_sync` cannot be reported as a measured stage without an independent CUDA dependency/wait frontier.

### 3.4 Server, API, and CLI

**Status: functionally verified in the current configuration.**

Capabilities include:

- OpenAI-compatible `/v1/models` and `/v1/chat/completions`.
- SSE streaming and non-streaming responses.
- CLI commands for serving, chat, benchmarking, and agent operation.
- Grammar-constrained JSON and tool-calling support.
- Continuous batching and asynchronous ingress.

This plane is not the current primary release blocker, but it must be rechecked against the final accepted code diff before an RC is declared.

### 3.5 Evidence, observability, and release

**Status: improved and partially verified; release gate remains rejected.**

Verified capabilities include:

- Machine-readable benchmark and decision artifacts.
- Runtime dispatch telemetry.
- FFN grouped attribution for five models.
- Projection attribution for five models.
- Finite/non-negative timing validation.
- Reconciliation of measured groups and unattributed residuals.
- Explicit status values for unavailable or blocked measurements.

Remaining evidence limitations:

- `ffn_sync` has no independent real frontier.
- Nsight counters are blocked by `ERR_NVGPUCTRPERM`.
- Some OpenSpec task files have not been synchronized with already-existing verified artifacts.
- The release baseline must be regenerated after the final accepted diff is known.

---

## 4. Correctness diagnosis

### 4.1 F32 production path

F32 remains the safest production correctness route. Its default dispatch was preserved throughout the candidate experiments.

This does not imply that every opt-in F32 candidate is correct. Correctness is evaluated per dispatch path and per numerical contract.

### 4.2 Q8 status

Q8 evidence is split into three separate conclusions:

1. Q8_1 representation and activation quantizer behavior are diagnostic-verified.
2. Q6_K dispatch incompatibility in the Q8 path was repaired and verified.
3. Full-driver Q8 still fails the strict contract and is not accepted.

Decision:

> Keep Q8 benchmark-only/experimental. Do not change the production correctness default and do not relax the strict full-driver contract to obtain a green result.

Q8 can be revisited in a separately scoped correctness change if the project later requires it, but it should not block F32 release hardening indefinitely once its experimental status is explicitly documented.

### 4.3 QKV multi-row batch=1 candidate

Candidate selector:

```text
TITAN_F32_QKV_VARIANT=q4k_multi_row_batch1
```

Candidate kernel:

```text
gemm_fused_qkv_multi_row_kernel
```

Scope:

- F32 driver path.
- Batch size 1.
- Non-Graph decode.
- Resident KV-cache.
- Q4_K/Q4_K/Q6_K weight combination.
- Five model fixtures.

Evidence:

- Isolated Q/K/V parity: 5/5 verified.
- Runtime dispatch smoke: 5/5 verified.
- Historical candidate decision: `rejected_regression`.
- Production status: `not_promoted`.
- Production dispatch change: `false`.

Authoritative decision artifact:

```text
local-artifacts/reviews/phase18-qkv-multi-row-batch1-decision-20260908.json
```

The historical benchmark recorded:

- Minimum relative delta: `-55.634076991989666%`.
- Median cold delta: `-9.080238424479903%`.
- Median warm delta: `45.24988926281488%`.
- Candidate/reference mean ratio: `0.44780836339879965x` cold.
- Candidate/reference mean ratio: `0.5849578722483578x` warm.
- Required absolute ratio: `0.95x`.

These benchmark runs were separate rather than same-invocation paired runs, so they are not clean causal acceptance evidence. They are nevertheless sufficient for the conservative formal rejection because the observed failures were far beyond the declared regression threshold.

### 4.4 QKV first-decode RCA

Authoritative RCA artifact:

```text
local-artifacts/reviews/qkv-first-decode-rca-decision-20260909T122609Z.json
```

Real diagnostic result:

- Harness exit code: `101`.
- Harness result: `token_sequence_mismatch`.
- Diagnostic capture: `verified`.
- `failed_pair`: persisted.
- `fixtures`: empty because the first fixture failed before matrix completion.
- `root_cause_status`: `unresolved`.
- `release_acceptance_evaluated`: `false`.

First observed pair:

- Control tokens: `[100905, 3433, 2780, 28339, 22379]`.
- Candidate tokens: `[100905, 99545, 147367, 64689, 138136]`.
- First decode input: `100905` in both arms.

Prefill logits:

- Length: `151936`.
- Relative L2: `0.0`.
- Cosine: `1.0`.
- Top-1: equal.

First-decode logits:

- Length: `151936`.
- Relative L2: `1.1629506279165833`.
- Cosine: `0.3923437654457373`.
- Maximum absolute difference: `20.243502616882324`.
- Control top-1: `3433`.
- Candidate top-1: `99545`.

Interpretation:

- Autoregressive input contamination does not explain the first pair because the inputs match.
- The divergence appears at or before the first decode output.
- The current evidence does not prove whether the cause is activation quantization, bias asymmetry, ABI/layout, memory behavior, ordering, internal state, or a legitimate pipeline difference.
- The diagnostic capture is verified, but engine/candidate correctness is not.

Required next experiment:

1. First decide whether to investigate or quarantine the rejected candidate. RCA is required before its repair/reconsideration, not before unrelated production work once isolation is verified.
2. Repeat control-versus-control with fresh state and identical configuration. Check prefill/KV state as well as logits; equal prefill logits do not establish identical internal state.
3. Capture original F32 activations at the first differing layer/projection. The default F32 arm is not assumed to consume Q8_1 bytes. Persist the candidate's actual Q8_1 bytes/scales and reconstruct their values independently on CPU.
4. Record original bias presence, shapes, values, weight layouts, stream dependencies, and call arguments before changing anything. Equalizing biases is a controlled diagnostic intervention, not evidence that either original branch was correct.
5. Separate three comparisons: original-F32 CPU/GPU control; CPU-Q8_1 reconstruction versus candidate GPU using identical quantized bytes; original-F32 reference versus reconstructed-Q8_1 reference to measure the quantization contribution.
6. Use a diagnostic-only reference-aligned GPU arm if needed; never silently convert the production control into a Q8 path to obtain parity. Compare Q, K, V independently before RoPE/cache insertion, then downstream attention and layer outputs only as necessary.
7. Preserve partial artifacts and stop at the first causal divergence. Use teacher-forced identical token inputs for later-step comparisons; free-running trajectories after divergence are not same-input evidence.
8. Open a separate correctness-repair change only if a causal defect is proven. An unresolved cause may remain recorded while the candidate is excluded from the release manifest.

No repair of this rejected QKV candidate should be attempted before causal evidence; this does not prohibit fixing an independently proven production defect.

**Source-traced numerical contract:** `DecodePath::F32` calls `record_decode_pass_f32`, whereas `DecodePath::Q8` calls `record_decode_pass` (`engine/engine-core/src/forward_driver.rs:4147-4148`, `:4822-4828`). The F32 default Q/K/V calls consume `input_norm_dev` (`:3771-3806`); only its opt-in QKV branch quantizes to `qx_dev`, `qd_dev`, `qs_dev` (`:3736-3769`). The quantization at `:3351` belongs to the separate Q8 route. Therefore, a requirement to capture Q8_1 buffers “consumed by both original F32 arms” is invalid. Record `driver_route`, `activation_domain`, and `weight_formats` per actual branch, not from a filename or an adjacent function.

**Missing diagnostic capability:** the current `ArmRecord` persists logits/tokens/telemetry but not activation buffers (`engine/engine-server/tests/real_f32_qkv_multi_row_batch1_paired_diagnostic.rs:270-288`). If RCA is reopened, first design a narrow, explicitly opt-in diagnostic copy-out interface in `ForwardDriver` and the paired harness: capture the default's F32 input and the candidate's actual quantized bytes/scales/raw sums at a bounded first-layer/first-decode scope, plus native bias and Q/K/V outputs. An integration-test crate cannot assume access to library-private buffers or rely on library `cfg(test)` visibility. Use an explicitly gated diagnostic feature/interface or in-module unit harness, verify it is inactive in production, and keep downloads/synchronizations out of throughput runs. This requires a new scoped change, not a claim that the current harness-only capture already provides it.

CPU reconstruction must consume the downloaded candidate buffers for exact same-buffer evidence. The separate CPU quantizer in `engine/engine-core/tests/real_f32_qkv_multi_row_batch1_parity.rs:204-240` remains useful as an independent quantizer test; its re-quantization is not by itself proof of identity with the bytes consumed in a specific failed driver invocation.

---

## 5. Performance diagnosis

### 5.1 Release gate

Authoritative artifact:

```text
local-artifacts/reviews/fresh-head-to-head-release-gate-20260902.json
```

Current result:

- Status: `rejected`.
- Aggregate ratio: `0.4955685063184848x`.
- Required aggregate ratio: `>=0.95x`.
- All five models fail the absolute ratio requirement.

Last recorded ratios:

- Qwen 2.5 1.5B:
  - cold: `0.5036268807958383x`;
  - warm: `0.48798950480392617x`.
- Llama 3.2 1B:
  - cold: `0.48840626305308227x`;
  - warm: `0.4974728956915047x`.
- Llama 3.2 3B:
  - cold: `0.45642004158495675x`;
  - warm: `0.4607510094707909x`.
- DeepSeek-R1-Distill 1.5B:
  - cold: `0.5013434692214681x`;
  - warm: `0.49366411694546497x`.
- Qwen3 0.6B:
  - cold: `0.7229566155803784x`;
  - warm: `0.6555136347763789x`.

The performance deficit is broad, not isolated to a single model. Llama 3.2 3B is a useful worst-case sentinel, while Qwen3 has the strongest historical absolute ratio but still does not reach the project gate. These September 2 observations are historical, not a fresh measurement of the current dirty build.

### 5.1.1 Quantified feasibility and denominator discipline

Calculated directly from the stored gate ratios during this review:

- Uniform speedup needed to move the historical aggregate to the unchanged target: `0.95 / 0.4955685063184848 = 1.9169902604534512x`.
- Equivalent latency reduction under that uniform-speedup assumption: `47.83489407173844%`.
- Llama 3.2 3B requires `2.0814160497883605x` cold and `2.061851152732471x` warm.
- Qwen3 requires `1.3140484221689495x` cold and `1.4492452171861867x` warm.

These are arithmetic targets, not forecasts. The aggregate is the **median of the ten per-model/condition ratios**, not the ratio of aggregate token rates. Keep that definition for continuity; never use it to hide a failing model/condition.

If a workload-weighted summary is added, name it separately and define direction consistently as Titan/reference: `(sum(Titan output tokens)/sum(Titan decode seconds)) / (sum(reference output tokens)/sum(reference decode seconds))`, using the same explicitly weighted workload per condition. It is supplementary diagnostic reporting unless a versioned release-contract change adopts it; do not retroactively replace the historical macro metric or introduce a reversed reference/Titan ratio.

For a stage occupying fraction `f` of total relevant latency and a stage speedup `s`, estimate the upper bound with `S = 1 / ((1-f) + f/s)`. Measure `f` on the same production configuration and timing interval. The approximately `64.8%` Gate/Up share is a fraction of **FFN time**, not total decode time. Do not multiply medians from different models or instrumentation configurations to predict end-to-end speedup. Account separately for host gaps, launch overhead, transfers, attention, sampling, and uninstrumented residuals.

Before implementing a new candidate, its hypothesis must include an optimistic bound and a plausible bound on end-to-end improvement. A useful incremental improvement may be retained without meeting the entire release target; it must not be presented as a complete gap-closure strategy.

### 5.2 Attribution evidence

FFN grouped attribution artifact:

```text
local-artifacts/reviews/five-model-ffn-attribution-matrix-1788823934726-3700-0.json
```

Verified properties:

- Overall status: `verified`.
- Model count: 5.
- Every model status: `verified`.
- Gate/Up and Down timings: finite, non-negative, reconciled.
- `ffn_sync`: `NotAvailable`, `missing_real_frontier`.
- Instrumentation explicitly marked diagnostic-only, not throughput acceptance.

Measured FFN shares across the five-model evidence indicate:

- Gate/Up is the dominant group, with a median share of approximately `64.8%`.
- Down contributes approximately `25.8%` median share.
- The remaining time is residual/unattributed and must not be assigned to synchronization without a real frontier.

Projection attribution artifact:

```text
local-artifacts/reviews/five-model-projection-attribution-control-1788830924265-9040-0.json
```

Verified groups:

- `qkv`.
- `attn_output`.
- `ffn`.
- `lm_head`.

QKV is the largest measured non-FFN projection group, with an observed median around `1.534976 ms` in the attribution evidence.

### 5.3 Candidate ledger

#### Fused Gate/Up

- Status: `rejected_regression`.
- No production promotion.
- Do not reactivate as a blind retry.

#### Q6_K single-row

- Status: `rejected_regression`.
- A Qwen3 regression exceeded the declared limit.
- No production promotion.

#### Qwen3 shape-specific Gate/Up

- Relative candidate result: accepted.
- Production: not promoted.
- Absolute release gate: rejected.
- Remains shape-scoped and opt-in.

#### Multi-shape Gate/Up single-row

Selector:

```text
TITAN_F32_FFN_GATE_UP_VARIANT=q4k_multi_shape_single_row
```

- Isolated parity: verified across five fixtures.
- Dispatch: verified across five fixtures.
- Relative candidate gate: accepted opt-in.
- Minimum relative improvement: `2.612874495571482%`.
- Median cold improvement: `7.505232877200196%`.
- Median warm improvement: `10.185411006978251%`.
- Absolute release gate: rejected.
- Candidate/reference mean: `0.5151486550565579x` cold and `0.4729248466771503x` warm.
- Production: not promoted.

This candidate is the only currently retained performance candidate, but it is not a release solution.

#### QKV multi-row batch=1

- Status: `rejected_regression`.
- First-decode mismatch observed.
- RCA unresolved.
- No retry or promotion before same-domain differential evidence.

#### FFN-down 2col

- Status: `rejected_neutral`.
- Measured improvement was not material.
- Do not spend another cycle on the same hypothesis without new evidence.

---

## 6. Governance and documentation diagnosis

### 6.1 Dirty-tree risk

The feasibility-review snapshot contains 13 tracked modified paths and 37 untracked Git status entries across code, tests, OpenSpec, documentation, and local planning files. This is acceptable for an experiment, but HEAD alone is not a reproducible build identity. Preserve the dirty checkout; create a content-hashed source/build manifest before acceptance runs and, at packaging time, an explicitly allowlisted release snapshot. A clean unrelated working directory is not itself a correctness gate. The untracked empty `total_ms` file was observed and preserved; its provenance is not established.

Required handling:

- Preserve all existing paths.
- Review hunks, not just filenames.
- Do not use `git clean`, `git reset`, `git restore`, or deletion as cleanup.
- Build any future commit from an explicit allowlist.
- Exclude `.hermes/`, `local-artifacts/`, unrelated experiments, and secrets unless deliberately reviewed.

### 6.2 OpenSpec synchronization debt

Two examples require explicit reconciliation:

- The five-model FFN matrix artifact is verified, but its `tasks.md` still leaves the four master-execution tasks unchecked.
- The QKV candidate retains unchecked tasks for same-run pairing because the paired attempt failed before matrix completion.

The correct response is not to check every task. Each task must be classified as one of:

- `verified`;
- `failed`;
- `blocked`;
- `not_exercised`;
- `rejected_neutral`;
- `rejected_regression`.

### 6.3 Constitution/workflow mismatch

`openspec/constitution.md` currently says code generation is delegated to Agy with a historical model selection. The active Hermes workflow uses the `coder` bot and requires independent master verification.

The constitution is explicitly immutable and must not be edited without Bernie’s direct approval. Until that approval exists:

- follow the active `coder` workflow for implementation;
- do not silently rewrite the constitution;
- record the mismatch as governance debt.

### 6.4 CUDA Rust compatibility with project rules

The current constitution requires stable Rust.

- `cutile-rs` is the first candidate for an isolated experiment because it is the higher-level tiled path and is compatible with a stable-toolchain direction.
- `cuda-oxide` requires a nightly/fixed compiler-oriented toolchain and therefore cannot be treated as an unreviewed default dependency.
- Neither track is a drop-in replacement for Titan’s current `cudarc`/NVRTC/manual-launch architecture.

Project documentation, specifications, and international-facing README content should remain in English. Operational notes that are currently in Spanish represent documentation debt and should be translated during the next documentation synchronization pass.

---

# 7. Consolidated roadmap

The roadmap is divided into historical foundation, active release hardening, an isolated CUDA Rust research track, and post-release scaling. The historical phases are not reopened unless current work proves a regression.

---

## 7.1 Historical foundation — F0 through F7

### F0 — Constitution, repository, and reproducibility

Scope:

- Repository and Cargo workspace.
- OpenSpec governance.
- Toolchain and reproducible commands.
- Artifact layout.
- VRAM and reference-hardware constraints.

Status: implemented historically.

### F1 — GGUF ingestion and tokenizer

Scope:

- GGUF reader.
- Tensor metadata.
- Model configuration.
- Tokenizer.
- Format and shape validation.

Status: implemented.

### F2 — CPU reference bank

Scope:

- Independent dequantization.
- RMSNorm.
- RoPE.
- GEMV/GEMM references.
- SwiGLU.
- Layer composition.
- Numerical goldens.

Status: implemented and required for all future kernel changes.

### F3 — CUDA dequantization and GEMV/GEMM

Scope:

- Q4_K, Q6_K, Q8_1, F16, and F32 paths.
- DP4A and vectorized loads.
- Format-specific dispatch.
- Kernel fallback policies.

Status: implemented with targeted repairs and experimental variants.

### F4 — Paged KV-cache and attention

Scope:

- Paged KV-cache.
- Paged attention.
- Flash decoding.
- Scattered layouts.
- Bounds and memory protection.

Status: implemented; preserve current parity before future changes.

### F5 — Layer streaming and double buffering

Scope:

- NVMe to pinned RAM to VRAM movement.
- Ping-pong buffers.
- Copy/compute overlap.
- VRAM budget enforcement.

Status: implemented.

### F6 — Real forward, server, and CLI

Scope:

- Forward driver.
- Prefill/decode.
- HTTP server.
- SSE.
- CLI.
- OpenAI compatibility.
- Tool calling.

Status: functionally implemented.

### F7 — Advanced runtime capabilities

Scope:

- Continuous batching.
- CUDA Graphs.
- Prefix/radix cache.
- StreamingLLM and attention sinks.
- FP8 KV experimentation.
- Speculative decoding.
- MoE expert streaming.
- Release instrumentation.

Status: implemented in large part; each feature retains its own acceptance gate.

---

## 7.2 Active release-hardening phase — F8

F8 is the critical path. No post-release expansion should displace it. The IDs below are stable references, **not a serial execution order**.

**Dependency order (revision 2):**

1. `R0` evidence/source identity + `R0.5` validator hardening.
2. `R1` experimental isolation + `R1.P` production correctness/operational contract. These can proceed independently of optional QKV RCA.
3. `R2` comparable baseline for the frozen iteration build + `R3` diagnostic attribution on the corresponding configuration.
4. `R4` quantified hypothesis → `R5` atomic candidate evaluation; repeat only when new evidence justifies it.
5. Freeze the final selected build; repeat `R1.P` and `R2` on that exact identity, then `R7` acceptance → `R8` packaging/manifest and final identity check.

`R1.5` is conditional research; `R6` is conditional profiling; neither blocks the main path merely by being unresolved. If either discovers a defect affecting the production path, that defect becomes a production correctness blocker. CUDA Rust remains a separate laboratory track with no production-default changes.

### F8.R0 — Freeze truth and reconcile the workspace

Objectives:

1. Preserve the dirty checkout and exact HEAD.
2. Classify every changed path by architectural plane.
3. Build one evidence matrix linking claims to artifacts.
4. Reconcile OpenSpec checkboxes with semantic results.
5. Mark stale or historical metrics explicitly.
6. Translate international-facing project documentation to English where required.
7. Keep the constitution unchanged unless Bernie explicitly approves a governance change.

Exit gate:

- No unexplained conflict between Git, OpenSpec, artifacts, and documentation.
- No historical rejection presented as a current pass.
- No production code changed merely to make documentation counts green.

Current status: partially complete; documentation and task synchronization remain.

### F8.R0.5 — Harden the acceptance evidence validator (new prerequisite)

**Why now:** `tools/release_gate.py` checks too little provenance and can accept malformed or failed evidence. During this planning review, the existing `unittest` suite passed, but isolated synthetic CLI probes reproduced `accepted` with each of: `status=failed`; a duplicate model; `samples=999` against two runs; mismatched identity metadata; and an incompatible supplied regression baseline silently skipped. These fixtures test the validator, not Titan performance.

Implementation scope, through `coder`, in a separate OpenSpec change:

- `tools/release_gate.py`, `tools/test_release_gate.py`.
- Inspect and align `tools/compare_static_baseline.py`, `tools/test_compare_static_baseline.py`, and `tools/checkpoint_matrix.py` for the same class of fail-open behavior.
- Extend producer provenance in `engine/engine-server/tests/multi_model_comparison_bench.rs` only where fields cannot be supplied by a verified manifest. Version schema changes explicitly; historical artifacts remain historical rather than being silently upgraded.

Required RED cases: missing/non-success/unknown status; duplicate/missing/extra model IDs; mismatched GGUF/prompt/build/config identity; insufficient/duplicate/incomplete repetitions; non-finite raw samples; summary statistics inconsistent with raw runs; supplied incompatible/incomplete regression baseline; diagnostic instrumentation enabled; correctness prerequisite absent; truncated/early-return output.

Required behavior:

1. Accept only explicit allowed terminal statuses and exact required model/condition coverage.
2. Validate complete raw samples, then recompute summaries; do not trust a positive summary median or a self-declared sample count.
3. Compare a documented compatibility manifest, including per-model hashes, prompt tokens, token count, numerical/dispatch/Graph/cache settings, runner versions, and environment. Engine-specific fields may differ intentionally, but the difference and timing contract must be explicit.
4. A supplied regression baseline that cannot be compared must return `incomparable`/non-success, never silently bypass the regression check. Absence must be declared `not_evaluated` and cannot satisfy a mandatory RC regression gate.
5. Keep `evidence_valid`, `correctness`, `candidate_relative`, `release_absolute`, and `regression` verdicts separate. A throughput pass alone is not a release pass.

Verification entry point (existing suite; new cases must be added before claiming this milestone complete):

```bash
uv run --project tools --no-sync python -m unittest discover -s tools -p 'test_*.py' -v
```

Exit: the original fail-open probes are rejected for the intended reasons, valid fixtures remain accepted, historical failed release evidence remains failed, and master independently reviews the diff/results. **Current status: gap reproduced; repair not implemented.**

### F8.R1 — Decide and isolate the Q8 contract

Objectives:

1. Preserve the verified Q8 quantizer and Q6_K dispatch repair.
2. Document full-driver Q8 as not accepted.
3. Keep F32 as the production correctness route.
4. Keep Q8 selectors opt-in and experimental.
5. Prevent Q8 drift from being used as F32 correctness evidence.

Exit gate:

- Q8 contract explicitly states its numerical tolerance and experimental status.
- No Q8 default dispatch.
- All Q8 artifacts retain diagnostic classification.

Current decision: keep Q8 experimental and remove it from the F32 release critical path unless a future product requirement reopens it.

### F8.R1.P — Verify the chosen production envelope (new prerequisite)

Define one release configuration first: F32 activation route, exact weight formats/models, supported batch/context ranges, Graph policy, KV format/cache policy, hardware/VRAM budget, and enabled server features. Unverified advanced modes remain experimental or explicitly unsupported, not implicitly certified by an F0–F7 heading.

Required evidence before accepting new production performance results:

- Default selectors and unsupported shape/format/mode fallback paths never dispatch rejected QKV/Q8 candidates; verify configuration tests plus observed runtime symbols under the chosen envelope.
- Independent CPU/reference operation parity and same-input full-driver/logit checks on production, plus deterministic greedy generation fixtures. Test prefill/decode, first decode, multi-step teacher forcing, and Graph/non-Graph only where claimed supported. Tolerances are declared per numerical domain before seeing candidate results, not one universal kernel tolerance applied to every layer/model.
- Long/context-boundary and VRAM-budget checks, including cache reuse/reset and concurrent request isolation when enabled.
- Minimum serving contract: real-model streaming/non-streaming consistency, stop/EOS/max-token handling, invalid input and oversized context rejection, bounded ingress, disconnect/cancellation resource cleanup, and explicit OOM/CUDA failure handling. Successful recovery or explicit fail-stop behavior must be documented and tested; do not promise recovery that does not exist.
- JSON/tool-call conformance belongs here if advertised in the release, rather than being postponed entirely to F9. Quality beyond the advertised contract remains post-release research.

Existing starting points (coverage must be inspected; filenames do not imply acceptance): `engine/engine-server/tests/driver_parity_gate.rs`, `e2e_real_forward_sse.rs`, `e2e_chat_completions.rs`, `multi_slot_concurrency_test.rs`, `grammar_constrained_tool_call_test.rs`, `vram_real_audit_gate.rs`, and `engine/engine-core/tests/decode_drift_gate.rs`. Enumerate their exact test filters and ignored status before execution; add missing cases through `coder` rather than assuming coverage.

Exit: a feature-to-entrypoint-to-test-to-artifact matrix for the chosen production configuration. Any correctness defect actually affecting this configuration blocks dependent acceptance; a failed quarantined candidate does not. **Current status: final-build matrix not exercised in this planning review.**

**Confirmed execution-evidence hazard:** `engine/engine-server/tests/e2e_unified_modes_gate.rs:75-82` is ignored by default and returns successfully when its fixture is absent. A required release runner must fail/mark `blocked` on missing prerequisites, assert that each intended case actually executed, wait for readiness with a timeout, and verify cleanup of test-owned tasks/listeners. The current `tokio::spawn` server helper (`:61-72`) does not return a shutdown handle. Add those harness prerequisites before calling its exit code a release pass; run a bounded repeated-request check for leaks/state carryover. This is a static inspection finding, not a newly observed production-server failure.

### F8.R1.5 — Resolve or quarantine the QKV first-decode RCA (conditional)

Objectives:

1. Run control-control reproducibility.
2. Capture original F32 activations and candidate Q8_1 bytes separately; follow the three-comparison ladder in section 4.4.
3. Record native bias handling before a controlled diagnostic equalization.
4. Compare each GPU arm to its correct independent numerical-domain reference before comparing arms.
5. Isolate Q, K, and V independently.
6. Identify the first causal divergence.
7. Open a separate repair change only if the defect is proven.

Exit states:

- `verified_cause`: a controlled experiment proves the mechanism.
- `quarantined_unresolved`: candidate excluded from the release configuration, runtime isolation verified, causal uncertainty preserved. Rejection does not require proving the absence of a defect.
- `rejected_candidate`: candidate formally abandoned; retain whether its cause was proven or unresolved.
- `blocked`: required same-domain evidence cannot be obtained.

Forbidden before causal evidence and fresh candidate acceptance:

- kernel repair;
- candidate promotion;
- benchmark rescue;
- production default change;
- use of CUDA Rust as a substitute for RCA.

Current status: causal mechanism unresolved; historical candidate rejected. Recommended disposition: quarantine after production isolation verification; investigate only if it informs a shared production defect or a justified new hypothesis. No global hold on independent F32 hardening.

### F8.R2 — Establish a final comparable reference

Objectives:

1. Pin the llama.cpp source/build identity.
2. Hash all five GGUF models.
3. Record GPU, driver, CUDA, NVRTC, Rust, compiler, and build metadata.
4. Use identical model files, prompt token IDs, actual generated-token counts, sampling and batch/context workloads. For Titan control/candidate, match Graph/cache/numerical settings except the declared intervention. For llama.cpp versus Titan, record engine-specific policies explicitly: run a like-for-like diagnostic comparison separately from the declared best-supported production configuration; do not disable a reference optimization merely to make the headline ratio favorable.
5. Use three repetitions only as a smoke/pilot floor, not proof of a small gain. For acceptance, predeclare a balanced randomized or AB/BA process schedule, minimum and maximum samples, warm-up/reset policy, dispersion and uncertainty rule. Provisional planning budget: five pairs per cell, extend once to ten if the predeclared uncertainty rule is inconclusive, otherwise report `inconclusive`; calibrate with the pilot before beginning acceptance and do not selectively rerun winning cells.
6. Separate cold, warm, prefill, decode, and end-to-end metrics.
7. Disable instrumentation for acceptance throughput.
8. Preserve raw logs and machine-readable JSON.
9. Benchmark arms sequentially on the same GPU, never concurrently. Record power mode, temperatures/clocks where available, competing GPU load and driver state; reject contaminated runs under a predeclared rule rather than cherry-picking them.
10. Define timing intervals exactly: process/model load, prefill/TTFT, time per output token/decode, and request end-to-end. “Cold” must name what is reset (process, model, graph, prefix/KV, OS cache); a cold label on a decode timer is not a load-latency measurement.

Exit gate:

- Exactly five models completed.
- Every model has complete cold and warm samples, actual token counts and declared uncertainty; near-threshold results that cannot establish the decision are `inconclusive`, not rounded into acceptance.
- The comparator and release gate agree programmatically.
- Every iteration reference is tied to a frozen Titan source/binary/config identity including dirty content hashes; final RC acceptance reruns against the final selected identity. This avoids requiring a final accepted diff before the experiments needed to select it.

Current status: a September 2 reference and failed release artifact exist; no new comparable baseline was executed in this review. Freeze an iteration identity for the next run, then repeat on the final selected RC identity.

**Producer audit prerequisite:** `engine/engine-server/tests/multi_model_comparison_bench.rs:807-809` ignores the result of Graph capture, `:835-852` performs a first decode before starting its decode timer, and `:917` labels cold/warm by prompt index. Establish actual Graph activation, decode position/token accounting and equal timing semantics for both engines before accepting these labels or ratios. A Graph request is not proof Graph replay occurred. Preserve old artifacts unchanged; if corrected boundaries change the metric, bump the measurement-contract version and regenerate both sides.

### F8.R3 — Complete runtime attribution

Objectives:

1. Retain the verified five-model FFN attribution.
2. Retain the verified five-model projection attribution.
3. Synchronize their OpenSpec task files with actual artifacts.
4. Preserve finite/non-negative checks.
5. Preserve group-sum reconciliation and residual reporting.
6. Keep `ffn_sync` explicitly `not_available` when no real frontier exists.
7. Never infer stage timing from launch counts.

Exit gate:

- Every claimed group maps to a real forward-driver boundary.
- Every duration is tied to a valid CUDA-event pair.
- Unmeasurable groups are explicitly unavailable.

Current status: measured groups historically verified; `ffn_sync` intentionally unavailable (not a task that must be made measurable to release); documentation synchronization pending. Reconcile stage coverage against total latency on the new baseline, quantify instrumentation overhead, and distinguish measured device intervals from host/wait gaps. Nsight Systems/host tracing may inform scheduling hypotheses where available, but does not establish Nsight Compute hardware counters.

### F8.R4 — Select one evidence-based optimization hypothesis

Objectives:

1. Start only after correctness boundaries are stable.
2. Select one measured group and one compatible shape/model scope.
3. Declare expected improvement and rejection threshold.
4. Define a narrow CPU reference and GPU parity contract.
5. Avoid reactivating rejected candidates.
6. Use per-model Amdahl/traffic bounds on the total relevant latency, not a stage-only percentage. State why the change could matter, expected fallback behavior and what observation would falsify the hypothesis.
7. Limit work in progress to one GPU candidate. First use an affected-shape sentinel plus a regression sentinel (historically Llama 3B and Qwen3 are informative); run the full matrix only after focused correctness and dispatch pass. Sentinel results are not release coverage.
8. After two candidate cycles without a reproducible material gain, stop the current hypothesis family and replan from evidence. This is a proposed effort guardrail, not a measured schedule. Do not change thresholds or reopen rejected ideas to avoid the stop rule.

Candidate priority, conditional on fresh evidence:

1. Audit broad execution-path overhead and the retained FFN Gate/Up opportunity against fresh total-decode attribution; neither automatically outranks the other from an FFN-internal share alone.
2. FFN Down if Q6_K evidence identifies a valid shape-specific opportunity.
3. QKV only after the first-decode RCA is resolved.
4. GEMV/GEMM shape, register, alignment, or memory-traffic changes.
5. Attention/KV only if attribution proves material impact.
6. Graph, synchronization, allocation, or copy work only if attributed by valid evidence.

Current decision: do not start a new blind kernel variation now.

### F8.R5 — Execute one atomic candidate cycle

For every candidate:

1. Write or verify the independent CPU reference.
2. Write the RED contract test.
3. Run the RED test and confirm it fails for the intended reason.
4. Implement the smallest change through `coder`.
5. Run GREEN compile, test, format, and Clippy checks.
6. Run focused GPU parity.
7. Observe the actual runtime kernel symbol.
8. Run control and candidate under identical conditions.
9. Run the five-model matrix.
10. Apply the candidate-relative and per-model/condition regression gates, including full-driver correctness and supported-mode fallback safety.
11. Report the absolute release gate independently; it is not a prerequisite for retaining a safe incremental experimental candidate.
12. Emit a machine-readable decision.
13. Update OpenSpec and documentation from actual output only.

Acceptance requirements:

- Applicable operation parity: relative L2 `< 1e-4`.
- Applicable operation cosine: `> 0.9999`.
- Finite output and safe memory behavior.
- No model regression greater than 5%.
- Titan/reference ratio `>=0.95x` per model and in aggregate for release acceptance.
- No default promotion during experimentation. A separately reviewed integration may combine accepted increments once full production correctness and regression gates pass; the project still remains `release_blocked` until the independent absolute gate passes. Do not require each increment to solve the entire deficit, and do not automatically promote the currently retained FFN candidate.
- RED tests for a new optimization must fail because the candidate/contract/dispatch is absent or wrong, not because the existing correct control is deliberately broken. Performance hypotheses use measured acceptance, not a fabricated deterministic RED speed test.

### F8.R6 — Nsight profiling

Run separately when the elevated Windows/UAC path is authorized and the permission blocker is removed.

This review excludes Nsight Compute counters as a mandatory release artifact unless a selected optimization/safety claim explicitly depends on them. Existing `ERR_NVGPUCTRPERM` remains a historical environmental blocker, not a fresh permission probe. A counter-dependent hypothesis stays blocked; independent correctness and wall-clock benchmarking may proceed. Do not relax OS/driver security settings without a scoped authorization; launch elevated tooling through the existing runner when authorized.

Procedure:

1. Use one small Titan-only workload.
2. Filter by kernel symbols observed in runtime telemetry.
3. Capture `.ncu-rep` and imported CSV.
4. Verify that the target process was actually profiled.
5. Reject outputs containing `No kernels were profiled`.
6. Keep Titan-only profiling separate from llama.cpp comparison.

Valid statuses:

- `verified`: valid counters and target capture.
- `partial`: only one side captured.
- `blocked`: `ERR_NVGPUCTRPERM`.

No occupancy, register, transaction, or Tensor Core claim is allowed while this gate is blocked.

### F8.R7 — Release gate

All conditions are mandatory:

- F32 production path correct.
- Q8 explicitly separated as experimental.
- QKV candidate resolved or excluded with verified runtime isolation and an explicit unresolved-cause ledger.
- Five models completed.
- Per-model ratio `>=0.95x`.
- Aggregate ratio `>=0.95x`.
- No regression greater than 5% against an explicitly compatible, complete Titan control baseline in any required model/condition; missing/incompatible comparison is not a pass.
- CPU and required GPU gates green.
- Graph parity and VRAM budget verified.
- API/CLI/E2E verified.
- Documentation and OpenSpec synchronized.
- Raw logs, hashes, environment, and decision artifacts preserved.
- Full diff reviewed and validator fail-open gaps closed.
- Minimum advertised generation/serving safety contract from `R1.P` verified.

If any condition fails:

```text
release_blocked
```

### F8.R8 — Local release candidate

Objectives:

1. Review the complete diff by architectural plane.
2. Separate production code, tests, documentation, artifacts, and plans.
3. Remove only confirmed unused experimental code, never unique evidence.
4. Run final software gates.
5. Execute required ignored tests explicitly.
6. Run local Markdown link checks.
7. Generate an RC manifest with hashes and gate statuses.
8. Prepare a local commit only after explicit authorization.
9. Stop before push unless Bernie explicitly orders push.

RC acceptance requires correctness, performance, evidence, documentation, and diff review to be green simultaneously. Run the final gates against the artifact actually packaged; any code, embedded kernel, build flag, numerical setting or dependency change after acceptance invalidates the affected evidence. Documentation-only edits do not imply new GPU measurements, but must be reflected in the manifest. Commit/push authorization is a publication permission, not numerical evidence.

---

## 7.3 CUDA Rust research track

This is an isolated research track. It must not bypass F8 or modify Titan production dispatch.

### T-R0 — Prepare the Linux/WSL laboratory

Previously recorded environment (not re-probed in this planning review; verify before installation):

- WSL2 Ubuntu exists.
- GPU visibility is available from WSL2.
- `nvcc`, Rust, `rustup`, and Clang are not currently installed inside WSL2.
- Windows has CUDA Toolkit 13.3.73 and the RTX 3060 Laptop GPU is compute capability 8.6.
- Current Titan uses `cudarc`, NVRTC, embedded `.cu` sources, and manual Driver API launch.

Required outcome:

- Reproducible WSL/Linux toolchain.
- Pinned Rust and CUDA versions.
- A separate artifact directory.
- No modification to production Titan dispatch.

### T-R1 — First CUDA Rust experiment with `cutile-rs`

Compatibility reviewed against primary sources on 2026-09-09: NVIDIA documents Linux, compute capability `8.0+`, stable Rust `1.89+`, and CUDA `13.3` for the recommended Tile path. The repository says Ampere `sm_8x` support arrived in CUDA `13.2`; use a pinned `13.3` laboratory configuration rather than assuming the Windows installation supplies WSL tooling. Both CUDA Rust projects are explicitly early-stage and not production-ready. RTX 3060 `sm_86` falls within the documented architecture range, but actual WSL compilation, execution and loader interoperability remain **not exercised**.

Primary sources:

- [NVIDIA CUDA Rust announcement](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/).
- [NVlabs cutile-rs requirements and project status](https://github.com/NVlabs/cutile-rs).
- [cuda-oxide ecosystem and cudarc interoperability](https://nvlabs.github.io/cuda-oxide/appendix/ecosystem.html).

Start with vector addition to prove toolchain/device execution, then one low-risk tiled operation:

- RMSNorm;
- RoPE;
- elementwise operations;
- isolated SwiGLU components.

Do not begin with:

- QKV multi-row;
- Q4_K/Q6_K dequantized kernels;
- production dispatch;
- a full runtime migration.

Gate:

- Rust kernel compiles.
- Kernel executes on the RTX 3060.
- Output is finite and matches an independent CPU reference.
- Generated artifact and launch metadata are preserved. `cutile-rs` JITs captured AST through Tile IR to a CUBIN; do not assume a standalone PTX export or a stable external ABI is automatically available.

### T-R2 — Bridge compatible artifacts through the existing loader

Validate whether the current `cudarc`/loader path can consume the generated artifact without adopting the entire NVIDIA Rust host/runtime ecosystem.

Required checks:

- ABI compatibility.
- Module loading.
- Symbol lookup.
- Launch parameters.
- Stream behavior.
- Synchronization behavior.
- CPU/GPU parity.
- Production dispatch remains unchanged.

### T-R3 — Evaluate `cuda-oxide`

Optional after Tile evaluation, not dependent on Tile succeeding. A Tile-specific incompatibility may justify a separately scoped SIMT spike rather than making it unreachable:

- Test SIMT control.
- Test shared memory and warp-level operations.
- Compare generated PTX to the existing CUDA C++ implementation.
- Measure only with a clean, isolated benchmark.

Because `cuda-oxide` requires a nightly-oriented toolchain while Titan’s constitution requires stable Rust, this track requires explicit governance review before it becomes a project dependency.

Use one isolated setup attempt and one small operator as the initial research budget; stop on unsupported toolchain/export/ABI rather than migrating Titan to force a result. No speedup forecast may be imported from B200/RTX 5090 examples to the RTX 3060. The primary-source interoperability statement supports evaluating `cuda-oxide` PTX with `cudarc`, not claiming Titan's existing loader has already accepted it.

### T-R4 — CUDA Rust adoption decision

Possible outcomes:

- `validated_sidecar`: useful for selected kernels without replacing the runtime.
- `partial`: useful only for Linux/build-time experiments.
- `rejected`: toolchain/API/performance cost does not justify adoption.
- `blocked`: unsupported Windows or loader integration.

CUDA Rust is not accepted for production until it passes the same parity, dispatch, performance, and release gates as a CUDA C++/NVRTC candidate.

---

## 7.4 Post-release scaling — F9

F9 begins only after F8 is green or the project formally redefines its release objective.

### F9.1 — Hardware portability

- Ampere, Ada, and Blackwell validation.
- Compute-capability dispatch matrix.
- Shape-specific policies.
- RTX 3060-specific versus general optimizations.
- Clear incompatibility messages.

### F9.2 — Larger models and MoE

- Expert streaming.
- Resident expert cache.
- PCIe fetch measurement.
- CPU overflow path.
- Fetch fraction and hit-rate accounting.
- VRAM budget planner.

### F9.3 — Multi-GPU

Only after single-GPU correctness, memory, and observability are stable:

- Layer and tensor partitioning.
- Inter-device communication.
- Synchronization.
- Per-GPU memory budgets.
- Real batch and latency benchmarks.

### F9.4 — Production operations

Advanced operational maturity only. Minimum bounded ingress, request isolation, cleanup, invalid-input and memory/error behavior needed for the advertised local server move to `F8.R1.P`.

- Versioned packages.
- Windows and Linux installation guides.
- Health/readiness endpoints.
- Exportable metrics.
- Backpressure.
- Cancellation.
- OOM recovery.
- CUDA error recovery.
- Concurrency limits.

### F9.5 — Generation quality

Separate from throughput. The minimum advertised generation/tool/JSON contract moves to `F8.R1.P`; broader model/quality evaluation remains here:

- Deterministic greedy outputs.
- Logit and token-ID comparisons.
- Stop sequences.
- Truncation.
- Tool-calling correctness.
- JSON Schema correctness.
- Model-specific quality checks.

---

## 8. Decision rules for all future work

1. Do not advance a dependent production path past its failed correctness gate. A quarantined, non-dispatched experiment does not create an unrelated global dependency.
2. Do not use a passing isolated parity test to claim driver correctness.
3. Do not use an instrumented timing run as acceptance throughput.
4. Do not infer hardware counters without valid Nsight evidence.
5. Do not treat a registered or compiled kernel as a dispatched kernel.
6. Do not promote a candidate from a worker report alone; inspect the diff, artifact, and decisive command.
7. Do not check a blocked or rejected task merely to improve OpenSpec percentages.
8. Do not repeat rejected candidates without new causal evidence.
9. Do not modify production defaults while an experimental candidate is under evaluation.
10. Do not use CUDA Rust to bypass an unresolved numerical RCA.
11. Do not modify the constitution without Bernie’s explicit approval.
12. Do not commit or push without explicit authorization.

---

## 9. Immediate next gate

The smallest technically meaningful next step is:

> **Freeze the evidence/source identity and execute `F8.R0.5`: reproduce and close the acceptance-validator fail-open cases through a narrowly scoped `coder` change. In parallel planning, define the `R1.P` production envelope and candidate isolation tests.**

The first implementation handoff must:

1. Recheck Git identity and inventory; preserve existing edits, artifacts and the immutable constitution.
2. Create one scoped OpenSpec proposal/tasks through the implementation workflow, naming `tools/release_gate.py` and its tests, explicit non-goals (no kernel/default changes, no benchmark rescue), and a versioned evidence contract.
3. Add the reproduced negative tests, demonstrate intended RED results, fix the validator through `coder`, and run the complete tools suite. Inspect related comparators for the same class of failure.
4. Independently review the diff and replay both valid fixtures and malformed fixtures. Do not claim a synthetic fixture is a real-model benchmark.
5. Synchronize that milestone from actual results, then execute the chosen production correctness/isolation gate and comparable baseline before choosing another performance candidate.

The outcome is `validator_verified`, `failed`, or `blocked/incomplete`. This review establishes the gap but does not implement the repair. Optional QKV RCA follows section 4.4 only if its value is justified; quarantine is an acceptable branch after isolation is verified.

### Resource and viability checkpoints

- One coding change and one GPU benchmark owner at a time; read-only review can run in parallel. Do not mix profiler and acceptance workload scheduling on the laptop GPU.
- Re-estimate effort after validator repair and after the new baseline. No calendar completion date is supportable while the comparable gap and causal opportunity are unknown.
- If bounds and two bounded candidate cycles do not support material progress, publish a `competitive_target_unproven` checkpoint and choose: new evidence-backed architecture hypothesis, defer competitive release, or explicitly propose a different product target. Do not silently weaken `0.95x` or relabel an experimental snapshot as a competitive RC.
- The constitution's MoE expert-streaming use case remains strategically relevant, but moving it onto the critical path or changing the release objective requires an explicit follow-up decision. This plan does not buy hardware, migrate runtimes, amend the constitution, or authorize commit/push.

---

## 10. Release definition

Titan may be called release-ready only when all of the following are simultaneously true:

- Production F32 correctness is verified.
- Q8 status is explicit and does not contaminate the F32 claim.
- All promoted kernels have independent parity and observed dispatch evidence.
- Five-model benchmark coverage is complete and comparable.
- Titan/reference ratio is at least `0.95x` per model and in aggregate.
- No model regresses by more than 5%.
- Required GPU, Graph, VRAM, API, and E2E gates pass.
- Counter-dependent claims require verified profiling; Nsight Compute counters are otherwise explicitly excluded from the mandatory release criteria in `R6`.
- OpenSpec, README, architecture, testing, benchmark, and workspace documentation agree with artifacts.
- The dirty diff has been reviewed and isolated.
- The RC manifest is reproducible.
- Packaging identity matches the verified manifest; commit/push remain separately authorized publication actions and do not substitute for engineering acceptance.

Until then, the correct project status remains:

```text
release_blocked
```

---

## 11. Primary evidence index

### Repository and plans

- `docs/WORKSPACE_STATE.md`
- `docs/ARCHITECTURE.md`
- `docs/BENCHMARKS.md`
- `docs/TESTING.md`
- `.hermes/plans/2026-09-02_183017-titan-next-roadmap.md`
- `.hermes/plans/2026-09-07_115609-titan-ffn-attribution-roadmap.md`
- `.hermes/plans/2026-09-07_220657-titan-projection-attribution-release.md`
- `.hermes/plans/2026-09-08_205336-titan-viabilidad-rca-coder.md`

### QKV evidence

- `local-artifacts/reviews/phase18-qkv-multi-row-batch1-decision-20260908.json`
- `local-artifacts/reviews/qkv-first-decode-rca-decision-20260909T122609Z.json`
- `local-artifacts/benchmarks/real-f32-qkv-multi-row-batch1-paired-diagnostic-1788956770585-6596-0.json`
- `local-artifacts/reviews/real-f32-qkv-multi-row-batch1-paired-diagnostic-1788956770585-6596-0.log`
- `openspec/changes/2026-09-08-qkv-multi-row-batch1-candidate/`
- `openspec/changes/2026-09-08-qkv-first-decode-rca/`

### Attribution evidence

- `local-artifacts/reviews/five-model-ffn-attribution-matrix-1788823934726-3700-0.json`
- `local-artifacts/reviews/five-model-ffn-attribution-matrix-1788823934726-3700-0.log`
- `local-artifacts/reviews/five-model-projection-attribution-control-1788830924265-9040-0.json`
- `local-artifacts/reviews/phase17-projection-attribution-decision-20260908.json`
- `openspec/changes/2026-09-07-ffn-grouped-attribution-boundary/`
- `openspec/changes/2026-09-07-five-model-ffn-attribution-matrix/`
- `openspec/changes/2026-09-08-projection-attribution-boundary/`

### Performance evidence

- `local-artifacts/reviews/fresh-head-to-head-release-gate-20260902.json`
- `local-artifacts/reviews/phase15-q4k-gate-up-multi-shape-decision-20260907.json`
- `local-artifacts/reviews/phase9-q4k-gate-up-single-row-decision-20260907.json`
- `local-artifacts/reviews/phase4-f32-single-row-q6k-decision-20260905.json`
- `local-artifacts/benchmarks/final-head-to-head-20260902_165459.json`

### Final software gates

- `local-artifacts/reviews/qkv-rca-final-gates-20260909T124824Z.log`

---

## Final status

### Feasibility-review execution record

Performed during this revision:

- Read the roadmap, release/RCA evidence, validator/test code, and targeted architecture/contracts; independent read-only reviewers inspected numerical and performance concerns.
- Live Git: HEAD unchanged; 13 tracked modified paths and 37 untracked status entries. No checkout cleanup, source edits, commit or push.
- Live `openspec validate --all`: **26 passed, 0 failed**. Programmatic active-task counts: **222 checked, 30 unchecked, 252 total**; these remain mechanical counts.
- Existing tools suite: `PYTHONDONTWRITEBYTECODE=1 uv run --project tools --no-sync python -B -m unittest discover -s tools -p 'test_*.py' -v`: **13 tests passed**, exit 0. This is unittest-discovered coverage, not a claim that every tools test style was run.
- Real validator CLI probes using isolated synthetic fixtures: all five malformed/incompatible cases in `R0.5` returned `accepted` with empty `failed_gates`. This is evidence of validation gaps, not fabricated production output and not a repaired validator.
- All 28 distinct local document/artifact references extracted from the original roadmap existed. Existence does not independently validate every historical claim.
- NVIDIA/NVlabs primary sources checked for CUDA Rust toolchain/architecture requirements; no installation or kernel experiment performed.
- Independent review reconciled against source: rejected the claim that default F32 QKV consumes Q8_1 because it cited the Q8 function rather than the F32 dispatch; retained the independently confirmed missing activation copy-out, benchmark timing/Graph-label hazards, and silent E2E fixture skip. Reviewer recommendations were not accepted as facts without checking their call sites.

Not performed: fresh Rust workspace build/tests, ignored GPU tests, new five-model throughput runs, Nsight capture, final production correctness matrix, RC acceptance, OpenSpec task-state edits, or validator repairs. Historical results above retain their original timestamps and scope. Only this roadmap is edited by the review.

```text
Project maturity: functional experimental engine
Correctness default: F32
Q8: experimental / not accepted as full-driver production path
QKV multi-row candidate: rejected_regression / not promoted
FFN multi-shape candidate: accepted_opt_in / not promoted
Performance release gate: rejected
FFN attribution: verified for measured groups
Projection attribution: verified for measured groups
ffn_sync: not_available / missing_real_frontier
Nsight: blocked / ERR_NVGPUCTRPERM
OpenSpec validation: 26 passed / 0 failed
Working tree: dirty and preserved
Release: blocked
Acceptance validator: fail-open cases reproduced / repair pending
QKV RCA: optional after release-path isolation / cause unresolved
Competitive 0.95x feasibility: unproven / quantified gap, no delivery promise
Next executable gate: F8.R0 + F8.R0.5, then production envelope verification
```
