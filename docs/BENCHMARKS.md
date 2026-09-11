# 📊 Titan — Performance Benchmarks & Empirical Evaluation

> **Stable Functional MVP Contract:** [`docs/MVP.md`](MVP.md)
> **Workspace State & Evidence:** [`docs/WORKSPACE_STATE.md`](WORKSPACE_STATE.md)
> **Strict Production Envelope:** [`docs/RELEASE_ENVELOPE_V1.json`](RELEASE_ENVELOPE_V1.json)

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
> **Performance Parity Boundary:**
> * A throughput ratio of $\ge 0.95\times$ compared to `llama.cpp` is **NOT** an acceptance criterion for the Stable Functional MVP. The $0.95\times$ threshold is exclusively a strict production-performance diagnostic enforced by `tools/release_gate.py`.
> * In the functional MVP evaluation, throughput ratios (aggregate median `0.5286115177544821`, cold `0.5758x`, warm `0.4981x`) and baseline regressions are treated as **advisory warnings** (`ratio_advisory: advisory_warning`, `regression_advisory: failed`). They do not reject the MVP.
> * Conversely, **generation correctness, operational E2E verification, and clean benchmark execution safety remain strictly blocking**.
>
> **Exact MVP Blockers:**
> 1. **`build_binding_unverified`:** The F8 correctness execution verified model numerical accuracy across 5 models, but does not record the Titan binary/source identity matching the clean benchmark's `engine_identity.titan` build hash (`sha256:bc92211356bb0eb87ddd59e252d9d4497f8669d7257ac0133459040ce4eb131c`).
> 2. **`model_hash_missing`:** The final clean benchmark artifact (`local-artifacts/benchmarks/p5-final-clean-20260911T151844Z.json`, 30 rows) records model names but lacks per-result GGUF weight file hashes required for exact binding to F8.
>
> **Evidence Taxonomy & State Classification:**
> * **Verified:** F8.R1.P generation correctness (5 models, F32, batch 1, resident KV, Graph=false, candidate selectors absent); operational E2E (Resident, Streaming SSE with N-gram, grammar JSON parsed as `{"city": "Tokyo"}`); clean benchmark structure (30 rows, 41 tokens/sample); clean telemetry; selector provenance; MVP gate tests (25); current full Python suite (88 tests + 8 subtests; the earlier MVP execution record reported 84); OpenSpec (40/40).
> * **Blocked:** F8-to-Titan-build binding; benchmark per-result model hashes; MVP terminal decision (`mvp_blocked_evidence`); strict release gate (`rejected`); Nsight profiling (`ERR_NVGPUCTRPERM`).
> * **Advisory:** Titan/llama.cpp throughput ratios (aggregate `0.5286115177544821`); per-model ratios; baseline regression status.
> * **Deferred:** `ffn_sync` full-loop attribution (`missing_real_frontier`); asymmetric reference stage coverage; candidate kernel optimizations; Q8 production integration; RC generation.
> * **Not Exercised:** Fresh GPU benchmark rerun (not justified; existing blockers are schema/binding gaps, not transient execution failures).
>
> **Local-Only Evidence & Publishing Boundary:**
> `local-artifacts/`, `.hermes/`, `models/`, and `target/` are local runtime assets and not publishable repository content. Historical benchmark tables below are preserved for provenance and are not current release claims.

---

## Current Release & Hardening Status

The current benchmark evidence does not grant release sign-off:

- **Strict Five-Model Release Gate:** `local-artifacts/reviews/p5-release-gate-20260912T130000Z.json`, status `rejected`, aggregate ratio `0.5286115177544821` (cold `0.5758494419340426`, warm `0.4980834957173451`); the required $\ge 0.95\times$ per-model and aggregate gates are not met, and regression failed.
- **F8 Correctness Verification:** `local-artifacts/reviews/f32-five-model-correctness-matrix-1789097311038-15004.json` independently verified 5/5 models on F32, batch 1, resident KV, Graph=false, candidate selectors absent; `production_dispatch_changed = false`.
- **Clean Baseline Benchmark:** `local-artifacts/benchmarks/p5-final-clean-20260911T151844Z.json` contains 30 valid rows (5 models × 3 repetitions × cold/warm), 41 generated tokens, observed Graph disabled, and clean telemetry.
- **FFN Attribution:** Grouped `ffn_gate_up` and `ffn_down` timing is verified; `ffn_sync` is explicitly `not_available/missing_real_frontier` as no independent stream dependency frontier exists.
- **Projection Attribution:** Verified 5/5 groups in `local-artifacts/reviews/phase17-projection-attribution-decision-20260908.json`, confirming QKV as the largest non-FFN projection group (`1.534976 ms` median).
- **QKV Multi-Row Batch 1 Candidate:** Decisively rejected as a regression (`local-artifacts/reviews/phase18-qkv-multi-row-batch1-decision-20260908.json`); production dispatch unchanged (`production_dispatch_changed = false`).
- **Hardware Profiling:** Nsight profiling remains blocked by `ERR_NVGPUCTRPERM`.

Instrumentation-enabled runs are attribution evidence and must not be compared as clean throughput acceptance runs. Preserve all JSON and raw logs under `local-artifacts/`; do not overwrite historical checkpoints.

## Candidate gate checkpoint — 2026-09-07

The new `q4k_gate_up_single_row` candidate was evaluated against a control under identical no-Graph conditions:

- Control: `local-artifacts/benchmarks/phase9-five-model-control-f32-nograph-20260907.json`.
- Candidate: `local-artifacts/benchmarks/phase9-five-model-candidate-q4k-single-row-f32-nograph-20260907.json`.
- Matrix: 5 models, 3 repetitions, 41 generated tokens, temperature 0, `cuda_graphs=false`.
- Candidate-vs-control result: `rejected_regression`; Qwen3 0.6B warm delta `-5.112480663017738%`, exceeding the `5%` limit.
- Candidate-vs-reference aggregate mean ratio: `0.46204798109421563`, below the required `0.95`.
- Decision: `local-artifacts/reviews/phase9-q4k-gate-up-single-row-decision-20260907.json`.

The candidate remains opt-in and did not change production dispatch. These measurements are not release approval; the baseline release gate remains rejected.

The follow-up RCA artifact `local-artifacts/reviews/phase10-qwen3-candidate-causal-rca-20260907.json` shows that Qwen3 never dispatched `gemm_q4k_f32_single_row_kernel`; the candidate is scoped to Llama 3B dimensions. Therefore the formal cross-model rejection is preserved, but its Qwen3 regression must not be described as causal kernel evidence.

The paired eligible-shape diagnostic `local-artifacts/reviews/real-f32-q4k-gate-up-paired-diagnostic-1788815893628-17936-0.json` passed three alternating pairs with identical tokens and candidate dispatch in every candidate arm. Its mean decode delta was `-5.822233333333315 ms` in favor of the candidate. This is causal diagnostic evidence only and does not override the formal five-model rejection.

## Qwen3 shape-specific candidate — 2026-09-07

- Candidate: `qwen3_q4k_single_row`; exact selector `TITAN_F32_FFN_GATE_UP_VARIANT=qwen3_q4k_single_row`; shape `1024 x 3072`.
- Parity artifact: `local-artifacts/reviews/real-f32-qwen3-q4k-gate-up-single-row-parity.json`.
- Dispatch artifact: `local-artifacts/reviews/real-f32-qwen3-q4k-gate-up-dispatch-smoke-1788821034798-16636-0.json`; exact kernel `gemm_q4k_f32_single_row_kernel`, 56 launches, fused/legacy absent.
- Fresh control/candidate matrix: 5 models × 3 repetitions, 41 tokens, temperature 0, `cuda_graphs=false`; control and candidate artifacts are `local-artifacts/benchmarks/phase13-five-model-control-qwen3-q4k-f32-nograph-20260907.json` and `local-artifacts/benchmarks/phase13-five-model-candidate-qwen3-q4k-f32-nograph-20260907.json`.
- Candidate-relative verdict: `accepted`; worst delta `-1.2729686130771078%`, Qwen3 warm `+1.726895992021782%`.
- Production/release verdict: `not_promoted` / `rejected`; candidate/reference mean ratio `0.4637685580716343` remains below `0.95`.

## Multi-shape Gate/Up candidate — 2026-09-07

- Attribution matrix: `../local-artifacts/reviews/five-model-ffn-attribution-matrix-1788823934726-3700-0.json`; Gate/Up was the largest verified group in all five models, with median share `64.80421322699188%`.
- Candidate: `q4k_multi_shape_single_row`; selector `TITAN_F32_FFN_GATE_UP_VARIANT=q4k_multi_shape_single_row`; allowlisted shapes `1024x3072`, `1536x8960`, `2048x8192`, `3072x8192`.
- Real parity: `../local-artifacts/reviews/real-f32-q4k-gate-up-multi-shape-single-row-parity.json`, 5/5 fixtures passed; maximum relative L2 `1.1331776146647264e-6`.
- Real dispatch: `../local-artifacts/reviews/real-f32-q4k-gate-up-multi-shape-dispatch-smoke-1788826168653-30052-0.json`, 5/5 fixtures observed `gemm_q4k_f32_single_row_kernel`; default control remained default.
- Fresh control/candidate benchmarks: `../local-artifacts/benchmarks/phase15-five-model-control-q4k-multi-shape-f32-nograph-20260907.json` and `../local-artifacts/benchmarks/phase15-five-model-candidate-q4k-multi-shape-f32-nograph-20260907.json`.
- Relative candidate verdict: `accepted_opt_in`; every one of 10 median candidate/control cells improved, minimum `+2.612874495571482%`.
- Absolute release verdict: `rejected`; candidate/reference mean is `0.5151486550565579x` cold and `0.4729248466771503x` warm, below `0.95x`. No default promotion is allowed.

## QKV Multi-Row Batch=1 candidate — 2026-09-08

- Predecessor projection attribution control: `../local-artifacts/reviews/five-model-projection-attribution-control-1788838534847-10564-0.json` and decision `../local-artifacts/reviews/phase17-projection-attribution-decision-20260908.json` verified 5/5 groups; QKV confirmed as largest non-FFN projection group (`1.534976 ms` median).
- Candidate: `q4k_multi_row_batch1`; selector `TITAN_F32_QKV_VARIANT=q4k_multi_row_batch1`; kernel `gemm_fused_qkv_multi_row_kernel`.
- Scope: F32 input, `batch_size = 1`, `graph = false`, resident KV (`cache_condition = "resident"`), weight formats `Q4_K/Q4_K/Q6_K`, five fixtures.
- Real parity: `../local-artifacts/reviews/real-f32-qkv-multi-row-batch1-parity-1788845218784-10980-0.json`; **5/5 strict Q/K/V parity** verified against independent CPU reference calculations (`forward_cpu::matmul`), with `relative_l2 < 1e-4` and `cosine > 0.9999` across Q, K, and V on all five models.
- Real dispatch smoke: `../local-artifacts/reviews/real-f32-qkv-multi-row-batch1-dispatch-smoke-1788846368939-35648-0.json`; **5/5 actual `gemm_fused_qkv_multi_row_kernel` observations** under candidate arm with default control under control arm; fused Gate/Up, Q6K single-row, and Q8 verified absent.
- Fresh five-model benchmark runs:
  - Control: `../local-artifacts/benchmarks/phase18-five-model-control-qkv-batch1-f32-nograph-20260908.json`
  - Candidate: `../local-artifacts/benchmarks/phase18-five-model-candidate-qkv-batch1-f32-nograph-20260908.json`
  - Execution parameters: 5 models, 3 repetitions, 41 generated tokens, temperature 0, `cuda_graphs = false`.
  - **Pairing Status**: These are separate non-paired runs; same-run pairing was NOT satisfied. Do not call them a clean benchmark or performance acceptance.
- Authoritative Decision: `../local-artifacts/reviews/phase18-qkv-multi-row-batch1-decision-20260908.json`
  - `candidate_status`: `rejected_regression`
  - `production_status`: `not_promoted`
  - `release_gate_status`: `rejected`
  - `production_dispatch_changed`: `false`
  - `ffn_sync`: `not_available/missing_real_frontier`
  - `nsight`: `blocked/ERR_NVGPUCTRPERM`
- Exact Gate Metrics:
  - Relative Regression Gate: FAILED (required min delta >= -5.0%)
    - `min_delta_pct`: `-55.634076991989666%` (DeepSeek cold)
    - `max_delta_pct`: `186.9843315262684%` (Qwen3 cold)
    - `median_cold_delta_pct`: `-9.080238424479903%`
    - `median_warm_delta_pct`: `45.24988926281488%`
  - Absolute Release Gate: FAILED (required ratio >= 0.95x)
    - Candidate/reference mean cold: `0.44780836339879965x` (min `0.15281763273442298x`)
    - Candidate/reference mean warm: `0.5849578722483578x` (min `0.3931167772903779x`)
- Decision Rationale: The formal decision is a decisive rejection despite high run-to-run variability because the observed relative failures (-55.63% on DeepSeek cold, -44.87% on Llama 3B warm, -24.44% on Qwen 1.5B cold) are far beyond the -5% threshold. The phase18 decision did not authorize a release rescue; the later paired diagnostic is recorded below and is not acceptance. Production dispatch remains strictly unchanged (`production_dispatch_changed = false`).
- Preserved Baselines: `q4k_multi_shape_single_row` remains accepted opt-in only; Q6K single-row and fused Gate/Up remain rejected; Q8 experimental; Graph, defaults, and APIs unchanged.

### QKV paired diagnostic follow-up — 2026-09-08

- The same-invocation paired diagnostic harness was attempted but **executed_failed/incomplete** before the five-model matrix.
- Command: `cargo test -p engine-server --test real_f32_qkv_multi_row_batch1_paired_diagnostic -- --ignored --nocapture`; exit code `101`.
- Artifact: `../local-artifacts/benchmarks/real-f32-qkv-multi-row-batch1-paired-diagnostic-1788879532145-3500-0.json`.
  - `status = "token_sequence_mismatch"`
  - `diagnostic_only = true`
  - `acceptance_claimed = false`
  - `release_acceptance_evaluated = false`
  - `fixtures = []` because it stopped at the first fixture.
- Raw log: `../local-artifacts/reviews/real-f32-qkv-multi-row-batch1-paired-diagnostic-1788879532145-3500-0.log`.
  - Qwen 2.5 1.5B Pair 1 control tokens: `[100905, 3433, 2780, 28339, 22379]`.
  - Qwen 2.5 1.5B Pair 1 candidate tokens: `[100905, 99545, 147367, 64689, 138136]`.
  - The first token is equal, then the sequences diverge.
- This is not a passing acceptance benchmark and does not change the prior authoritative `rejected_regression` / `not_promoted` / release `rejected` decision. Do not infer whether the mismatch is an arithmetic bug or expected drift; RCA remains unresolved and requires a same-domain first-decode differential before any fix or promotion.

### QKV first-decode RCA diagnostic — 2026-09-09

The new run is diagnostic-only. It is not a throughput measurement, clean
benchmark, performance acceptance run, or release evaluation. Task 10 used:

```bash
cargo test --manifest-path engine/Cargo.toml -p engine-server --test real_f32_qkv_multi_row_batch1_paired_diagnostic real_f32_qkv_multi_row_batch1_paired_diagnostic -- --exact --ignored --nocapture --test-threads=1
```

The command exited `101` with `status = "token_sequence_mismatch"` after the
first fixture. Exact evidence paths are:

- Benchmark JSON: `../local-artifacts/benchmarks/real-f32-qkv-multi-row-batch1-paired-diagnostic-1788956770585-6596-0.json`
- Raw log: `../local-artifacts/reviews/real-f32-qkv-multi-row-batch1-paired-diagnostic-1788956770585-6596-0.log`
- Decision: `../local-artifacts/reviews/qkv-first-decode-rca-decision-20260909T122609Z.json`
- Baseline: `../local-artifacts/reviews/qkv-rca-task2-baseline-20260909T120345Z.log`

The artifact records schema `1.1.0`, `failed_pair` present, `fixtures = []`,
`diagnostic_only = true`, `acceptance_claimed = false`, and
`release_acceptance_evaluated = false`. The first-decode input is `100905` in
both arms; finite logits and diagnostic metrics were persisted. This evidence
does not make the engine correct or the candidate accepted.

The Phase 18 control and candidate artifacts remain separate non-paired runs,
and their authoritative rejection is preserved above. Neither those runs nor
this diagnostic should be called a clean benchmark.

---

## 1. Hardware & Environment Specifications

All benchmarks were executed on the following reference hardware and operating environment:

| Specification | Hardware / Environment Value |
| :--- | :--- |
| **GPU Model** | **NVIDIA GeForce RTX 3060 Laptop GPU** |
| **GPU Architecture** | Ampere (Compute Capability 8.6, SMs: 30) |
| **VRAM Capacity** | 6,144 MB GDDR6 (192-bit bus width) |
| **Theoretical Memory Bandwidth**| **336.0 GB/s** |
| **Host CPU** | AMD Ryzen 7 5800HS with Radeon Graphics (8 cores / 16 threads) |
| **Host RAM** | 40,351 MB DDR4 |
| **Host Operating System** | Microsoft Windows 11 (x86_64) |
| **CUDA Driver / Runtime** | CUDA 12.4 (NVRTC Driver API: `nvcuda.dll`) |
| **Rust Toolchain** | `rustc 1.85.0` (Edition 2024, `--release` profile) |

---

## 2. Historical Multi-Model Head-to-Head: Titan vs. llama.cpp (2026-09-01 checkpoint)

Fresh controlled run from the current checkout using the installed CUDA-enabled `llama-server.exe`, identical GGUF files, two prompts per model, 41 generated tokens, batch size = 1, greedy sampling ($T=0$), and three repetitions per model. The test completed with `1 passed, 0 failed` in 100.76 seconds.

Artifact: `../local-artifacts/benchmarks/rerun-20260901-085229.json`
Raw log: `../local-artifacts/benchmarks/rerun-20260901-085229.log`

| Model | Cold llama.cpp | Cold Titan | Cold ratio | Warm llama.cpp | Warm Titan | Warm ratio | Reported ratio |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 2.5 1.5B Instruct | 138.6 | 124.1 | 1.018x | 135.5 | 128.5 | 0.987x | **1.009x** |
| Llama 3.2 1B Instruct | 178.6 | 170.2 | 0.947x | 189.0 | 179.7 | 0.848x | **0.896x** |
| Llama 3.2 3B Instruct | 104.7 | 77.0 | 0.734x | 105.0 | 76.9 | 0.732x | **0.735x** |
| DeepSeek-R1-Distill 1.5B | 157.0 | 143.3 | 0.912x | 173.1 | 143.3 | 0.826x | **0.864x** |
| Qwen3 0.6B Base/Chat | 240.2 | 274.1 | 1.142x | 278.1 | 274.1 | 0.984x | **1.057x** |

The reported ratio is the artifact's average across prompt/cache statistics; raw cold and warm medians are shown separately. The simple mean of the five reported ratios is **0.912x**. This is a valid fresh checkpoint, not a release sign-off: the `>=0.95x` per-model and aggregate gates remain open, with Llama 3.2 3B as the principal deficit.

---

## 3. Cross-Engine Comparative Landscape

Historical comparison estimates from prior work are not comparable to the latest controlled head-to-head run. They are intentionally omitted here: the only current cross-engine measurements are the five llama.cpp comparisons above.

---

## 4. Kernel-Level Profiling & Latency Breakdown

The per-stage timing profile below is historical and has not been rerun as a symmetric cross-engine profile with the current checkpoint. It must not be used to derive the current head-to-head numbers.

| Pipeline Stage | Kernel Implementation | Execution Time | % Total Time |
| :--- | :--- | :--- | :--- |
| **Dynamic Activation Quant** | `quantize_row_q8_1_kernel` (shared mem / 128-bit `int4`) | 0.28 ms | 3.8% |
| **QKV Projection (Fused)** | `gemm_fused_qkv_kernel` (DP4A `__dp4a` + uint4 loads) | 1.62 ms | 22.1% |
| **RoPE & Paged KV Append** | `fused_qk_norm_rope_kernel` (fused in 1 launch) | 0.41 ms | 5.6% |
| **Paged Attention** | `paged_attention_kernel` (shared memory reductions) | 0.89 ms | 12.1% |
| **Output Projection (Wo)** | `gemm_q4k_kernel` + in-place residual addition | 1.15 ms | 15.7% |
| **SwiGLU Gate/Up (Fused)** | `gemm_q4k_fused_gate_up_swiglu_kernel` + silu | 1.48 ms | 20.2% |
| **Down Projection (Wdown)**| `gemm_q4k_kernel` + in-place residual addition | 1.18 ms | 16.1% |
| **LM Head & Sampling** | `lm_head_gemm` + GPU argmax reduction | 0.32 ms | 4.4% |
| **Total Forward Step** | **Captured Autonomous CUDA Graph** | **7.33 ms** | **100.0%** |

---

## 5. Multi-Model GPU Speculative Decoding Benchmark

Evaluating dual-model resident acceleration:
* **Draft Model ($M_1$):** Llama 3.2 1B Instruct (807 MB VRAM, decoding at 166.0 tok/s).
* **Target Model ($M_2$):** Llama 3.2 3B Instruct (2.02 GB VRAM, decoding at 70.2 tok/s).
* **Total VRAM Consumption:** **2.83 GB** (Fitting comfortably within 6GB VRAM).

```
================================================================================
===                   SPECULATIVE DECODING SPEEDUP SUMMARY                   ===
================================================================================
Target Model:           Llama 3.2 3B Instruct
Draft Model:            Llama 3.2 1B Instruct
Baseline 3B Throughput: 70.2 tok/s (14.24 ms/tok)
Speculative Throughput: 138.4 tok/s (7.22 ms/tok)
Candidate Window (K):   3 tokens
Effective Speedup:      1.97x Acceleration vs Target 3B Baseline
================================================================================
```

---

## 6. How to Reproduce All Benchmarks

### 1. Download Test GGUF Models
Place the following standard GGUF files in the `models/` directory:
- `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`
- `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`
- `models/Llama-3.2-3B-Instruct-Q4_K_M.gguf`
- `models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf`

### 2. Run Head-to-Head Test Suite
```bash
cd engine
cargo test --release -p engine-server --test multi_model_comparison_bench -- --ignored --nocapture
```

The suite resolves the five model paths under `models/` plus the Qwen3 fixture under `testdata/`, skips missing GGUFs, starts the installed CUDA `llama-server.exe`, and prints only the models actually benchmarked. On Windows, ensure the process can load the NVRTC DLL used by Titan.

### 3. Run Speculative Decoding Speedup Suite
```bash
cargo test --release -p engine-server --test speculative_speedup_bench -- --ignored --nocapture
```

## Stable Functional MVP Checkpoint — 2026-09-11

The current live run passes 88 full-suite tests plus 8 subtests. The earlier MVP execution record retained in the research log reported 84 tests; it is historical, not the current count. The decision artifact is `local-artifacts/reviews/mvp-decision-20260911T162326Z.json` and the blocker bundle is `local-artifacts/bundles/stable-functional-mvp-blocked-20260911T162326Z/`. See [`docs/MVP.md`](MVP.md) for full reproduction commands and contracts.

- **MVP Validator Tests:** 25 passed (`uv run --project tools --no-sync pytest tools/test_mvp_gate.py`); current full Python suite: 88 tests plus 8 subtests (the earlier MVP execution record reported 84); OpenSpec latest: 40/40 passed.
- **P5 Clean Benchmark & Operations:** P5 clean benchmark structure (30 rows, clean telemetry, selectors absent) and operational E2E cases (Resident, Streaming SSE with N-gram, grammar JSON parsed as `{"city": "Tokyo"}`) passed their sub-gates.
- **Advisory Warnings:** Titan/llama.cpp throughput ratios (aggregate median `0.5286115177544821`, cold `0.5758x`, warm `0.4981x`) and regression are advisory warnings in MVP; they do not reject the MVP.
- **Exact MVP Blockers:**
  1. `build_binding_unverified`: F8 correctness evidence does not record the Titan binary/source identity matching the P5 benchmark's `engine_identity.titan` build hash (`sha256:bc92211356bb0eb87ddd59e252d9d4497f8669d7257ac0133459040ce4eb131c`).
  2. `model_hash_missing`: The P5 clean benchmark artifact lacks per-result model hashes required for exact model-identity binding to F8.
- **Strict Release Gate:** Status `rejected` on performance and regression gates (`local-artifacts/reviews/p5-release-gate-20260912T130000Z.json`).
- **Production Status:** `promotion_authorized = false`, `rc_generated = false`, `production_dispatch_changed = false`.
