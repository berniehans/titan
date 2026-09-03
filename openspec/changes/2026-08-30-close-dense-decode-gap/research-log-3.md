# Research Log — Point 3 Optimization Experiments

Date: 2026-08-31
Project: Titan dense decode gap
Hardware: NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6
Working tree: intentionally dirty, no commit or push

## Purpose

This document records research experiments performed during Point 3. Each entry preserves the hypothesis, scope boundary, command/evidence, artifact, result, and decision. A research result is not automatically a release acceptance result.

## Experimental protocol

- One functional hypothesis per iteration.
- Preserve CPU/reference parity and existing APIs.
- Keep FFN, Q6_K, Norm/SwiGLU, attention, graph, synchronization, and KV-cache changes separate unless explicitly identified as a differential regression fix.
- Use three repetitions for throughput comparisons where possible.
- Treat historical baselines with different repetition counts or build states as provisional.
- Do not update final benchmark documentation with an unverified or incomplete result.

## Experiment register

### R3-FFN-001 — Cached RMSNorm/Q8_1 source row

- Hypothesis: caching the FFN gate/up activation row in shared memory avoids a second global-memory read during RMSNorm/Q8_1 quantization.
- Scope: `gemm_quant.cu`, `batched_gemm.rs`, `forward_driver.rs`, focused parity test.
- Artifact: `local-artifacts/benchmarks/ffn-iteration-full.json`.
- Workload: five models, three repetitions, llama.cpp comparison.
- Result: Llama 3.2 3B aggregate Titan throughput 73.312 tok/s; ratio 0.7044. Compared with the archived one-repetition baseline artifact, this was a provisional +16.20% throughput delta for 3B.
- Correctness: focused GPU parity passed; CPU synthetic forward passed; format and diff checks passed.
- Decision: **Accepted provisionally** as an isolated FFN iteration. No final project gate claimed.

### R3-Q6K-001 — Q6_K multi-row tile of four

- Hypothesis: reducing the Q6_K batched tile from eight to four rows reduces live accumulators/register pressure while preserving partial-batch correctness.
- Scope: `gemm_q6k_multi_row_kernel` and its host launch geometry only.
- Evidence: Nsight showed the previous Q6_K path at high register pressure; the tile-four implementation uses `batch_base = blockIdx.y * 4`, `tile_batch_size = min(4, ...)`, and `acc[4]`.
- Artifact: `local-artifacts/benchmarks/q6k-iteration-full.json`.
- Workload: five models, three repetitions, llama.cpp comparison.
- Result versus the FFN iteration: Qwen 1.5B +5.87%, Llama 1B +10.32%, Llama 3B +6.50%, DeepSeek 1.5B +6.25%, Qwen3 +14.50%.
- Correctness: Q4_K/Q6_K batched parity 2 passed; Q6_K GEMV 1 passed; Q6_K dequant 2 passed.
- Decision: **Accepted provisionally**.

### R3-NORM-001 — Two Norm/SwiGLU rows per block

- Hypothesis: packing two independent rows into a 64-thread block improves the low-utilization Norm/SwiGLU launch shape; odd batches retain the 32-thread fallback.
- Scope: `norm_rope.cu`, `norm_rope.rs`, focused Norm/SwiGLU parity.
- Artifact: `local-artifacts/benchmarks/norm-swiglu-iteration-full.json`.
- Result versus Q6_K iteration: Qwen 1.5B +0.03%, Llama 1B +0.08%, Llama 3B +0.15%, DeepSeek +0.23%, Qwen3 +0.88%.
- Correctness: Norm/RoPE/SwiGLU parity passed; fused GPU parity passed; CPU references passed. Two Clippy issues were fixed using idiomatic `is_multiple_of` and removal of an unnecessary cast.
- Decision: **Accepted as a safe neutral iteration**; no material performance gain claimed.

### R3-FLASH-001 — Warp-local block-table broadcast

- Hypothesis: lane 0 can read `block_table[b]` once and broadcast the physical block index with `__shfl_sync`, eliminating identical per-lane global reads.
- Scope: `flash_attention_2.cu` and focused parity contract.
- Artifact: `local-artifacts/benchmarks/flash-attention-iteration-full.json`.
- Result versus Norm/SwiGLU iteration: Qwen 1.5B +0.07%, Llama 1B +0.35%, Llama 3B approximately 0.00%, DeepSeek +0.05%, Qwen3 -0.19%.
- Correctness: FlashAttention parity passed for sequence lengths 1, 4, 16, 64, and 128; FlashDecoding parity passed.
- Decision: **Accepted as safe but performance-neutral**. No material gain claimed.

### R3-Q6K-002 — Remove forced unrolling in Q6_K helper

- Hypothesis: removing the forced row-loop unroll reduces live accumulator lifetime and register pressure.
- Scope: one `#pragma unroll` in `compute_q6k_block_multi_row`.
- Artifact: `local-artifacts/benchmarks/q6k-unroll-iteration-full.json`.
- Result versus Q6_K tile-four checkpoint: Qwen 1.5B -0.94%, Llama 1B -1.10%, Llama 3B -1.40%, DeepSeek -0.88%, Qwen3 +0.19%.
- Correctness: Q6_K parity remained green.
- Decision: **Rejected** and reverted. The forced unroll remains.

### R3-DECODE-001 — HD64 paged-attention dispatch regression

- Hypothesis under test: the new `paged_attention_hd64_decode_kernel` might improve head-dimension-64 decode.
- Differential evidence: dirty checkout failed prompt 0 with cosine 0.665956; clean `HEAD` passed prompt 0 with cosine 0.993794. The new HD64 dispatch was the functional difference.
- Fix: production dispatch returned to validated `paged_attention_decode_kernel`; the experimental HD64 kernel was no longer loaded/selected.
- Artifact: `local-artifacts/decode-drift-after-hd64-fix.log`.
- Result: `decode_reuses_resident_kv_and_emits_logits` passed all 12 prompts: minimum cosine vs CPU 0.993334, maximum rel-L2 0.1266, minimum cosine vs prefill 0.993275.
- Decision: **HD64 dispatch rejected for production**; validated original dispatch retained.

### R3-Q6K-003 — Restore Q6_K multi-row dispatch

- Observation: after the HD64 correction, `gemm_q6k()` had accidentally stopped selecting `fn_q6k_multi_row`; the benchmark fell from 0.748x to 0.684x on Llama 3B.
- Fix: restore the Q6_K multi-row launch with `grid_y = div_ceil(batch_size, 4)`.
- Artifact: `local-artifacts/benchmarks/q6k-restored-final.json`.
- Result versus the degraded state: Llama 3B recovered from 0.684x to 0.748x; Qwen 1.5B to 0.886x; Llama 1B to 0.847x; DeepSeek to 0.900x; Qwen3 to 1.114x.
- Decision: **Accepted** as restoration of the previously validated Q6_K iteration.

### R3-LMHEAD-001 — Single-row Q6_K LM-head dispatch

- Hypothesis: for `batch_size == 1`, `gemm_q6k_2col_kernel` might outperform the Q6_K multi-row path for the tied embedding LM head.
- Scope: LM-head dispatch only; experiment was reverted.
- Artifact: `local-artifacts/benchmarks/lm-head-q6k-single-row-3b-20260831.json`.
- Workload: Llama 3.2 3B, three repetitions, Titan-only (`TITAN_BENCHMARK_SKIP_LLAMA=1`).
- Result: baseline LM-head median 43.4744 ms; single-row variant 71.4568 ms; regression +64.365%.
- Decision: **Rejected** and reverted. Multi-row dispatch remains active.

## Nsight research evidence

### Broad capture (partial)

- Artifact: `local-artifacts/profiles/ncu-post-iterations.ncu-rep`.
- Imported CSV: `local-artifacts/profiles/ncu-post-iterations-preview.csv`.
- The capture was stopped after replaying for several hours; it is useful as partial Titan-side evidence, not as a complete symmetric reference profile.

### Small per-kernel captures

- `post-iterations-small/q6k-retry.ncu-rep` and `.csv`:
  - Q6_K multi-row: 55 registers/thread, 56 allocated, register limit 4 blocks/SM, block 256, duration 2.047904 ms in the profiled launch.
- `post-iterations-small/norm-swiglu-retry.ncu-rep` and `.csv`:
  - Norm/SwiGLU: 42 registers/thread, block 64, grid 39, duration 24.512 us; low global work saturation.
- `post-iterations-small/flash-attention-retry.ncu-rep` and `.csv`:
  - FlashAttention-2: 38 registers/thread, 40 allocated, block 32, theoretical register limit 48 blocks/SM, duration 201.920 us.

No valid symmetric `llama.cpp` Nsight report was obtained. Resource comparisons to `llama.cpp` therefore remain partial.

## Final measured checkpoint for the current Point-3 state

Artifact: `local-artifacts/benchmarks/q6k-restored-final.json`.

- Qwen 2.5 1.5B: ratio 0.8855.
- Llama 3.2 1B: ratio 0.8467.
- Llama 3.2 3B: ratio 0.7481.
- DeepSeek-R1-Distill 1.5B: ratio 0.8997.
- Qwen3 0.6B: ratio 1.1143.

All five entries have three repetitions. The project target of at least 0.95x for every model is not met.

## Validation register

- `openspec validate --all`: 17 passed, 0 failed.
- `cargo check --workspace`: passed.
- `cargo clippy --workspace --all-targets -- -D warnings`: passed after cleanup.
- `cargo test --workspace`: passed for non-ignored tests.
- Directed GPU parity tests: passed for Q6_K, paged attention, Norm/SwiGLU, FlashAttention, and related paths.
- Teacher-forced drift: partial execution produced passing checkpoints but did not produce a complete final summary; status remains **inconclusive**, not passed.

## Research conclusions

- The Q6_K tile-four dispatch is the only Point-3 optimization with a clear multi-model throughput benefit in the measured checkpoints.
- Norm/SwiGLU and block-table broadcast are correctness-safe but nearly neutral at the tested workload.
- Q6_K unroll removal and single-row LM-head dispatch are rejected.
- The HD64 paged-attention dispatch is rejected for production because it caused a large decode-correctness regression.
- The final `>=0.95x` per-model milestone remains open; no release sign-off is claimed.

## Fresh checkpoint — 2026-09-01

- Artifact: `local-artifacts/benchmarks/rerun-20260901-085229.json`.
- Raw log: `local-artifacts/benchmarks/rerun-20260901-085229.log`.
- Command: `cargo test --release -p engine-server --test multi_model_comparison_bench -- --ignored --nocapture`.
- Configuration: three repetitions per model, five models, two prompts, 41 generated tokens, greedy sampling (`temperature = 0.0`).
- Execution: `1 passed, 0 failed`, 100.76 seconds.
- Reported ratios: Qwen 2.5 1.5B `1.009x`; Llama 3.2 1B `0.896x`; Llama 3.2 3B `0.735x`; DeepSeek 1.5B `0.864x`; Qwen3 0.6B `1.057x`.
- Decision: fresh benchmark evidence is valid, but the `>=0.95x` per-model and aggregate release gates remain open. This checkpoint does not justify archiving F8 or claiming general parity.


## R4-FFN-DOWN-001 — Shape-specific candidate comparison

- Date: 2026-09-01
- Target: Llama 3.2 3B, Q4_K FFN down, shape 8192 x 3072, batch=1.
- Harness: `engine-server/tests/ffn_3b_isolation.rs`, five measured repetitions after one warm-up, real CUDA/NVRTC path.
- Baseline artifact: `local-artifacts/benchmarks/ffn-down-candidates-20260901/default.json`.

| Requested variant | Median decode wall time (ms) | Delta vs default | Decision |
|---|---:|---:|---|
| `default` | 678.589 | +0.00% | baseline retained |
| `gemm_q4k_2col_kernel` | 683.134 | +0.67% | inconclusive/neutral |
| `gemm_q4k_mma_kernel` | 739.126 | +8.92% | rejected: regression |
| `gemm_q4k_mma_splitk2_kernel` | 680.549 | +0.29% | inconclusive/neutral |

- The `gemm_q4k_mma_kernel` candidate regressed by approximately 8.92% and is rejected.
- The `gemm_q4k_2col_kernel` candidate regressed by approximately 0.67%; this is not a positive result and is rejected for this shape.
- Re-requesting the current `gemm_q4k_mma_splitk2_kernel` measured within noise (+0.29%) and is retained as the current default, not accepted as an optimization.
- No production dispatch change was made.


## R5-FFN-GATE-UP-001 — Fused Gate/Up candidate comparison

- Date: 2026-09-01
- Target: Llama 3.2 3B, Q4_K fused Gate/Up, shape 3072 x 8192, batch=1.
- Harness: `engine-server/tests/ffn_3b_isolation.rs`, five measured repetitions after one warm-up, real CUDA/NVRTC path.
- Artifacts: `local-artifacts/benchmarks/ffn-gate-up-candidates-20260901/*.json` and matching `.log` files.

| Requested variant | Median decode wall time (ms) | Delta vs default | Decision |
|---|---:|---:|---|
| `gemm_q4k_fused_gate_up_swiglu_mma_kernel` | 533.586 | baseline | retained |
| `gemm_q4k_fused_gate_up_swiglu_mma_2col_kernel` | 557.850 | +4.55% | rejected: regression |
| `gemm_q4k_fused_gate_up_swiglu_mma_splitk2_kernel` | 553.783 | +3.79% | rejected: regression |

- Both alternatives were selected and observed in telemetry, so the result is a valid runtime comparison.
- The current fused MMA kernel remains the production benchmark path. No dispatch change was accepted.

## Current execution reconciliation — 2026-09-02

The following artifacts record the current Q8/release execution. These are current measurements and statuses, distinct from the historical observations and checkpoints above. The classifications retain their caveats: verified evidence is not release acceptance, diagnostics are not production gates, blocked work is not passed work, rejected/deferred candidates made no production change, and provisional measurements are not final sign-off.

- phase0 verified: `local-artifacts/reviews/phase0-q8-release-execution-20260902.json`.
- phase1 release gate implemented and independently verified: `tools/release_gate.py`, `tools/test_release_gate.py`; 9 tests pass. The real current gate rejects with missing models, low ratios, and regression.
- phase2 matrix diagnostic verified: `tools/checkpoint_matrix.py`, `tools/test_checkpoint_matrix.py`; 4 tests pass; `local-artifacts/reviews/phase2-checkpoint-matrix-20260902.json`. No compatible build identity was available, so ratios and regressions are suppressed.
- phase3 Q8-vs-FP32 diagnostic: `engine/engine-core/tests/real_q8_vs_f32_differential.rs` and `local-artifacts/reviews/phase3-q8-f32-differential-20260902.json`; Q8 gate rel_l2 `6.479085e-3`, Q8 up rel_l2 `7.424915e-3`. Fused/full FFN is unsupported and is not accepted as a production parity gate.
- phase4 dispatch observability verified after the role lifecycle fix: `engine/engine-core/src/forward_driver.rs`; `local-artifacts/reviews/phase4-nograph-dispatch-20260902.json`; 197 launches, 8 records, zero missing `tensor_role`/`selected_variant`. Roles: `qkv`, `attn_output`, `ffn_gate_up`, `ffn_down`, `lm_head`.
- phase5 benchmark checkpoint: `local-artifacts/benchmarks/phase5-current-f32-qwen3-clean-20260902_135440.json` and sibling `.log`; 3 repetitions; Qwen3 0.6B `131.173849 tok/s` cold and `130.204819 tok/s` warm; static ratios `0.546083x` and `0.468223x`; provisional only.
- phase6 profiling blocked: `local-artifacts/reviews/phase6-nsight-status-20260902.json`; `ERR_NVGPUCTRPERM`; no valid `.ncu-rep`.
- phase7 candidate rejected/deferred: `local-artifacts/reviews/phase7-candidate-q8-reenable-20260902.json`; no production change.
- phase8 llama rebaseline blocked: `local-artifacts/reviews/phase8-llama-rebaseline-status-20260902.json`; no `llama-server.exe`, four required GGUFs, or fresh same-harness reference.
- phase9 current status: `openspec validate --all` passes 17/17, but release remains blocked and the change must not be archived. No release-ready claim is made.

## Full-plan continuation and current benchmark checkpoint — 2026-09-02

- Verified full Titan benchmark artifact: `local-artifacts/benchmarks/full-current-titan-20260902_151148.json` and sibling `.log`. Five models, three repetitions, three serialized runs per model, 41 generated tokens, temperature `0.0`, cold/warm measurements, and all five model paths exist. Model hashes: `local-artifacts/reviews/full-benchmark-status-20260902_151148.json`.
- Verified current direct-F32 path throughput, cold/warm tok/s: Qwen2.5 `65.711/66.087`; Llama3.2-1B `89.501/89.809`; Llama3.2-3B `36.723/36.959`; DeepSeek1.5B `65.283/65.383`; Qwen3 `131.460/133.603`.
- Exploratory/static comparison: `local-artifacts/benchmarks/full-static-vs-current-titan-20260902_151148.json`; exploratory/static, not fresh reference evidence. Ratios cold/warm: Qwen2.5 `0.474/0.488`; Llama1B `0.501/0.475`; Llama3B `0.351/0.352`; DeepSeek `0.416/0.378`; Qwen3 `0.547/0.480`; overall `0.475x`. `local-artifacts/reviews/full-release-gate-20260902.json` rejects with `per_model_ratio` and `aggregate_ratio`.
- Diagnostic benchmark-only selector `TITAN_FORWARD_DECODE_PATH=q8` is compiled. Full Q8 diagnostic artifact: `local-artifacts/benchmarks/full-current-titan-q8-diagnostic-20260902_153013.json`. Rejected/deferred: integrated Q8 decode gate cosine `0.665718`, rel_l2 `7.484e-1`; diagnostic projection errors `6.479085e-3` and `7.424915e-3`. Candidate `local-artifacts/reviews/phase7-candidate-q8-reenable-20260902.json` has status `rejected_deferred`.
- Diagnostic Q8 dispatch probe: `local-artifacts/reviews/q8-dispatch-probe-20260902.json`. Variants were observed, but three records still have missing `tensor_role` (fused gate/up and two MMA projection records). This is an observability gap, not a correctness conclusion. Q8 telemetry is not complete.
- Verified llama.cpp source exists only under `local-artifacts/llama.cpp` at commit `9cffdcc801582616250520966699cb5b25d28243`. CUDA configure through `local-artifacts/llama-build-cuda/configure.bat` is blocked because CMake could not find CUDA Toolkit/nvcc; do not classify GGUF models as missing. The exact blocker is in command output.
- Blocked Nsight status remains `ERR_NVGPUCTRPERM`; artifact `local-artifacts/reviews/phase6-nsight-status-20260902.json`.
- Correction to the preceding reconciliation: all five model paths are verified to exist. The llama.cpp comparison remains blocked by CUDA configuration/reference execution, not by missing GGUF models.
- Verified OpenSpec validation remains 17 passed, 0 failed. This change remains active and must not be archived. RC remains blocked.
- Next technical gate: full-driver Q8 correctness/telemetry, followed by one optimization hypothesis. No release claim is allowed.

## Final technical continuation — 2026-09-02

This final continuation is documentation-only. Historical research entries and the historical checklist state of 7/21 are preserved.

- Q8 full-driver mode is benchmark-only via `TITAN_FORWARD_DECODE_PATH=q8`. It reproducibly fails the integrated gate: cosine `0.665718`, rel_l2 `7.484e-1`. It is **not accepted**.
- Final Q8 dispatch probe after the final role patch: `local-artifacts/reviews/q8-dispatch-probe-final-20260902.json`; 211 launches, 8 records, zero missing `tensor_role`/`selected_variant`. The artifact contains only these roles: `activation_quantization`, `qkv`, `attn_output`, `ffn_gate_up`, `ffn_down`.
- The Q8 Llama 3.2 3B FFN-down 2col candidate was measured in 5 repetitions against control. Control artifact: `local-artifacts/benchmarks/q8-3b-ffn-control-20260902.json`; candidate artifact: `local-artifacts/benchmarks/q8-3b-ffn-2col-20260902.json`. Median wall-clock control `665.2911 ms`; candidate `664.9643 ms`; throughput delta `+0.0491%`; status `rejected_neutral`; no production dispatch change. Decision artifact: `local-artifacts/reviews/candidate-q8-3b-ffn-down-2col-20260902.json`.
- CUDA Toolkit 13.3 installed via winget; nvcc `13.3.73`; MSVC `19.51.36248.0`. CMake configure and llama-server build passed `474/474`. llama.cpp source: `local-artifacts/llama.cpp`, commit `9cffdcc801582616250520966699cb5b25d28243`; binary: `local-artifacts/llama-build-cuda/bin/llama-server.exe`.
- Fresh head-to-head artifact `local-artifacts/benchmarks/final-head-to-head-20260902_165459.json` and sibling log `local-artifacts/benchmarks/final-head-to-head-20260902_165459.log`: 5 models, 3 repetitions, CUDA0 RTX 3060 observed, `USE_GRAPHS=1`, `generated_tokens=41`, temperature `0.0`, cold/warm measurements, and all llama and Titan metrics present.
- Fresh release-gate derived artifacts: `local-artifacts/benchmarks/final-head-to-head-llama-reference-20260902_165459.json` and `local-artifacts/benchmarks/final-head-to-head-titan-20260902_165459.json`. Fresh gate expected rejected because ratios remain below `0.95`; no release-ready claim.
- Nsight remains blocked by `ERR_NVGPUCTRPERM` at `local-artifacts/reviews/phase6-nsight-status-20260902.json`.
- OpenSpec validation must remain 17 passed, 0 failed. The active change must not be archived because performance/correctness/release acceptance remain open.
- Precise classification: the next blocker is **Q8 full-driver correctness/performance**, not model availability or llama.cpp availability.

## Q6_K Q8-dispatch correction and final technical checkpoint — 2026-09-02

This section is documentation-only. Historical research entries and the historical 7/21 checklist remain preserved.

- Root cause verified: `engine-cuda/src/batched_gemm.rs::gemm_q6k` previously selected `fn_q6k_multi_row` (a kernel that expects float x) for Q8 `qx`/`qd`/`qs`; real Qwen3 V Q6_K projection measured `rel_l2=0.667568`, `cosine=0.744549`.
- Atomic production fix verified: Q8 Q6_K dispatch now uses Q8-compatible `splitk2`/`2col`/`normal` kernels; no CUDA kernel source or F32 path changed.
- After the fix, V projection measured `rel_l2=0.006830045`, `cosine=0.999976680`. Full real layer-0 Q8 diagnostic artifact: `local-artifacts/reviews/real-q8-layer0-parity-final-20260902.json`; 13 stages, `full_layer_output` measured, `first_failing_stage=q_projection_q8`; q projection `rel_l2=9.186926e-3`; full layer `rel_l2=1.261698e-2`, `cosine=0.999922688`. Diagnostic only; Q8 remains unaccepted under strict thresholds.
- Post-fix suite passed from `engine/`: `cargo test --workspace -- --ignored --test-threads=1` (engine-cuda ignored tests, including `driver_graph_parity`), `cargo test -p engine-server --test e2e_chat_completions`, `cargo fmt --all -- --check`, `cargo check --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` (non-ignored workspace suite).
- Post-fix Q8 five-model benchmark: `local-artifacts/benchmarks/final-titan-q8-after-q6k-fix-20260902_180740.json` and sibling `.log`; 3 repetitions, five models. Throughput diagnostic, not correctness acceptance.
- Fresh llama.cpp reference: CUDA Toolkit `13.3.73`/`nvcc`, commit `9cffdcc801582616250520966699cb5b25d28243`, build `474/474`; head-to-head `local-artifacts/benchmarks/final-head-to-head-20260902_165459.json` and sibling `.log`. Fresh release gate `local-artifacts/reviews/fresh-head-to-head-release-gate-20260902.json` rejects: aggregate overall ratio `0.4955685`, all models below `0.95`.
- `local-artifacts/reviews/candidate-q8-3b-ffn-down-2col-20260902.json` remains `rejected_neutral`; throughput delta `+0.0491%`.
- Nsight remains blocked by `ERR_NVGPUCTRPERM`; `openspec validate --all` remains `17 passed/0 failed`; change active, not archived; RC blocked.
- Classification: Q6_K dispatch fix is a **verified correctness repair**; Q8 strict acceptance remains blocked by quantization error; performance release gate remains rejected.
- Next blocker: Q8 activation quantization contract/full-driver acceptance, not Q6_K ABI or llama.cpp availability.

## Q8 Q6_K repair and final single-step differential — 2026-09-02

This section is documentation-only. Historical entries and checklist 7/21 remain preserved. The change stays active and is not archived; RC remains blocked.

- Root cause fixed in `engine/engine-cuda/src/batched_gemm.rs`: Q8 `gemm_q6k` previously selected the F32-input multi-row kernel; it now uses Q8-compatible `splitk2`/`2col`/`normal` kernels. Before V Q6_K `rel_l2=0.667568`, `cosine=0.744549`; after `rel_l2=0.006830045`, `cosine=0.999976680`.
- Full layer-0 diagnostic artifact `local-artifacts/reviews/real-q8-layer0-parity-final-20260902.json` has 13 stages and measured `full_layer_output`: first strict-threshold failure `q_projection_q8` with `rel_l2=9.186926e-3`; full layer `rel_l2=1.261698e-2`, `cosine=0.999922688`. This is quantization approximation, not Q6_K ABI corruption.
- New single-step full-driver diagnostic artifact `local-artifacts/reviews/real-q8-single-decode-differential-20260902.json`: F32 vs Q8 on the same real model/prompt/token, `rel_l2=0.046530314913217234`, `cosine=0.99892385922607`, `finite=true`, `diagnostic_only`. This does not claim full multi-prompt Q8 acceptance; the long full Q8 gate remains uncompleted after the repair.
- Q8 five-model post-fix benchmark `local-artifacts/benchmarks/final-titan-q8-after-q6k-fix-20260902_180740.json` and sibling log exist; throughput diagnostic only.
- Q8 FFN-down 2col candidate `local-artifacts/reviews/candidate-q8-3b-ffn-down-2col-20260902.json`: `rejected_neutral`, median throughput delta `+0.0491%`; no production dispatch change.
- Fresh llama.cpp CUDA head-to-head `local-artifacts/benchmarks/final-head-to-head-20260902_165459.json` and sibling log completed 5 models x 3 repetitions. Fresh gate `local-artifacts/reviews/fresh-head-to-head-release-gate-20260902.json` rejected overall ratio `0.4955685`.
- CUDA Toolkit `13.3.73`/`nvcc` and llama-server build `474/474` are verified. Nsight remains blocked by `ERR_NVGPUCTRPERM`.
- Post-fix engine-cuda ignored suite, graph parity, E2E, workspace tests, fmt/check/strict-clippy pass. `openspec validate --all` must remain `17 passed/0 failed`.
- Classification: Q6_K correctness repair verified; Q8 strict full-driver acceptance still open due quantization drift; performance gate rejected; candidate neutral/rejected; reference build resolved; Nsight blocked.
