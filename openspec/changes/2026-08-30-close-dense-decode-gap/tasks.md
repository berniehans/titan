# Tasks: Close the Dense Decode Performance Gap

## 1. Baseline and observability

- [x] 1.1 Freeze the current five-model baseline with model SHA-256, environment metadata, and the exact benchmark command.
- [x] 1.2 Extend `multi_model_comparison_bench` to emit machine-readable results and distinguish cold/warm cache, prefill, decode, and end-to-end timing.
- [x] 1.3 Add Titan stage/launch/synchronization telemetry sufficient to attribute decode time to GEMV/GEMM, attention, FFN, LM head, copies, and waits.
- [x] 1.4 Run three or more repetitions per model and define the baseline median and variance; do not use a single run as a gate.

## 2. Bottleneck isolation

- [x] 2.1 Profile Llama 3.2 3B first and identify the top two contributors by measured GPU time or launch overhead.
- [ ] 2.2 Compare Titan kernel shapes, occupancy, register/shared-memory use, and memory transactions with the pinned `llama.cpp` reference.
- [x] 2.3 Produce a short decision record naming one falsifiable optimization hypothesis and its expected gain.

## 3. Focused optimization iterations

- [ ] 3.1 Optimize the highest-impact Q4_K/Q6_K GEMV/GEMM path for the 3B shapes, preserving DP4A fallback behavior.
- [ ] 3.2 Optimize measured FFN/QKV/output-projection traffic and fusion boundaries; avoid speculative fusion.
- [ ] 3.3 Optimize graph replay and synchronization only if profiling attributes material time to those boundaries.
- [ ] 3.4 Repeat the loop for Llama 3.2 1B and DeepSeek 1.5B, then recheck Qwen 2.5 1.5B and Qwen3 0.6B for regressions.

## 4. Correctness and safety gates per iteration

- [ ] 4.1 Add or update an independent CPU reference test before changing each arithmetic/layout path.
- [ ] 4.2 Run targeted GPU parity, CUDA Graph parity, VRAM-budget, and finite-output tests after each change.
- [ ] 4.3 Preserve the SwiGLU regression gate and all existing GGUF/dequantization edge-case tests.
- [ ] 4.4 Run `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo check --workspace`, and `cargo test --workspace` before accepting an iteration.

## 5. First-instance release gate

- [x] 5.1 Run the final benchmark with at least three repetitions per model and archive raw logs plus machine-readable results. Fresh checkpoint: `local-artifacts/benchmarks/rerun-20260901-085229.json` and matching `.log`; this is evidence for the checkpoint, not release completion.
- [ ] 5.2 Verify Titan/`llama.cpp` decode ratio >= 0.95 for every model and aggregate ratio >= 0.95.
- [ ] 5.3 Verify no model regresses more than 5% from the frozen baseline.
- [ ] 5.4 Update `docs/BENCHMARKS.md` with only the final measured results and methodology.
- [ ] 5.5 Run `openspec validate --all` and record the final evidence.
- [ ] 5.6 Review the complete diff, separate functional changes from formatting/artifacts, prepare a local commit, and stop before push.

## Current execution reconciliation — 2026-09-02

This section reconciles the current Q8/release execution with the historical checklist above. It does not change the historical 7/21 task state and does not mark historical tasks complete.

- phase0 verified: `local-artifacts/reviews/phase0-q8-release-execution-20260902.json`.
- phase1 release gate implemented and independently verified: `tools/release_gate.py`, `tools/test_release_gate.py`; 9 tests pass. The real current gate rejects with missing models, low ratios, and regression.
- phase2 matrix diagnostic verified: `tools/checkpoint_matrix.py`, `tools/test_checkpoint_matrix.py`; 4 tests pass; `local-artifacts/reviews/phase2-checkpoint-matrix-20260902.json`. No compatible build identity was available, so ratios and regressions are suppressed.
- phase3 Q8-vs-FP32 diagnostic: `engine/engine-core/tests/real_q8_vs_f32_differential.rs` and `local-artifacts/reviews/phase3-q8-f32-differential-20260902.json`; Q8 gate rel_l2 `6.479085e-3`, Q8 up rel_l2 `7.424915e-3`. Fused/full FFN is unsupported and is not accepted as a production parity gate.
- phase4 dispatch observability verified after the role lifecycle fix: `engine/engine-core/src/forward_driver.rs`; `local-artifacts/reviews/phase4-nograph-dispatch-20260902.json`; 197 launches, 8 records, zero missing `tensor_role`/`selected_variant`. Roles: `qkv`, `attn_output`, `ffn_gate_up`, `ffn_down`, `lm_head`.
- phase5 benchmark checkpoint: `local-artifacts/benchmarks/phase5-current-f32-qwen3-clean-20260902_135440.json` and sibling `.log`; 3 repetitions; Qwen3 0.6B `131.173849 tok/s` cold and `130.204819 tok/s` warm; static ratios `0.546083x` and `0.468223x`; provisional only.
- phase6 profiling blocked: `local-artifacts/reviews/phase6-nsight-status-20260902.json`; `ERR_NVGPUCTRPERM`; no valid `.ncu-rep`.
- phase7 candidate rejected/deferred: `local-artifacts/reviews/phase7-candidate-q8-reenable-20260902.json`; no production change.
- phase8 llama rebaseline blocked: `local-artifacts/reviews/phase8-llama-rebaseline-status-20260902.json`; no `llama-server.exe`, four required GGUFs, or fresh same-harness reference.
- phase9 current status: OpenSpec validation passes 17/17, but release remains blocked and this change must not be archived.

## Full-plan continuation and current benchmark checkpoint — 2026-09-02

- Verified full Titan benchmark artifact: `local-artifacts/benchmarks/full-current-titan-20260902_151148.json` and sibling `.log`. It covers five models, three repetitions, three serialized runs per model, 41 generated tokens, temperature `0.0`, cold/warm measurements, and all five model paths exist. Model hashes are recorded in `local-artifacts/reviews/full-benchmark-status-20260902_151148.json`.
- Verified current direct-F32 path throughput, cold/warm tok/s: Qwen2.5 `65.711/66.087`; Llama3.2-1B `89.501/89.809`; Llama3.2-3B `36.723/36.959`; DeepSeek1.5B `65.283/65.383`; Qwen3 `131.460/133.603`.
- Exploratory/static llama comparison artifact: `local-artifacts/benchmarks/full-static-vs-current-titan-20260902_151148.json`. It is exploratory/static, not fresh reference evidence. Ratios cold/warm: Qwen2.5 `0.474/0.488`; Llama1B `0.501/0.475`; Llama3B `0.351/0.352`; DeepSeek `0.416/0.378`; Qwen3 `0.547/0.480`; overall `0.475x`. The release gate artifact `local-artifacts/reviews/full-release-gate-20260902.json` rejects with `per_model_ratio` and `aggregate_ratio` failures.
- Diagnostic benchmark-only selector `TITAN_FORWARD_DECODE_PATH=q8` is compiled, and the full Q8 diagnostic artifact is `local-artifacts/benchmarks/full-current-titan-q8-diagnostic-20260902_153013.json`. It is rejected/deferred: integrated Q8 decode gate fails at cosine `0.665718` and rel_l2 `7.484e-1`; diagnostic projection errors are `6.479085e-3` and `7.424915e-3`. Candidate record: `local-artifacts/reviews/phase7-candidate-q8-reenable-20260902.json`, status `rejected_deferred`.
- Diagnostic Q8 dispatch probe: `local-artifacts/reviews/q8-dispatch-probe-20260902.json`. Variants were observed, but three records still have missing `tensor_role` (fused gate/up and two MMA projection records). This is an observability gap, not a correctness conclusion; Q8 telemetry is not complete.
- Verified llama.cpp source is cloned only under `local-artifacts/llama.cpp`, at commit `9cffdcc801582616250520966699cb5b25d28243`. The CUDA configure attempt through `local-artifacts/llama-build-cuda/configure.bat` is blocked because CMake could not find the CUDA Toolkit/nvcc. The GGUF models are not classified as missing; the exact blocker is recorded in command output.
- Nsight remains blocked by `ERR_NVGPUCTRPERM`; artifact: `local-artifacts/reviews/phase6-nsight-status-20260902.json`.
- Correction to the preceding phase8 wording: the current full benchmark verifies that all five model paths exist. The remaining llama.cpp comparison limitation is blocked CUDA configuration/reference execution, not missing GGUF models.
- Verified OpenSpec validation remains 17 passed, 0 failed. This change remains active and must not be archived. RC remains blocked.
- Next technical gate: full-driver Q8 correctness/telemetry, followed by one optimization hypothesis. No release claim is allowed.

## Final technical continuation — 2026-09-02

This final continuation is documentation-only. All historical entries and the historical checklist state of 7/21 remain preserved; no historical task is marked complete here.

- Q8 full-driver mode is benchmark-only via `TITAN_FORWARD_DECODE_PATH=q8`. It reproducibly fails the integrated gate: cosine `0.665718`, rel_l2 `7.484e-1`. It is **not accepted**.
- The final Q8 dispatch probe after the final role patch is `local-artifacts/reviews/q8-dispatch-probe-final-20260902.json`: 211 launches, 8 records, zero missing `tensor_role`/`selected_variant`. The artifact contains only these roles: `activation_quantization`, `qkv`, `attn_output`, `ffn_gate_up`, `ffn_down`.
- The Q8 Llama 3.2 3B FFN-down 2col candidate was measured in 5 repetitions against control. Control: `local-artifacts/benchmarks/q8-3b-ffn-control-20260902.json`; candidate: `local-artifacts/benchmarks/q8-3b-ffn-2col-20260902.json`. Median wall-clock control was `665.2911 ms`; candidate was `664.9643 ms`; throughput delta was `+0.0491%`. Status: `rejected_neutral`; no production dispatch change. Decision artifact: `local-artifacts/reviews/candidate-q8-3b-ffn-down-2col-20260902.json`.
- CUDA Toolkit 13.3 was installed via winget. Verified toolchain: nvcc `13.3.73`, MSVC `19.51.36248.0`. CMake configure and the llama-server build passed `474/474`. llama.cpp source: `local-artifacts/llama.cpp`, commit `9cffdcc801582616250520966699cb5b25d28243`; binary: `local-artifacts/llama-build-cuda/bin/llama-server.exe`.
- Fresh head-to-head artifact: `local-artifacts/benchmarks/final-head-to-head-20260902_165459.json`; sibling log: `local-artifacts/benchmarks/final-head-to-head-20260902_165459.log`. It covers 5 models, 3 repetitions, CUDA0 with RTX 3060 observed, `USE_GRAPHS=1`, `generated_tokens=41`, temperature `0.0`, cold/warm measurements, and all llama and Titan metrics.
- Fresh release-gate derived artifacts are `local-artifacts/benchmarks/final-head-to-head-llama-reference-20260902_165459.json` and `local-artifacts/benchmarks/final-head-to-head-titan-20260902_165459.json`. The fresh gate is expected to be rejected because ratios remain below `0.95`; do not claim release readiness.
- Nsight remains blocked by `ERR_NVGPUCTRPERM`; evidence: `local-artifacts/reviews/phase6-nsight-status-20260902.json`.
- `openspec validate --all` must remain at 17 passed, 0 failed. The active change must not be archived because performance, correctness, and release acceptance remain open.
- Precise classification: the next blocker is **Q8 full-driver correctness/performance**, not model availability or llama.cpp availability.

## Q6_K Q8-dispatch correction and final technical checkpoint — 2026-09-02

This section is documentation-only. All historical entries and the historical 7/21 checklist remain preserved.

- Root cause verified: `engine-cuda/src/batched_gemm.rs::gemm_q6k` previously selected `fn_q6k_multi_row` (a kernel that expects float x) for Q8 `qx`/`qd`/`qs`. This caused real Qwen3 V Q6_K projection `rel_l2=0.667568`, `cosine=0.744549`.
- Atomic production fix verified: Q8 Q6_K dispatch now uses Q8-compatible `splitk2`/`2col`/`normal` kernels. No CUDA kernel source or F32 path changed.
- After the fix, V projection is `rel_l2=0.006830045`, `cosine=0.999976680`. Full real layer-0 Q8 diagnostic artifact: `local-artifacts/reviews/real-q8-layer0-parity-final-20260902.json`; 13 stages, `full_layer_output` measured, `first_failing_stage=q_projection_q8`; q projection `rel_l2=9.186926e-3`; full layer `rel_l2=1.261698e-2`, `cosine=0.999922688`. This is diagnostic; Q8 remains unaccepted under strict thresholds.
- Post-fix suite passed: from `engine/`, `cargo test --workspace -- --ignored --test-threads=1` (engine-cuda ignored tests, including `driver_graph_parity`), `cargo test -p engine-server --test e2e_chat_completions`, `cargo fmt --all -- --check`, `cargo check --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` (non-ignored workspace suite).
- Post-fix Q8 five-model benchmark: `local-artifacts/benchmarks/final-titan-q8-after-q6k-fix-20260902_180740.json` and sibling `local-artifacts/benchmarks/final-titan-q8-after-q6k-fix-20260902_180740.log`; 3 repetitions, five models. This is a throughput diagnostic, not correctness acceptance.
- Fresh llama.cpp reference verified with CUDA Toolkit `13.3.73`/`nvcc`; llama.cpp commit `9cffdcc801582616250520966699cb5b25d28243`; build `474/474`. Head-to-head artifacts: `local-artifacts/benchmarks/final-head-to-head-20260902_165459.json` and sibling `.log`. Fresh release gate `local-artifacts/reviews/fresh-head-to-head-release-gate-20260902.json` rejects: aggregate overall ratio `0.4955685`, all models below `0.95`.
- Optimization candidate `local-artifacts/reviews/candidate-q8-3b-ffn-down-2col-20260902.json` remains `rejected_neutral`; throughput delta `+0.0491%`.
- Nsight remains blocked by `ERR_NVGPUCTRPERM`. `openspec validate --all` remains `17 passed/0 failed`. The change is active, not archived; RC remains blocked.
- Classification: the Q6_K dispatch fix is a **verified correctness repair**; Q8 strict acceptance remains blocked by quantization error; the performance release gate remains rejected.
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
