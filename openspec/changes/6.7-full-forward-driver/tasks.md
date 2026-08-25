# Tasks: 6.7-full-forward-driver

> Execute via bot coder. Strict TDD. One commit per task group. GPU tests: `#[ignore]` + NVRTC PATH trick.

## 1. Prefill entry point (engine-core/src/forward_driver.rs)
- [x] 1.1 Per-layer parity: `run_prefill` streamed GPU block == in-test CPU fp32 reference every token/layer; full-pass rel-L2 (GPU vs own CPU fp32 reference) max across 12 prompts = **6.319e-6** (gate < 1e-3). First-layer state reproduces the 6.6 wiring result in the same fp32-dequant class.
- [x] 1.2 Implement `run_prefill` (all layers on prompt, K/V appended into a resident per-layer paged-KV pool). NVRTC local-GPU verified.
- [x] 1.3 Failing test→gate: full-prompt logits vs the 12 committed llama.cpp goldens AND vs own CPU fp32 reference (loop `forward_cpu` per token/layer in-test). Test: `engine-core/tests/prefill_golden_gate.rs` (`#[ignore]`).
- [x] 1.4 **RE-BASELINED (owner decision, extends the 6.6 ruling)**: primary correctness gate =
      rel-L2 < 1e-3 vs own CPU fp32 reference → MET (max 6.3e-6, 12/12); golden cos-sim floor
      lowered to > 0.99 vs llama.cpp goldens → MET (min 0.99137, 12/12). The >0.999-vs-golden leg
      is the same known fp32-dequant-vs-i8-dot architecture limit, now measured end-to-end.
      Original measurement record:
      - [ ] 1.4 Golden cos-sim gate `> 0.999`: **NOT met at full stack** — min cos-sim vs golden = **0.991370** (prompt 10, len 10); only prompt 04 (len 11, 0.999215) exceeds the target. This is the fp32-dequant-vs-llama-i8-dot class gap (6.6 owner re-baseline) now measured end-to-end: GPU == own CPU fp32 reference to rel-L2 ≤ 2.4e-6 per prompt, but both sit 0.991–0.999 cos from llama. **Correctness gate (rel-L2 vs own CPU fp32 ref < 1e-3, max 6.3e-6) PASSES.** The rel per-prompt (cos_sim golden / rel-L2 CPU ref): 00=0.9971/5.0e-7, 01=0.9986/2.4e-6, 02=0.9927/6.3e-6, 03=0.9984/2.6e-6, 04=0.9992/1.8e-6, 05=0.9973/2.6e-6, 06=0.9962/2.9e-6, 07=0.9976/3.2e-6, 08=0.9984/2.4e-6, 09=0.9982/3.0e-6, 10=0.9914/4.2e-6, 11=0.9984/2.7e-6.

> **RoPE root cause fixed (was a real bug):** Qwen3 NeoX-Partial RoPE rotates the **full head dim (n_rot = head_dim = 128)**, NOT head_dim/2=64 as the earlier design assumed. With n_rot=64, positional rotation at pos≥1 collapsed cos-sim vs golden to 0.577 (prompt 01); with n_rot=128 it recovers to 0.9986 and the whole set to the ≥0.99 class. Documented in `forward_driver.rs` / `forward_cpu.rs`.

## 2. Single-token decode entry point
- [x] 2.1 Single-token decode reusing resident KV emits next-token logits: dec-vs-CPU-fp32 **cos=1.000000 / max rel-L2 6.319e-6** (gate rel-L2<1e-3), and dec == full prefill **bit-identical (rel-L2=0, cos=1.0)** across all 12 prompts. Test: `engine-core/tests/decode_drift_gate.rs` (`decode_reuses_resident_kv_and_emits_logits`).
- [x] 2.2 Implement `ForwardDriver` struct and `decode`/`step_one` single-topology step over resident KV; `run_prefill` re-wrapped over the same struct (working path byte-identical, rel-L2=0), `n_rot=head_dim`=128 kept.
- [x] 2.3 Teacher-forced drift curve: **85 checkpoints** (gate >=10) across 12 prompts, prefill=1 token + sequential `decode` reusing resident KV; per-step vs own CPU fp32 reference: min cos=**1.000000**, max rel-L2=**1.044e-5** (gate <1e-3). Per-prompt full drift table recorded in the test's printed output (real NVRTC GPU run, release). Test: `teacher_forced_drift_curve_10_checkpoints`. Drift mapping declared in test doc-comment: goldens pin only final positions; per-step drift is measured against the model's own CPU fp32 reference, and final-position decode cos vs llama.cpp golden matches the group-1 class gap (min 0.991370).
- [x] 2.4 PASS: both group-2 GPU tests green local (RTX 3060, `cargo test --release --test decode_drift_gate -- --ignored --test-threads=1` → 2 passed).

## 3. VRAM guards + additive safety
- [ ] 3.1 Failing test: per-kernel declared worst-case asserted ≤ budget (resident pingpong + kv_pool + activations + logits ≤ 5.2 GB)
- [ ] 3.2 Failing test: `stub_next_token` path byte-identical result (additive safety — stub untouched)
- [ ] 3.3 Verify PASS (budget trace logged)

## 4. Gate
- [ ] 4.1 Full suite green (existing + new; stub untouched)
- [ ] 4.2 Gate sealed: cumulative drift ≤ tolerance over ≥10 tokens, VRAM ≤ 5.2 GB every step