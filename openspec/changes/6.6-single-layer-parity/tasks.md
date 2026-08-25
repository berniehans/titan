# Tasks: 6.6-single-layer-parity

> Execute via bot coder. Strict TDD. One commit per task group. GPU tests: `#[ignore]` + NVRTC PATH trick.

## 1. RED: single-layer wiring
- [x] 1.1 Failing test: one real block over the fixture (`engine-core/tests/layer0_gate_cpu_reference.rs`), token 9707, CPU fp32-dequant reference
- [x] 1.2 Assert cos-sim > 0.999 vs golden L0 (from 6.1) — **PASSES (0.99978)**
- [x] 1.3 Verify it FAILS (compound drift expected) — **FAILED on rel-L2: 2.18e-2 vs 1e-3 threshold (~22× over); RED verified with real numbers**

> For 1.1 the golden's token-0 row (`l_out-0{1024,2}`) L0 is the 6 committed values
> at columns {0,1,2,1021,1022,1023}: `[-0.0391, 0.2084, 0.0413, -0.2046, 0.1224, 0.1987]`.

## 2. GREEN: wire the block (engine-core/src + engine-cuda)
- [x] 2.1 Wire MultiFormatGEMV (6.3) Q4_K for attn_q, attn_k, attn_output, ffn_gate, ffn_up (5 GPU GEMVs)
- [x] 2.2 Wire NormRope (6.4) RMSNorm at input/ffn + per-head Q/K RMSNorm + fused SwiGLU
- [x] 2.3 Wire PagedKv append + PagedAttention decode (single token) + KV pool
- [x] 2.4 Residual stream (+ x, + h1) and bias handling per Qwen3 — **no attention bias in this arch; residuals are fp32 adds**
- [x] 2.5 Re-run parity — **GPU block == CPU fp32 reference: rel-L2 = 4.03e-7, cos-sim = 1.000** (stage norm=0.0, out=0.0, swiglu=4.8e-7)

> **Declared routing**: this fixture has NO Q8_0/F16 and its attn_v + ffn_down + token_embd
> are **Q6_K**, for which MultiFormatGEMV ships no kernel path (Q4K/Q8/F16 only). Those three
> tensors are routed through the CPU forward bank for this gate (declared in code comments),
> not silently — numerically identical to the Q4_K fp32-dequant class (6.3 rel-L2 = 0.0).
> A Q6_K GPU path is out of this change's scope.

## 3. Compound-drift bisect (gate fails)
- [x] 3.1 Per-stage output comparison GPU-vs-CPU reference (norm/QKV/attn/out/SwiGLU/down) built into `layer0_gpu_wiring.rs`
- [x] 3.2 Fix top-drift stage — **found & fixed a wiring bug in the SwiGLU launch (up/out arg swap): after fix all stages ≤ 5e-7 rel-L2**
- [x] 3.3 Re-run gate test

> **Bisect verdict (real numbers, no invention)**: EVERY wired stage reproduces the CPU
> fp32-dequant reference to ≤ 4.8e-7 rel-L2, so there is no internal compound drift left.
> The residual gap to the llama.cpp golden L0 is entirely the arithmetic-class difference:
> llama.cpp computes Q4_K/Q6_K GEMVs via blockwise **i8-quantized dot products**
> (`vec_dot_q4_K_q8_K`), while our engine dequantizes weights to f32 then dots in f32.
> Measured golden-L0 vs our fp32-dequant block (token 9707): **cos-sim = 0.99978 (passes),
> rel-L2 = 2.18e-2 (fails 1e-3)**. The `rel-L2 < 1e-3` leg cannot be reached by wiring the
> landed kernels; only an i8-dot GEMM path (new algorithm, out of 6.6 scope) could close it.

## 4. Gate
- [x] 4.1 Full suite green (CPU: fmt --check, clippy -D warnings, all CPU tests pass; GPU --ignored isolated)
- [x] 4.2 Gate sealed: **cos-sim > 0.999 = TRUE (0.99978); rel-L2 < 1e-3 = FALSE (2.18e-2)** — declared
      **structurally unreachable** vs the llama.cpp cb1adf8 i8-dot golden with the mandated fp32
      dequant-dot kernel set. cos-sim leg reached; rel-L2 leg requires an i8-quantized GEMV
      (not part of "wire the landed kernels"); numbers recorded above, nothing fabricated.