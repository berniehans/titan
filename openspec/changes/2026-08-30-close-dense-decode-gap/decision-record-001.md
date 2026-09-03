# Decision Record 001: Prioritize FFN and Q4_K/Q6_K GEMV

## Context

The current Llama 3.2 3B Titan profile shows FFN as the largest measured stage and GEMV/GEMM as the second largest. Copies, waits, and the CUDA Graph boundary are comparatively small. The current Titan decode result is approximately `0.70x` the CUDA-enabled `llama.cpp` result for this model.

## Decision

The next optimization iteration SHALL target the measured FFN path first, with the Q4_K/Q6_K GEMV/GEMM path as the follow-up. It SHALL NOT begin with CUDA Graph replay or host-copy changes.

The first hypothesis is:

> H1: FFN intermediate materialization and/or its GEMV execution shape accounts for enough of the 3B gap that reducing FFN memory traffic or improving FFN occupancy will lower Titan's median decode time by at least 10% without changing numerical results.

## Falsification Test

1. Freeze the current 3B profile and correctness outputs.
2. Implement one FFN-only optimization behind the existing path.
3. Run the independent CPU/golden and GPU parity tests.
4. Run at least three repeated 3B benchmark measurements using the Phase 0 JSON output.
5. Accept H1 only if:
   - median Titan 3B decode improves by at least 10%;
   - no existing model regresses by more than 5%;
   - all parity, VRAM, format, Clippy, check, and workspace tests pass.

If the improvement is below 10%, or parity/regression gates fail, reject H1 and profile the Q4_K/Q6_K GEMV path next. Do not combine FFN and GEMV changes in one iteration.

## Expected Gain

The 10% threshold is an acceptance threshold for a meaningful iteration, not a fabricated forecast of final parity. The final milestone remains Titan/`llama.cpp` >= 0.95x for every benchmark model.

## Constraints

- Preserve DP4A fallback behavior.
- Preserve independent CPU references.
- Any upstream arithmetic/layout port must cite a pinned `llama.cpp` or vLLM commit.
- Do not alter HTTP, CLI, or KV-cache APIs.
- No commit or push is implied by this record.
