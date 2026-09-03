# Comparison Evidence: Titan vs `llama.cpp` on Llama 3.2 3B

## Captured Titan hardware data

Nsight Compute 2025.4.1 was run elevated with GPU-counter permission enabled by the user. The report and imported CSV are:

- `local-artifacts/profiles/llama3b-titan-only-admin-ncu.ncu-rep`
- `local-artifacts/profiles/llama3b-ncu-import.csv`

Representative kernel resources observed on the RTX 3060 Laptop (SM 8.6):

| Kernel | Block | Registers/thread | Shared/block | Launch occupancy limits | Compute/memory metric |
|---|---:|---:|---:|---|---:|
| `gemm_q4k_batched_kernel` | 128 threads | 48 | 1.024 KB | register limit 10 blocks, warp limit 12 | 95.29% compute/memory throughput metric |
| `gemm_q6k_batched_kernel` | 128 threads | 72 | 1.024 KB | register limit 7 blocks, warp limit 12 | 63.81% compute/memory throughput metric |
| `norm_rope_swiglu_kernel` | 32 threads | 42 | 1.024 KB | block limit 16, warp limit 48 | 19.63% compute/memory throughput metric |
| `flash_attention_2_kernel` | 32 threads | 40 | 1.024 KB | block limit 16, warp limit 48 | 60.34% compute/memory throughput metric |

Derived theoretical occupancy from the reported resource limits (not measured active occupancy):

- Q4_K GEMM: `min(16, 10, 16, 12) * 4 / 48 = 83.3%` warp capacity.
- Q6_K GEMM: `min(16, 7, 16, 12) * 4 / 48 = 58.3%` warp capacity.
- Norm/SwiGLU and FlashAttention: `16 * 1 / 48 = 33.3%` warp capacity.

The Q6_K path is register-limited relative to Q4_K. This is actionable evidence for the next GEMV/GEMM investigation, but it does not by itself prove that Q6_K is the dominant model-level bottleneck; the stage profile identifies FFN first.

## Reference engine evidence

The exact `llama-server.exe` used by the head-to-head benchmark is:

- Version: `1 (74ade5274)`
- SHA-256: `0da24b1e082e87df7db915d0e0e5d162eebe423e25b687d73475e6d357d5b92a`
- Runtime log: CUDA enabled, CUDA Graphs enabled, RTX 3060 detected, `n_threads=8`, `n_threads_batch=8`, `n_parallel=4`, `n_ctx=2048`.
- Current controlled 3B head-to-head result: Titan `72.7 tok/s`, `llama.cpp` `104.5 tok/s`, ratio `0.70x`.

Two elevated direct `llama-server` Nsight attempts were made with a single 3B request. The server accepted the request, but Nsight did not emit a kernel profile or `.ncu-rep`; it remained waiting for a matching launch. Therefore the reference engine's occupancy/register/shared-memory/transaction counters are **not available** and are not inferred from Titan's counters.

The binary's short version identifier does not resolve to a public `ggml-org/llama.cpp` commit, so a source-level exact tile comparison is also unavailable.

## Gate result

- Titan-side hardware profile: **captured and imported successfully**.
- Reference-side hardware profile: **not captured**.
- 2.2 remains open under the original OpenSpec wording because a symmetric hardware comparison against the pinned reference has not been obtained.
- The comparison is still sufficient to select the next hypothesis: investigate FFN first, then the register-limited Q6_K GEMV path.

## Fresh benchmark checkpoint — 2026-09-01

The current checkout was rerun after the Point-3 iterations with three repetitions per model. The machine-readable artifact is `local-artifacts/benchmarks/rerun-20260901-085229.json` and the raw log is `local-artifacts/benchmarks/rerun-20260901-085229.log`.

| Model | Titan / llama.cpp reported ratio |
| :--- | ---: |
| Qwen 2.5 1.5B | 1.009x |
| Llama 3.2 1B | 0.896x |
| Llama 3.2 3B | 0.735x |
| DeepSeek 1.5B | 0.864x |
| Qwen3 0.6B | 1.057x |

The benchmark test passed (`1 passed, 0 failed`) but does not resolve requirement 2.2: it provides runtime throughput evidence, not symmetric reference-engine hardware counters. The `>=0.95x` performance gate remains open.
