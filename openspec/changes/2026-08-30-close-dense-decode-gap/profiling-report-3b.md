# Profiling Report: Llama 3.2 3B Dense Decode

## Run Identity

- Repository commit: `97993d0c44faff430db224911cdf51d43a09464c`
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6, 6144 MiB
- Driver: 610.74
- Titan NVRTC: `local-artifacts/nvrtc-cu12/runtime/nvrtc64_120_0.dll`
- Benchmark artifact: `local-artifacts/benchmarks/task-1-final-telemetry.json`
- Workload: two prompts, batch size 1, 41 generated tokens, greedy decoding
- Measurement: CUDA events for compute buckets, explicit host boundaries for copies/waits, explicit graph replay and overlap

## Llama 3.2 3B Measurements

The artifact contains two prompt rows for this model. Values below are medians across those rows:

| Stage | Median |
|---|---:|
| FFN | 352.099 ms |
| GEMV/GEMM | 72.947 ms |
| LM head | 68.098 ms |
| Attention | 58.463 ms |
| CUDA Graph replay boundary | 5.546 ms |
| Copies | 0.355 ms |
| Explicit waits | 0.006 ms |

Decode wall-clock median: `653.567 ms` for the measured generation window.

## Findings

1. FFN is the largest measured compute bucket and is therefore the primary optimization target.
2. GEMV/GEMM is the second-largest measured compute bucket and the secondary target.
3. LM head and attention are material but smaller than the first two buckets.
4. Copies and final synchronization are not the cause of the current 3B deficit in this run.
5. CUDA Graph replay/host boundary is small relative to FFN/GEMV/GEMM; graph work should not be optimized first without new evidence.

## Limitations

- Nsight Systems and Nsight Compute are not installed on this host, so hardware occupancy, register count, shared-memory usage, and DRAM transaction counters were not available.
- The installed `llama-server.exe` exposes runtime logs and a binary identity, but its source tree and build flags are not present locally. A source-level tiling comparison against the exact binary cannot be claimed.
- Stage buckets are Titan-side measurements; they identify Titan's cost centers but do not prove that `llama.cpp` uses identical internal stage boundaries.

Nsight Compute 2025.4.1 was subsequently installed and invoked with the new
`TITAN_BENCHMARK_MODEL_FILTER=Llama 3.2 3B` filter. The profiler attached to the
benchmark process but stalled while the child `llama-server.exe` was starting;
no `.ncu-rep` artifact was produced. The attempt was stopped after the log
showed the child server waiting at model startup. This is an environment/tooling
limitation of the current Windows child-process/benchmark flow, not a measured
occupancy result.

The benchmark was then given a Titan-only mode (`TITAN_BENCHMARK_SKIP_LLAMA=1`)
and Nsight was run with `--target-processes all`, allowing it to connect directly
to the Titan test executable. Nsight reported `ERR_NVGPUCTRPERM`: the current
Windows/NVIDIA driver policy denies access to GPU performance counters. The run
was stopped without a report because occupancy/register/memory-counter values
would not be valid. The remaining prerequisite for this comparison is enabling
GPU performance-counter access for the profiling user in the NVIDIA driver
policy/control panel (or running under an approved profiling account).

## Gate Status

- 2.1: complete — the top two Titan contributors are identified from measured data.
- 2.2: pending — hardware-counter and exact-reference-shape comparison requires Nsight and/or the pinned `llama.cpp` source/build metadata.
