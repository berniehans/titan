# 📊 Titan — Performance Benchmarks & Empirical Evaluation

This document contains reproducible benchmark results measured directly on real GPU hardware. The current head-to-head baseline compares **Titan (100% Pure Rust)** against CUDA-enabled **llama.cpp (C++)**. Older cross-engine estimates are retained only when explicitly labelled historical; they are not current measurements.

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

## 2. Multi-Model Head-to-Head: Titan vs. llama.cpp (Latest Reproduction)

Automated head-to-head comparison running against the installed CUDA-enabled `llama-server.exe` with CUDA Graphs enabled, under the same GGUF files, two prompts per model, 41 generated tokens, batch size = 1, and greedy sampling ($T=0$). The test completed with `1 passed, 0 failed` in 34.41 seconds.

```
=========================================================================================================
===                                MULTI-MODEL HEAD-TO-HEAD COMPARISON                                ===
=========================================================================================================
Model Name                   | llama.cpp (C++)      | Titan (Pure Rust)    | Ratio (Titan / llama.cpp)
-----------------------------|----------------------|----------------------|-------------
Qwen3 0.6B Base/Chat         | 225.5  tok/s (4.46 ms) | 228.2  tok/s (4.38 ms) | 1.01x (+1.2%)
DeepSeek-R1-Distill 1.5B     | 148.9  tok/s (6.72 ms) | 121.0  tok/s (8.27 ms) | 0.81x (-18.7%)
Qwen 2.5 1.5B Instruct       | 149.5  tok/s (6.70 ms) | 128.2  tok/s (7.80 ms) | 0.86x (-14.3%)
Llama 3.2 1B Instruct        | 214.3  tok/s (4.67 ms) | 158.8  tok/s (6.30 ms) | 0.74x (-25.9%)
Llama 3.2 3B Instruct        | 97.4   tok/s (10.27 ms)| 66.6   tok/s (15.01 ms)| 0.68x (-31.6%)
=========================================================================================================
```

### Key Takeaways:
1. **Qwen3 0.6B:** Titan measured **228.2 tok/s** versus `llama.cpp` at 225.5 tok/s, a marginal **+1.2%** advantage.
2. **Models from 1B to 3B:** `llama.cpp` was faster by **14.3% to 31.6%** in this run.
3. **The previous published figures are superseded:** the current checkout did not reproduce the older 220.8/202.2 and 139.3/136.7 pairs.
4. **Throughput is not numerical parity:** current numerical validation remains a separate gate, and the SwiGLU gate still fails with rel-L2 = 1.047.

---

## 3. Cross-Engine Comparative Landscape

Historical comparison estimates from prior work are not comparable to the latest controlled head-to-head run. They are intentionally omitted here: the only current cross-engine measurements are the five llama.cpp comparisons above.

---

## 4. Kernel-Level Profiling & Latency Breakdown

The older per-stage timing profile below is historical and has not been rerun with the current checkout. It must not be used to derive the current head-to-head numbers.

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
