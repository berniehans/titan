# 📊 Titan — Performance Benchmarks & Empirical Evaluation

This document contains reproducible benchmark results measured directly on real GPU hardware. The current workspace status is maintained in [`WORKSPACE_STATE.md`](WORKSPACE_STATE.md). The head-to-head table below is the historical 2026-09-01 checkpoint; the fresh 2026-09-02 comparison is in `local-artifacts/benchmarks/final-head-to-head-20260902_165459.json` and remains release-blocked.

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
