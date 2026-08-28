# Proposal: Ampere Tensor Cores Acceleration with PTX `mma.sync`

## Motivation

Titan currently uses vectorized DP4A SIMD (`__dp4a`) on standard integer ALUs for quantized matrix multiplication (`Q4_K` / `Q6_K` x `Q8_1`). While this delivers up to 166.0 tok/s on 1B models and 136.5 tok/s on 1.5B models (96.8% parity with `llama.cpp`), it leaves the 120 dedicated **3rd-Generation Tensor Cores** on the RTX 3060 Laptop (Ampere GA106) unutilized.

By implementing custom Warp-Level Matrix Multiply and Accumulate (WMMA / MMA) micro-kernels using PTX inline assembly (`mma.sync.aligned.m16n8k32` for INT4/INT8 and `mma.sync.aligned.m16n8k16` for FP16), Titan can accelerate GEMV/GEMM math by over 3x at the warp level, achieving maximum GDDR6 memory bandwidth saturation (>85% efficiency) and decisively surpassing `llama.cpp` to reach **>200 tok/s** on 1B/1.5B models and **>120 tok/s** on 3B models.

## Metrics & Target Criteria

1. **Throughput Scaling on NVIDIA RTX 3060 Laptop (6 GB VRAM):**
   * **Qwen 2.5 1.5B Instruct (Q4_K_M):** From 136.5 tok/s $\to$ **>180.0 tok/s** ($<5.5\text{ ms/tok}$).
   * **Llama 3.2 1B Instruct (Q4_K_M):** From 166.0 tok/s $\to$ **>210.0 tok/s** ($<4.7\text{ ms/tok}$).
   * **Llama 3.2 3B Instruct (Q4_K_M):** From 70.2 tok/s $\to$ **>110.0 tok/s** ($<9.0\text{ ms/tok}$).
2. **Numerical Bit-Parity:**
   * 100% exact numerical agreement with reference CPU dequantization and previous DP4A golden gates within floating-point tolerance ($<10^{-4}$).
3. **Zero Host Overhead & CUDA Graph Compatibility:**
   * MMA kernels must be 100% capturable in Autonomous CUDA Graphs in VRAM without dynamic allocations.

## Compatibility & Safety

* Architecture Target: Ampere `compute_86` (NVIDIA RTX 30-series / A100 / RTX 40-series).
* Fallback: Transparently falls back to DP4A SIMD if running on older Turing/Volta GPUs without `m16n8k32` instruction support.

## Explicit Non-Goals

1. Supporting FP4/FP8 quantization formats (which require Ada Lovelace / Hopper). Focus is strictly on GGUF standard `Q4_K`, `Q6_K`, and `Q8_0` with `mma.sync`.
2. Modifying the high-level server or KV-cache APIs.
