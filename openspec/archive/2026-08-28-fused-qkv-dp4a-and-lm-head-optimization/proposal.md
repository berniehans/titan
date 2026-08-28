# Proposal: Hardware DP4A SIMD Vectorization & Fused QKV for Complete Parity with llama.cpp

## Motivation
To close the remaining latency gap and achieve full throughput parity across all models (including Llama 3.2 1B & 3B), this change implements:
1. **Fused QKV Projection Kernel for all Q4_K models**: Combines Wq, Wk, Wv into a single GPU kernel launch per layer (reducing kernel launch count by 56 launches per decode step).
2. **Hardware DP4A SIMD Acceleration**: Replaces scalar byte conversions with hardware `__dp4a` 4-way int8 dot products in 1 clock cycle.
3. **LM Head Tiled Parallelism**: Accelerates 152k vocabulary projection down from 0.85 ms to <= 0.35 ms.
