## Purpose

Provides on-the-fly Q8_1 activation quantization and NVIDIA hardware integer SIMD __dp4a GEMV execution for quantized LLM inference at wire-saturating throughput.

## ADDED Requirements

### Requirement: On-the-fly Q8_1 Activation Quantization
The GPU resident engine SHALL quantize FP32 hidden-state activation vectors into Q8_1 format (int8 values with FP16 scale and block sum) directly in GPU shared memory / warp registers prior to matrix-vector multiplication.

#### Scenario: Activation vector quantization before GEMV
- **WHEN** a forward decode step produces or normalizes an FP32 activation vector of dimension $N$
- **THEN** the engine quantizes the vector into blocks of 32 int8 elements with pre-computed block scales and sums in under 0.005 ms with zero CPU intervention

### Requirement: Hardware Integer SIMD GEMV Execution (__dp4a)
The GPU GEMV kernels (QKV projection, SwiGLU Gate/Up projection, Down projection, and LM Head) SHALL compute dot products between quantized weights (Q4_K, Q6_K) and Q8_1 activations using the NVIDIA hardware `__dp4a` 4-way 8-bit integer dot product instruction.

#### Scenario: Q4_K matrix-vector multiplication with DP4A
- **WHEN** multiplying a Q4_K weight matrix by an activation vector
- **THEN** each CUDA thread multiplies 4 4-bit weights with 4 8-bit activations in a single clock cycle using `__dp4a`, accumulating into a 32-bit integer register

#### Scenario: Full token decode speed parity
- **WHEN** executing autonomous GPU stream decoding on Qwen 2.5 1.5B Q4_K_M on an NVIDIA RTX 3060 Laptop GPU
- **THEN** the decode throughput SHALL reach 135+ tokens per second, matching or exceeding official llama.cpp performance

### Requirement: Vectorized 128-bit Memory Transactions
The GPU GEMV kernels SHALL load weight blocks using 128-bit aligned vector load instructions (`uint4`) to maximize L1/L2 cache line utilization and eliminate serialized scalar read stalls.

#### Scenario: Coalesced memory streaming
- **WHEN** warps stream weight blocks from global memory
- **THEN** memory transactions are issued as 16-byte aligned vector loads (`uint4`), saturating >90% of physical GPU memory bus bandwidth
