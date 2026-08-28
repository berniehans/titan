# Architecture Design: Ampere Tensor Cores Acceleration (`mma.sync`)

## 1. Technical Overview & PTX Instruction Layout

On NVIDIA Ampere GPUs (Compute Capability 8.0 & 8.6), Tensor Cores feature the `mma.sync` instruction for sub-byte and integer matrix multiply-accumulate operations:

```cuda
// Instruction: mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32
// Computes D = A * B + C for M=16, N=8, K=32 in 1 warp instruction cycle.
asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 "
    "{%0, %1, %2, %3}, "      // D: 4 x 32-bit integer accumulators
    "{%4, %5}, "              // A: 2 x 32-bit registers (8 x 4-bit nibbles each = 16 weights)
    "{%6}, "                  // B: 1 x 32-bit register (8 x 4-bit nibbles = 8 activation weights)
    "{%7, %8, %9, %10};"      // C: 4 x 32-bit input accumulators
    : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
    : "r"(a0), "r"(a1), "r"(b0),
      "r"(c0), "r"(c1), "r"(c2), "r"(c3)
);
```

## 2. Micro-Architecture Pipeline & Memory Layout

```
                        TENSOR CORE GEMV EXECUTION FLOW
                        
  Global VRAM (GGUF Q4_K Weights)        Shared Memory (Q8_1 Activations)
         │                                            │
         ▼ (128-bit uint4 Vectorized Load)            ▼ (L1 / Smem BroadCast)
  Warp Register Bank [A0, A1]              Warp Register Bank [B0]
         │                                            │
         └───────────────────┬────────────────────────┘
                             ▼
              [ Tensor Core Pipeline (m16n8k32) ]
                             │
                             ▼ (Hardware Accumulate)
                   Accumulator Registers [D0..D3]
                             │
                             ▼ (Fused Scale Dequantization: out = d_w * d_a * sum)
                     FP32 Output Vector
```

### Scale Hoisting & Math Correctness:
For `Q4_K` blocks (256 weights, 8 sub-blocks of 32 weights with 6-bit scales):
$$\text{Weight}_{i} = (\text{nibble}_i) \cdot d \cdot \text{scale}_s - d_{\text{min}} \cdot \text{mins}_s$$
By computing the integer dot product in hardware:
$$\sum \text{Weight}_i \cdot \text{Act}_i = d \cdot \text{scale}_s \sum (\text{nibble}_i \cdot \text{Act}_i) - d_{\text{min}} \cdot \text{mins}_s \sum \text{Act}_i$$
The sum of activations $\sum \text{Act}_i$ is pre-calculated once during `quantize_row_q8_1_kernel` and stored in shared memory, allowing the entire 256-element block to be computed with 8 `mma.sync` instructions with zero register spill.

## 3. Fused MMA SwiGLU Kernel

Instead of launching separate kernels for Gate and Up projections:
1. One thread block loads a pair of output rows $(W_{\text{gate}}[r], W_{\text{up}}[r])$.
2. Executes dual-stream MMA accumulations in Tensor Core registers.
3. Applies in-place scale dequantization and evaluates SiLU in registers:
   $$\text{SwiGLU}(r) = \left( \text{Gate}_r \cdot \frac{1}{1 + e^{-\text{Gate}_r}} \right) \cdot \text{Up}_r$$
4. Writes the fused result directly to `zh_dev`, cutting DRAM memory traffic by 50%.
