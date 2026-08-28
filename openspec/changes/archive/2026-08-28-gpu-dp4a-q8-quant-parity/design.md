## Context

See `proposal.md` - Why. Titan executes autonomous GPU decode steps via CUDA Graphs, but its GEMV kernels perform FP32 floating-point dequantization and arithmetic. Implementing on-the-fly Q8_1 quantization and `__dp4a` integer SIMD instructions will saturate GPU wire memory bandwidth.

## Goals / Non-Goals

**Goals:**
- Reach 140+ tok/s (sub-7.1 ms latency) on Qwen 2.5 1.5B Q4_K_M on NVIDIA RTX 3060 Laptop GPU.
- Implement on-the-fly `Q8_1` activation quantization with per-block scale and sum.
- Upgrade `gemm_q4k_kernel`, `gemm_fused_qkv_kernel`, `gemm_q4k_fused_gate_up_swiglu_kernel`, and `gemm_q4k_splitk_kernel` to use `__dp4a`.
- Maintain 100% pure Rust, zero external C++ / Python dependencies, and zero precision degradation.

**Non-Goals:**
- Modifying CPU inference pipeline or GGML file loaders.
- Adding non-NVIDIA GPU backends (e.g. Vulkan / Metal).

## Decisions

### Decision 1: Fused In-Kernel Q8_1 Quantization vs Separate Pre-pass Kernel
- **Chosen Approach**: Perform Q8_1 activation quantization directly inside the GEMV shared memory loader (`load_shared_x_with_optional_norm_and_q8_1`).
- **Rationale**: Eliminates a separate kernel launch and global memory round-trip. Each thread block normalizes and quantizes its required activation slice directly into shared memory.
- **Alternatives Considered**: Dedicated global quantization kernel before each GEMV layer (rejected due to kernel launch queue and VRAM read/write overhead).

### Decision 2: Hardware `__dp4a` SIMD Instruction for Q4_K x Q8_1 Dot Products
- **Chosen Approach**: Unpack 4-bit weights into two 32-bit registers containing 4 bytes each, and multiply against `Q8_1` activations using `__dp4a(weights, activations, acc)`.
- **Rationale**: `__dp4a` computes 4 multiply-accumulates in 1 clock cycle on NVIDIA Ampere CUDA cores, cutting ALU instruction cycles by >75%.
- **Alternatives Considered**: FP32 FMA arithmetic (rejected because it is ALU-bound and cannot saturate memory bandwidth).

### Decision 3: 128-bit Vectorized Memory Streaming (`uint4`)
- **Chosen Approach**: Issue memory transactions as 16-byte aligned vector loads (`uint4`), reading 32 weights at a time per thread.
- **Rationale**: Saturates memory bus lines and prevents warp serialization on 1-byte `LD.GLOBAL.U8` instructions.

## Risks / Trade-offs

- [Risk: Numerical Drift from Integer Quantization] → **Mitigation**: `Q8_1` quantization uses block size 32 with exact FP16 scale and block sum accumulation, identical to official GGML `ggml-cuda/mmvq.cu`.
- [Risk: Shared Memory Limits with Q8_1 + RMSNorm] → **Mitigation**: Q8_1 uses only 1 byte per element (1.5 KB for 1536 hidden dim), fitting comfortably in the 100 KB shared memory per SM on RTX 3060 with 100% warp occupancy.
