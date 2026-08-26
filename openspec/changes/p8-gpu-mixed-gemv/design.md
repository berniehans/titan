# Design Document: Phase 8 — Native GPU Mixed-Quant GEMV

## Architecture & Data Flow

```
Raw Pinned RAM (Q6_K Tensor)
          │
          │  DMA Copy on Transfer Stream (H2D)
          ▼
DeviceBuffer (GPU Ping-Pong Slot)
          │
          │  Fused Launch: gemv_q6_k<<<blocks, threads, smem, stream>>>
          ▼
┌─────────────────────────────────────────────────────────────┐
│  CUDA SM (Shared Memory & Warp Registers)                   │
│  1. Unpack 210-B superblock (128B ql, 64B qh, 16B scales)    │
│  2. Reconstruct q6: (ql_low | ((qh_low & 3) << 4)) - 32     │
│  3. Scale: d * d8 * q6                                      │
│  4. Warp Fused Multiply-Add with activation x_dev[k]        │
│  5. Warp Shuffle Reduction into y_dev[row]                  │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
Output Activation DeviceBuffer (y_dev)
(Remains purely in VRAM for subsequent Norm/RoPE/Attention stages)
```

## 1. Q6_K Superblock Layout & Bit Unpacking in CUDA

A `Q6_K` super-block quantizes 256 weights into **210 bytes**:
- `ql`: `[u8; 128]` (128 bytes) — lower 4 bits of the 256 quants (packed 2 quants per byte).
- `qh`: `[u8; 64]` (64 bytes) — upper 2 bits of the 256 quants (packed 4 quants per byte: 2 bits each).
- `scales`: `[i8; 16]` (16 bytes) — 8-bit signed scale factors for each 16-weight sub-block.
- `d`: `f16` (2 bytes) — base super-block scale factor (IEEE 754 half precision).

### Mathematical Reconstruction:
For sub-block $j \in [0, 15]$ (each 16 weights):
$$w_{j \times 16 + i} = d \cdot \text{scales}[j] \cdot (q_6 - 32)$$
where $q_6 \in [0, 63]$ is formed by extracting the lower 4 bits from `ql` and the corresponding 2 bits from `qh`.

## 2. Fused `gemv_q6_k` Kernel Design

Rather than materializing dequantized weights in intermediate VRAM, the GEMV kernel loads the 210-byte superblocks directly into shared memory and warp registers:
- **Grid Strategy:** Each CUDA block computes 1 or 2 rows of the output vector.
- **Warp Level Reduction:** Threads in the warp unpack 8 or 16 elements each, perform FMAs with `x[col]`, and execute `__shfl_down_sync` warp reductions.
- **Memory Coalescing:** Superblocks are read in 128-bit vectorized loads (`uint4`).

## 3. Integration into `MultiFormatGEMV` & `ForwardDriver`

- In `engine-cuda/src/multiformat_gemv.rs`, add `gemv_q6_k` alongside existing `gemv_q4_k`, `gemv_q8_0`, and `gemv_f16`.
- Update `ForwardDriver` to execute `attn_v`, `ffn_down`, and `token_embd` on GPU via `MultiFormatGEMV` instead of routing through `cpu_reference_bank`.
- Keep activation buffers device-resident across the entire layer stack.
