# Change: On-the-fly GPU dequantization kernels (Phase 3)

## Why
The Phase 2 pipeline moves raw Q4_K_M bytes but cannot compute yet. This change adds GPU kernels that dequantize Q4_K_M blocks inside the compute stage of the pipeline, without materializing FP16 copies in VRAM. It also gives the compute stream real work, making the double-buffer overlap measurable.

## What Changes
- Implement Q4_K_M block format support in `engine-core` (or new module): super-block of 256 weights = 8 sub-blocks x 32, scales (6-bit) + mins, fp16 scale factors — matching llama.cpp `block_q4_K` layout.
- CPU reference dequantizer (scalar, tested against known-good vectors).
- CUDA kernel (PTX inline asm or cudarc-compatible kernel launch) that dequantizes a device buffer of Q4_K_M blocks into FP16/FP32 in registers/shared memory, writing only the output tile needed by downstream matmul stubs.
- Wire kernel into `Pipeline::run` compute stage: layer bytes → dequant kernel on compute stream after copy_done wait.
- Parity test: GPU output vs CPU reference block-by-block, max error < 0.01 per element.

## Non-goals
- No full GGML matmul / attention yet.
- No other quant formats (Q4_0/Q8_0 later if needed).
- No FP16 materialization of whole layers in VRAM.

## Impact
- **Affected specs:** layer-streaming-engine ("On-the-fly GPU dequantization" requirement)
- **Affected code:** `engine-core/src/dequant.rs` (CPU ref + layout), new kernel module in `engine-cuda`
- **Gate:** parity max error < 0.01/elem vs CPU reference AND benchmark shows real compute work overlapping transfer

## Tasks (summary — details in tasks.md)
1. Q4_K_M layout + CPU reference dequant + unit tests
2. Kernel source + launch wrapper in engine-cuda
3. GPU parity test vs CPU reference
4. Pipeline integration (compute stage calls kernel)
5. Gate: parity + overlap re-benchmark
