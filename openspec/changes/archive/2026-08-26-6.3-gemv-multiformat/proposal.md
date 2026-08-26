# Change: Multi-format GEMV kernel (`gemv_q4k.cu`) (Phase 6.3)

## Why
The layer-streamed weights are dominated by quantized Q4_K_M tensors, but GGUF files routinely keep the embedding and output/logit head in Q8_0 or F16 (they are rarely Q4_K_M). A Q4-only kernel would leave those paths broken. This change ships the real on-device matmul as a multi-format GEMV: Q4_K_M warp-dequantize-and-dot plus Q8_0 and F16 tensor paths, all in one translation unit.

## What Changes
- `gemv_q4k.cu` (new `engine-cuda` module): Q4_K_M super-block (16 sub-blocks x 32, 6-bit scales/mins) matching llama.cpp `vecdotq.cuh::vec_dot_q4_K_q8_K`.
- Q8_0 path (symmetric per-block scale + dot) and F16 path (plain fp16 dot), selected per tensor group from GGUF metadata.
- Per-block dequant staged in shared memory/registers — no full-layer FP16 materialization in VRAM.
- Wrapper `MultiFormatGEMV` (RAII, safe launch) dispatching Q4K / Q8 / F16 from the tensor's declared format.
- CPU parity reference reused from the 6.2 forward bank.

## Traceability gate (port of llama.cpp)
- Top-of-source comment REQUIRED in `gemv_q4k.cu`:
  `// Port of llama.cpp vcdotq.cuh::vec_dot_q4_K_q8_K (ggml/src/ggml-cuda/vecdotq.cuh @ cb1adf8)`
- Q8/F16 scale layout follows llama.cpp quant conventions at the same pinned commit `cb1adf8`.

## Gate
Teacher-forced parity on REAL fixture tensors (`Qwen3-0.6B-Q4_K_M.gguf`): rel-L2 < 1e-3 vs CPU forward bank AND vs llama.cpp golden L0 activations (from 6.1), for all three formats.

## Non-goals / scope guard
- Single balanced translation unit; NVRTC-only constraint (no cuBLAS dependency).
- No fused MLP/attention matmul here; no fused-MLA-style dim over-engineering.

## Impact
- **Affected code:** `engine-cuda/src/multiformat_gemv.rs`, `gemv_q4k.cu`, `engine-cuda/tests/`
- **Gate:** rel-L2 < 1e-3 vs CPU ref and vs golden L0 on real fixture, all suites green (GPU tests `#[ignore]`, `%LOCALAPPDATA%/Temp` PATH trick for NVRTC)

## Tasks (summary — details in tasks.md)
1. CPU harness + failing Q4_K parity test
2. Implement `gemv_q4k.cu` (all 3 formats) + wrapper
3. Real-tensor parity (CPU 6.2 + llama golden L0)
4. Gate