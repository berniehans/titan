# Change: CPU reference forward bank on a real-GGUF synthetic (Phase 6.2)

## Why
GPU kernels (6.3-6.5) and layer wiring (6.6) need an independent CPU authority to validate against. That authority must NOT be derived by transliterating our own CUDA back — that would share the same bugs between twin implementations. It is written from the paper formulas / ggml.c semantics and runs over a minimal REAL `.gguf` synthetic carrying the same tensor names and the same quant formats (incl. non-Q4 embedding/head).

## What Changes
- Minimal real `.gguf` synthetic under `tests/fixtures/synthetic/` (same naming convention as llama.cpp fixture; Q4_K_M + Q8_0 + F16 tensors).
- FP32 CPU-only forward bank `engine-core/src/forward_cpu.rs`: RMSNorm, quantized matmul (via dequant→dot), RoPE, attention, residual stream, SwiGLU MLP, output logits.
- Readable CPU authority for ParceledParity (6.3), fused norm/rope/swiglu parity (6.4) and paged attention (6.5).
- Cross-validated against llama.cpp golden layer-0 (from 6.1) at cos-sim ≥ 0.9999.

## Why this is not a numeric transliteration of our CUDA
By policy, kernels are ports of upstream C; the CPU reference is written from paper formulas / ggml.c semantics — NOT by copying the CUDA kernel back. Traceability note: `forward_cpu.rs` cites `/ggml.c` RMSNorm eps placement and RoPE (NeoX partial) conventions, not a line-for-line port.

## Impact
- **Non-goals:** no GPU kernels here (this bank is CPU), no throughput optimization, no batching beyond single-sequence reference.
- **Affected code:** `engine-io` synthetic fixture loader, `engine-core/src/forward_cpu.rs`, `tests/fixtures/synthetic/*`
- **Gate:** synthetic logits bit-exact vs CPU bank; bank vs llama.cpp golden L0 cos-sim ≥ 0.9999

## Tasks (summary — details in tasks.md)
1. Synthetic GGUF + loader tests
2. CPU forward core (RMSNorm/RoPE/attention/SwiGLU/logits)
3. Bit-exactness + cross-validation vs golden L0
4. Gate

## Environment notes
- NVRTC not used this change (CPU-only). llama.cpp pinned `cb1adf8`, binaries `%LOCALAPPDATA%/llama.cpp/build/bin/Release/`.