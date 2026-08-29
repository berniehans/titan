# Design: FP8 KV-Cache Quantization

## Quantization Scheme
- **Format:** `FP8 E4M3` (1 sign bit, 4 exponent bits, 3 mantissa bits).
- **Scale:** Per-block or per-head FP32 scale factor stored alongside physical block metadata.
- **Attention Kernel:** `flash_decoding_splitk` loads packed 8-bit integers, dequantizes to FP32 registers via `f16_to_f32` / LUT in shared memory, and computes dot products.
