## 1. Implementation Tasks
- [x] 1.1 Add `gemm_fused_qkv_q4k_kernel` to `engine-cuda/kernels/gemm_quant.cu`.
- [x] 1.2 Enable fused QKV dispatch in `ForwardDriver::decode_single_token` and `ForwardDriver::decode_batched`.
- [x] 1.3 Add head_dim=64 RoPE specialization and strict grammar JSON decode anti-looping rules.
- [x] 1.4 Benchmark decode throughput across all models.
