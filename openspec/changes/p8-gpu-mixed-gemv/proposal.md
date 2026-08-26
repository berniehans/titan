# Change Proposal: Phase 8 — Native GPU Dequantization and GEMV for Mixed Quantization (Q6_K / Q8_0)

## Motivation & Problem Statement
In Phase 6 and Phase 7, Titan established full numerical parity and coherent autoregressive generation with streaming SSE and MoE execution under strict VRAM bounds. However, in the reference Qwen3-0.6B model (and many production GGUF models), weights are quant-mixed: while attention query/key/output and FFN gate/up projections use `Q4_K_M`, critical projections (`attn_v`, `ffn_down`, and `token_embd`) are formatted in `Q6_K`.

Currently, `ForwardDriver` falls back to CPU arithmetic for `Q6_K` tensors, causing costly device-to-host-to-device synchronizations and CPU compute per layer, resulting in ~0.16 tok/s.

## Objective
Implement native GPU CUDA dequantization and fused GEMV kernels for `Q6_K` and multi-format tensors, lifting 100% of the transformer forward pass onto the GPU. This eliminates CPU layer roundtrips and accelerates autoregressive generation from ~0.16 tok/s to >15 tok/s on local RTX 3060 hardware.

## Scope of Changes
1. **Sub-change 8.1:** CUDA `dequant_q6k` kernel with bit-level unpacking of 210-byte superblocks and block-by-block numerical parity verification against CPU reference.
2. **Sub-change 8.2:** Fused GPU `MultiFormatGEMV` extension for `Q6_K` weight matrices and FP32 activation vectors in CUDA shared memory/warp registers.
3. **Sub-change 8.3:** GPU embedding lookup for `token_embd` and full GPU wiring in `ForwardDriver`, guaranteeing zero CPU syncs during layer decode.
4. **Sub-change 8.4:** End-to-end benchmark measurement, latency profiling, and validation seal in `docs/BENCHMARKS.md`.

## Non-Goals
- Modifying the underlying GGUF parser or PagedAttention block structures (already stabilized in Phase 4-6).
- Replacing FP32 KV-cache activation representations (kept as standard).

## Success Criteria & Gates
- **Numerical Parity:** GPU `Q6_K` dequant kernel matches CPU reference with relative L2 error $< 10^{-4}$ and cosine similarity $> 0.9999$.
- **Zero CPU Fallbacks:** All 28 layers of Qwen3 execute purely on CUDA streams without host synchronization during steady-state decode.
- **Throughput Speedup:** Autoregressive decode throughput achieves $\ge 15.0$ tok/s on RTX 3060 (a $>50\times$ speedup over the 0.16 tok/s baseline).
