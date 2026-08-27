# Proposal: Phase 13 — Large Model Scaling (>6 GB VRAM: 14B & 32B via Layer Streaming Pipeline)

## 1. Summary

Implement the production **Layer Streaming Pipeline Engine** (`StreamingForwardDriver`) that streams arbitrary multi-layer transformer weights (e.g. Qwen 14B ~8.5 GB, Qwen 32B ~19 GB) on-demand through a double-buffered ping-pong VRAM window ($2 \times \text{layer\_size} \approx 600\text{ MB}$ total weight VRAM). Enables running models that vastly exceed physical VRAM (e.g. 14B on 6 GB RTX 3060) with asynchronous PCIe 4.0 DMA transfer and compute overlap.

---

## 2. Motivation

- **Overcoming Physical VRAM Limits:** Standard consumer laptop/desktop GPUs (6 GB - 8 GB VRAM) cannot fit models $> 7\text{B}$. Traditional engines throw CUDA OOM.
- **Core Architecture Activation:** Titan was architected from Phase 1 around zero-copy pinned RAM loading and double-buffered layer transfers. Phase 13 connects this pipeline to the real transformer forward driver, allowing any model size to execute within a bounded $\le 2.0\text{ GB}$ total VRAM budget.
- **PCIe Overlap:** Overlaps the DMA transfer of Layer $L+1$ over PCIe 4.0 with the GPU execution of Layer $L$ using dual CUDA streams and hardware events.

---

## 3. Scope & Sub-Changes

1. **Sub-change 13.1 — Double-Buffered Layer Weight Ring (`engine-cuda` / `engine-core`):**
   - Implement `LayerDoubleBuffer` allocating two ping-pong VRAM slots (`slot_a`, `slot_b`) each sized to hold exactly one layer's quantized weight tensors ($W_q, W_k, W_v, W_o, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$).
   - Zero reallocation during forward pass across all $N$ layers.

2. **Sub-change 13.2 — Dual-Stream Asynchronous DMA Pipeline (`engine-cuda` / `engine-core`):**
   - Implement `StreamingForwardDriver` with `compute_stream` and `transfer_stream`.
   - Pipeline loop: While `compute_stream` executes layer $L$ from `slot[p]`, `transfer_stream` asynchronously DMA-copies layer $L+1$ into `slot[1-p]`, synchronized with `CudaEvent`.

3. **Sub-change 13.3 — Large Model Topology & Golden Parity (`engine-io` / `engine-core`):**
   - Support arbitrary layer counts ($N \ge 48$) and dimension scaling ($H \ge 5120$).
   - Parity gate: Assert bit-exact mathematical logit identity ($\text{cos-sim} = 1.000000$) between `StreamingForwardDriver` and resident `ForwardDriver`.

4. **Sub-change 13.4 — End-to-End Large Model Verification & Phase 13 Seal (`engine-server` / `docs`):**
   - Hook `StreamingForwardDriver` into server runtime and `titan` CLI.
   - Benchmark streaming throughput (tok/s) and VRAM footprint ($\le 2.0\text{ GB}$).
   - Record metrics in `docs/BENCHMARKS.md`, sync delta spec, and archive change.
