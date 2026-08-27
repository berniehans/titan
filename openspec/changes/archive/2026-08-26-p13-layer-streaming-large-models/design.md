# Design: Phase 13 — Large Model Scaling (>6 GB VRAM: 14B & 32B via Layer Streaming Pipeline)

## 1. Streaming Ping-Pong Double-Buffer Architecture

```
Host RAM (Pinned, ~8.5 GB for 14B)
  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
  │ Layer 0 │ Layer 1 │ Layer 2 │ Layer 3 │ Layer 4 │  ... N  │
  └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴─────────┘
       │         │         │         │         │
       │ PCIe DMA Async Transfer Stream (PCIe 4.0 x8 ~6.4 GB/s)
       ▼         ▼         ▼         ▼         ▼
  ┌───────────────────────────────────────────────────────────┐
  │ GPU VRAM Double-Buffer Window (~600 MB weight VRAM total) │
  │   Slot A: [Layer 0] ──► [Layer 2] ──► [Layer 4]           │
  │   Slot B: [Layer 1] ──► [Layer 3] ──► [Layer 5]           │
  └──────────────┬───────────────────┬────────────────────────┘
                 │                   │
                 ▼                   ▼
           Compute Stream      Compute Stream
          (Executes Layer 0)  (Executes Layer 1)
```

---

## 2. Asynchronous Synchronization Timeline

For step $L \in [0, N-1]$:
1. **Pre-condition:** Layer $L$ is ready in `Slot[L % 2]`.
2. **Transfer Launch:** In `transfer_stream`, initiate asynchronous `cuMemcpyHtoDAsync` of Layer $L+1$ into `Slot[(L + 1) % 2]`.
3. **Compute Launch:** In `compute_stream`, launch Layer $L$ forward kernels (RMSNorm, Batched GEMM / GEMV, FlashAttention / PagedAttention, SwiGLU) reading from `Slot[L % 2]`.
4. **Event Barrier:**
   - Record `event_compute_done` on `compute_stream`.
   - Record `event_transfer_done` on `transfer_stream`.
   - Before next layer $L+1$ computes, make `compute_stream.wait_event(event_transfer_done)`.
   - Before next transfer $L+2$ overwrites `Slot[L % 2]`, make `transfer_stream.wait_event(event_compute_done)`.

---

## 3. VRAM Budget Invariants for 14B / 32B Models

| Component | Resident Sizing (14B Q4_K_M) | Resident Sizing (32B Q4_K_M) |
|---|---|---|
| Model Weights | 0 bytes (in pinned RAM) | 0 bytes (in pinned RAM) |
| Double-Buffer Window | $2 \times 177\text{ MB} = 354\text{ MB}$ | $2 \times 296\text{ MB} = 592\text{ MB}$ |
| Paged KV Cache (2048 ctx) | $48 \times 2 \times 8 \times 128 \times 2048 \times 4 = 805\text{ MB}$ | $64 \times 2 \times 8 \times 128 \times 2048 \times 4 = 1.07\text{ GB}$ |
| Scratch Activations & Logits | $\approx 80\text{ MB}$ | $\approx 120\text{ MB}$ |
| **Total Peak VRAM** | **$\approx 1.24\text{ GB}$** ($\le 5.2\text{ GB}$ budget!) | **$\approx 1.78\text{ GB}$** ($\le 5.2\text{ GB}$ budget!) |

**Conclusion:** Both 14B and 32B models run comfortably on a 6 GB RTX 3060 with $> 3\text{ GB}$ of VRAM headroom remaining!
