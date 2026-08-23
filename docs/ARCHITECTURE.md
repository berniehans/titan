# Titan — Architecture

> **Canonical spec:** [`openspec/specs/layer-streaming-engine/spec.md`](../openspec/specs/layer-streaming-engine/spec.md)
> governs the behavior described here. This document *narrates* the design; the spec
> is the source of truth. Project rules and hardware assumptions live in
> [`openspec/constitution.md`](../openspec/constitution.md).

Titan is a Rust + CUDA LLM inference engine for GGUF models whose **weights do not
fit in VRAM**. Tensors are loaded from NVMe into pinned host RAM **once** at startup
and streamed layer-by-layer to the GPU through a double-buffered pipeline that hides
transfer time behind kernel execution.

## Data-flow

```
NVMe (GGUF file on disk)
  │
  │  Single pass at startup — load_to_pinned(reader, path)
  │  read ONCE; there is NO read() during generation (spec: § Single weight load)
  │  Measured ~0.55 s for the ~400 MB fixture (≈0.7 GB/s, single/coarse pass)
  ▼
pinned host RAM                        ── RAII PinnedHost:
  │                                       cuMemAllocHost / cuMemFreeHost, 4096-B aligned,
  │  per-layer byte slices (layer(idx), tensor(name) via LoadedLayout)     non-pageable for
  │                                                                         async DMA
  │  async H2D copy enqueued per layer (copy_from_host_async on transfer stream)
  ▼
┌──────────────────────────────────────────────────────┐
│          2 ping-pong VRAM slots                      │   DeviceBuffer slots[0..1],
│        slot[N%2]           slot[(N+1)%2]             │   each sized to max_layer_bytes
└──────────────────────────────┬───────────────────────┘
                               │
   ┌───────────────────────────┴───────────────────────────┐
   │  TRANSFER stream (1)            COMPUTE stream (2)     │
   │  • H2D copy of layer N+1        • waits on copy_done[N] │
   │  • records copy_done[N+1]       • then launches kernel  │
   │  • (re)waits compute_done[N-1]  • records compute_done  │
   └───────────────────────────┬───────────────────────────┘
                               │  cuStreamWaitEvent(copy_done[N]) — device-side
                               │  dependency only; NO CPU busy-wait, NO streamSynchronize
                               ▼
                     compute stage
           Phase 2: timed stub (no-op); overlap already measurable
           Phase 3: in-GPU Q4_K_M dequant kernel (dequant in shared
                    memory / registers — no FP16 layer materialized in VRAM)
                               │
                               ▼
                     output / downstream stage (matmul, later phases)
```

The pipeline driver is [`engine/engine-core/src/pipeline.rs`](../engine/engine-core/src/pipeline.rs).
For the exact per-layer sequencing (slot index `N % 2`, `copy_done`/`compute_done`
event ordering), see the linked implementation; the behavioural contract is the spec's
"Double-buffered pipelining with overlap" requirement.

## Crate map

Mirrors [`../README.md`](../README.md) with responsibilities and key types.

| Crate | Responsibility | Key types / items |
|---|---|---|
| `engine-api` | Public engine contracts (API boundary) | `version()` |
| `engine-core` | Orchestration and generation loop; pipeline driver; Q4_K dequant reference | `Pipeline`, `PipelineStats`, `EngineError`, `dequant::dequant_q4k_cpu` |
| `engine-io` | GGUF v3 parser + pinned-memory loader (error-path hardened) | `GgufReader`, `GgufHeader`, `GgufType`, `GgufValue`, `GgmlType`, `TensorInfo`, `LayerIndex` (+`classify_layer`), `LoadedLayout`, `LoadedPinned`, `load_to_pinned`, `GgufError` |
| `engine-cuda` | CUDA FFI via cudarc: pinned host, streams, events, VRAM buffers (RAII) | `PinnedHost`, `CudaStream`, `CudaEvent`, `DeviceBuffer`, `CudaError` |
| `engine-kvcache` | KV cache for attention (later phase; placeholder) | `version()` |

Crate layout is `engine/<name>`. No circular dependencies between crates (constitution §2).

## Key design decisions

### 1. Read-once: weights enter pinned RAM at startup, never `read()` during generation

Weights do not fit in VRAM, so the generation loop is PCIe-bound and latency-critical.
Reading the model from NVMe into pinned host RAM is amortized **once** (~0.55 s) instead
of being paid per token. Any disk I/O inside the generation loop would stall the
double-buffer pipeline unpredictably and defeat overlap entirely. The spec *requires*
this: model tensors SHALL be loaded once into pinned RAM (via `cudaMallocHost`) and
SHALL never be read from disk during generation.

### 2. CUDA events, not stream sync or busy-wait

Two CUDA streams (transfer + compute) run concurrently. The compute stream depends on
each layer's H2D copy through a **`cuStreamWaitEvent` on `copy_done[N]`** (non-blocking
on the CPU, device-side ordering). This is chosen over:

- **`cudaStreamSynchronize`** — blocks until *all* prior work on the stream completes,
  serializing copy and kernel and killing overlap.
- **CPU busy-waiting** — spins the host, wastes power, and adds jitter; unnecessary
  when the hardware already supports event-gated stream dependency.

Result: the H2D copy of layer *N+1* overlaps the kernel of layer *N*, coordinated
entirely on the device. See the Phase 2 proposal
[`openspec/changes/f2-double-buffer-pipeline/proposal.md`](../openspec/changes/f2-double-buffer-pipeline/proposal.md).

### 3. Dequantize inside the GPU kernel — no FP16 materialization in VRAM

Q4_K_M stores 256 weights in a 144-byte super-block (≈4× denser than FP16). Materializing
a full FP16 copy of a layer in VRAM would blow the ~5.2 GB usable budget. Instead the
kernel dequantizes on the fly into shared memory/registers and writes only the tile the
downstream matmul needs. Native Q4_K_M layout: `d`/`dmin` fp16 scales, 12 packed 6-bit
scale/min bytes, 128 nibble-packed weight bytes (see
[`engine/engine-core/src/dequant.rs`](../engine/engine-core/src/dequant.rs)). Numerical
parity against the CPU reference is gated at `< 0.01` per element. Spec:
"On-the-fly GPU dequantization"; proposal
[`openspec/changes/f3-gpu-dequant/proposal.md`](../openspec/changes/f3-gpu-dequant/proposal.md).

### 4. VRAM budget — ~5.2 GB usable on a 6 GB RTX 3060

Fixed assumptions (constitution §4): of the 6 GB, ~5.2 GB is usable. That is split as
**buffers ~0.9 GB**, **activations/driver ~1.3 GB**, and the **remainder for the KV
cache**. This is why weights are streamed in and out rather than resident: only two
layer-sized ping-pong slots are held in VRAM, sized to `max_layer_bytes`, leaving the
bulk of VRAM for the KV cache and activations.

### 5. Pinned (page-locked) host memory

Asynchronous H2D copies (`copy_from_host_async`) require the source to be page-locked:
CUDA's DMA engine can read pinned memory directly and overlap it with kernels. Pageable
host memory would force the driver to stage through a hidden pinned bounce buffer — an
extra copy that serializes and prevents the overlap this engine is built around.
`PinnedHost` uses `cuMemAllocHost`/`cuMemFreeHost` and guarantees 4096-byte alignment
(`PinnedHost::ALIGNMENT`, `engine/engine-cuda/src/pinned_host.rs`).

## Related reading

- Canonical spec: [`openspec/specs/layer-streaming-engine/spec.md`](../openspec/specs/layer-streaming-engine/spec.md)
- Constitution (fixed hardware/scoping rules): [`openspec/constitution.md`](../openspec/constitution.md)
- Phase proposals: [`f0-f1 bootstrap`](../openspec/changes/bootstrap-f0-f1/proposal.md),
  [`f2 double-buffer`](../openspec/changes/f2-double-buffer-pipeline/proposal.md),
  [`f3 gpu-dequant`](../openspec/changes/f3-gpu-dequant/proposal.md),
  [`error-path hardening`](../openspec/changes/hardening-error-paths/proposal.md)
- Measured numbers: [`BENCHMARKS.md`](./BENCHMARKS.md)