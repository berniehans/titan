# Design: Asynchronous Paged Transfer & Streaming Pipeline

## Hardware & Bandwidth Architecture
- **PCIe Interface:** PCIe 3.0/4.0 x8/x16 providing 7.0 - 15.8 GB/s practical host-to-device transfer bandwidth.
- **Double Buffering:** Two layer weight buffers slot[0] and slot[1] allocated in GPU VRAM.
- **Dual Stream Topology:**
  1. stream_compute: High priority stream executing GEMV, Attention, and SwiGLU kernels.
  2. stream_transfer: DMA transfer stream executing asynchronous cuMemcpyHtoDAsync from pinned host RAM (PinnedHostBuffer).
- **Synchronization Handshake:**
  - event_transfer_ready[slot]: Signaled by stream_transfer when layer weights are uploaded.
  - event_compute_done[slot]: Signaled by stream_compute when layer finishes execution and slot can be overwritten.
