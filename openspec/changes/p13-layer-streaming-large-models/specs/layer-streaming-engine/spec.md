# Delta Specification: Phase 13 — Large Model Scaling (>6 GB VRAM: 14B & 32B via Layer Streaming Pipeline)

## ADDED Requirements

### Requirement: Double-Buffered Layer Weight Streaming
The engine SHALL support executing models exceeding total physical VRAM by dynamically streaming transformer layer weights from pinned host RAM to GPU via double-buffered VRAM slots (`slot_a`, `slot_b`).

#### Scenario: Sub-2GB VRAM execution on 14B/32B models
- **WHEN** executing inference on models with parameters $\ge 14\text{B}$ (~8.5 GB - 19 GB weights)
- **THEN** active weight memory in VRAM SHALL remain bounded to exactly 2 resident layers ($< 600\text{ MB}$)
- **AND** total peak device memory across weights, KV cache, and activations SHALL NOT exceed 2.0 GB

### Requirement: Asynchronous Dual-Stream PCIe DMA Overlap
The streaming engine SHALL overlap host-to-device PCIe layer transfer with GPU compute execution using dedicated `TransferStream` and `ComputeStream` synchronized via CUDA events.

#### Scenario: Overlapped layer forward execution
- **WHEN** evaluating layer $L$ on `ComputeStream`
- **THEN** layer $L+1$ SHALL simultaneously transfer over PCIe 4.0 DMA on `TransferStream` without blocking GPU kernel execution
- **AND** output logits SHALL be bit-for-bit identical to non-streamed full-GPU evaluation ($\text{cos-sim} = 1.000000$)
