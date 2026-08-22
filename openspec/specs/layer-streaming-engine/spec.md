# Specification: Layer-Streaming Engine Core

Central system capability: run LLMs whose weights do not fit in VRAM by streaming them per layer (or expert) from pinned RAM into double-buffered GPU memory.

## ADDED Requirements

### Requirement: Single weight load into pinned RAM
The system SHALL load all GGUF model tensors from NVMe into pinned host memory (cudaMallocHost) ONCE at startup, and SHALL NEVER read from disk during generation.

#### Scenario: 0.6B fixture load
- **WHEN** loading a ~400 MB Q4_K_M GGUF begins
- **THEN** tensors land in contiguous pinned regions per layer in <5 s
- **AND** the sum of loaded bytes equals the file size
- **AND** no read() occurs during the generation loop (verifiable via trace)

### Requirement: Double-buffered pipelining with overlap
The system SHALL maintain two CUDA streams (transfer and compute) synchronized by events, so that the H2D copy of layer N+1 overlaps execution of layer N's kernel.

#### Scenario: Overlap measured with Nsight
- **WHEN** the pipeline benchmark runs with a dummy 8-layer model
- **THEN** the trace shows ≥80% of the compute window covered by concurrent transfer
- **AND** total time per layer < sequential (t_copy + t_kernel)

#### Scenario: Dependency ordering
- **WHEN** compute evaluates layer N
- **THEN** it waits on the copy_done[N] event before launching kernels on that buffer
- **AND** there is no CPU busy-waiting

### Requirement: On-the-fly GPU dequantization
Q4_K_M weights SHALL be dequantized inside the GPU kernels (shared memory/registers), without materializing full FP16 copies in VRAM.

#### Scenario: Numerical parity
- **WHEN** GPU dequant is compared block-by-block against the CPU reference
- **THEN** maximum error is < 0.01 per element

### Requirement: Predictable throughput at scale
The system SHALL document real measured throughput per model scale, without theoretical promises lacking benchmarks.

#### Scenario: Dense 14B Q4 on PCIe x8
- **WHEN** generating text with a dense 14B Q4 (~8.5 GB) resident in RAM
- **THEN** expected throughput ≈ 1.4 tok/s (measured, not estimated)
- **AND** the first generated token is valid end-to-end (full layer topology)
