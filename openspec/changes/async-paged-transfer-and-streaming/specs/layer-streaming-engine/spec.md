# layer-streaming-engine Delta Specification

## Purpose

Provides double-buffered, event-synchronized PCIe DMA weight streaming enabling out-of-core execution of large models with zero host CPU synchronization bubbles.

## Requirements

### Requirement: Priority Stream DMA Overlapping
The CUDA runtime SHALL create asynchronous transfer streams with dedicated scheduling priorities to pre-fetch upcoming layer/expert weights concurrently with active compute execution.

#### Scenario: Layer N execution with Layer N+1 pre-fetch
- **WHEN** the compute stream executes Layer $ kernels on GPU slot[0]
- **THEN** the transfer stream simultaneously transfers Layer +1$ from pinned host memory into slot[1] via PCIe DMA.

### Requirement: Device-Side Event Synchronization
Stream synchronization SHALL be performed entirely via CUDA device events (cuEventRecord and cuStreamWaitEvent) without CPU thread blocking.

#### Scenario: Layer transition synchronization
- **WHEN** Layer $ compute finishes and Layer +1$ transfer completes
- **THEN** compute stream waits on TransferDone event directly in hardware before starting Layer +1$ compute.
