## Purpose

Captures repetitive autoregressive decode execution streams into reusable CUDA Graphs (cuGraphExec) to eliminate host driver submission latency and WDDM kernel launch overhead during continuous generation.

## ADDED Requirements

### Requirement: Decode Sequence Graph Capture
The CUDA execution subsystem SHALL capture the full multi-layer autoregressive decode execution stream (stream_decode) into a static CudaGraphExec instance on the first generation step.

#### Scenario: First token decode capture
- **WHEN** executing the first token decode step for an active request
- **THEN** all attention, projection, normalization, and sampling kernels in the decode loop are recorded into a static CUDA Graph without host CPU synchronizations.

### Requirement: Zero-Overhead Graph Replay
The runtime SHALL replay the captured CudaGraphExec instance for subsequent decode steps, executing the complete multi-layer pass with a single cuGraphLaunch invocation.

#### Scenario: Continuous autoregressive generation
- **WHEN** generating subsequent tokens in an active stream
- **THEN** cuGraphLaunch executes all 28 layers in $\le 1\mu\text{s}$ host dispatch latency, maintaining $\ge 90\text{ tok/s}$ throughput on 3B models.
