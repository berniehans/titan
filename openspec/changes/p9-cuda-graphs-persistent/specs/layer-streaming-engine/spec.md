# Delta Specification: Phase 9 — CUDA Graphs & Persistent Decode Kernel

## ADDED Requirements

### Requirement: CUDA Graph Capture and Replay
The system SHALL provide capabilities to capture an arbitrary sequence of CUDA stream operations into an instantiated `CudaGraphExec` and launch the entire graph in a single driver invocation.

#### Scenario: Stream capture and instantiate
- **WHEN** `begin_capture` is initiated on a `CudaStream`, subsequent kernel launches are recorded, and `end_capture` is called
- **THEN** an instantiated executable graph SHALL be returned
- **AND** launching the graph SHALL execute all captured kernels with topological ordering preserved

#### Scenario: Numerical parity of graph replay
- **WHEN** executing a 28-layer transformer forward pass via `CudaGraphExec::launch`
- **THEN** output logits SHALL match standard stream-by-stream execution with cosine similarity $\ge 0.9999$

### Requirement: Graph-Accelerated Decode Forward Driver
The `ForwardDriver` SHALL support capturing its steady-state single-token decode pass into a CUDA graph, updating per-token sequence position dynamically.

#### Scenario: Autoregressive decoding via graph launch
- **WHEN** decoding sequential tokens across multiple generation steps
- **THEN** all 28 layers SHALL execute via graph launch without per-layer host kernel dispatch
- **AND** generated token IDs SHALL be identical to standard decode execution
