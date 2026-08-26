# Layer-Streaming Engine Core Specification

## Purpose
Run LLMs whose weights do not fit in VRAM by streaming them per layer (or per expert) from pinned host RAM into double-buffered GPU memory with overlapped transfers, on-the-fly dequantization, paged KV-cache attention, real forward execution, and bandwidth-adaptive hybrid MoE decode.

## Requirements

### Requirement: Single weight load into pinned RAM
The system SHALL load all GGUF model tensors from NVMe into pinned host memory (`cudaMallocHost`) ONCE at startup, and SHALL NEVER read from disk during generation.

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
Q4_K_M weights SHALL be dequantized inside GPU kernels (shared memory/registers), without materializing full FP16 copies in VRAM.

#### Scenario: Numerical parity
- **WHEN** GPU dequant is compared block-by-block against the CPU reference
- **THEN** maximum error is < 0.01 per element

### Requirement: Paged KV-Cache management
The system SHALL allocate KV-cache in non-contiguous physical memory blocks with paged attention indexing, enforcing hard upper bounds on total block allocations.

#### Scenario: Block allocation and exhaustion
- **WHEN** attention tokens are appended across non-contiguous blocks
- **THEN** reads reconstruct the logical sequence identically
- **AND** pool exhaustion returns a typed OutOfMemory error without device crash

### Requirement: Real forward driver and generator parity
The system SHALL execute full-topology transformer forward passes combining GPU GEMV, Norm/RoPE/SwiGLU, and PagedAttention, producing next-token logits that match golden reference vectors.

#### Scenario: Golden logit cosine similarity
- **WHEN** prefill evaluates a teacher-forced prompt against llama.cpp reference logits
- **THEN** cosine similarity SHALL be >= 0.99
- **AND** total transient VRAM usage SHALL remain bounded within the 5.2 GB envelope

### Requirement: MoE expert streaming and bandwidth-adaptive hybrid decode
The system SHALL stream MoE expert weights dynamically over PCIe into a GPU slot cache, balancing host CPU execution and PCIe DMA transfers using hardware bandwidth profiling (`_balanced_fetch`).

#### Scenario: Hardware bandwidth profiling and balanced fetch
- **WHEN** bandwidth profile is recorded in `benchbw.json`
- **THEN** fetch fraction $q^\star = \text{pcie\_ov} / (\text{pcie\_ov} + \text{cpu\_ov})$ is resolved
- **AND** balanced rounding minimizes the longer execution path ($0.415 \times 3 \rightarrow 1$ fetch)
- **AND** resident hits, PCIe fetches, and CPU overflows are tracked per layer

### Requirement: Native GPU Q6_K dequantization and multi-format GEMV
The system SHALL provide native CUDA kernels for dequantizing `Q6_K` super-blocks (256 weights in 210 bytes) and executing fused matrix-vector products directly on GPU registers without CPU fallback or intermediate VRAM allocations.

#### Scenario: Q6_K GPU block-level parity
- **WHEN** GPU `dequant_q6k` unpacks raw Q6_K byte buffers
- **THEN** output floats SHALL match CPU reference floats with maximum relative error $< 10^{-4}$
- **AND** cosine similarity SHALL exceed 0.9999

#### Scenario: Full-layer GPU execution without host synchronization
- **WHEN** `ForwardDriver` executes decode steps across layers containing mixed Q4_K_M and Q6_K tensors
- **THEN** all matrix multiplications SHALL execute on CUDA streams
- **AND** no intermediate layer activations SHALL synchronize or transfer to host CPU memory

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

### Requirement: OpenAI Chat Completions Wire Protocol
The server SHALL expose standard endpoints conforming to the OpenAI REST API specification:
- `POST /v1/chat/completions`: accepts messages array (`system`, `user`, `assistant`), optional sampling controls, and streaming flag.
- `GET /v1/models`: returns list of currently loaded model identifiers.

#### Scenario: Non-streaming Chat Completion
- **WHEN** client sends a `POST /v1/chat/completions` request with `stream: false`
- **THEN** server SHALL return JSON object with `choices[0].message.content` and `usage` token accounting

#### Scenario: Streaming Server-Sent Events (SSE)
- **WHEN** client sends a `POST /v1/chat/completions` request with `stream: true`
- **THEN** server SHALL emit chunks of type `text/event-stream` formatted as `data: {"choices": [{"delta": {"content": "..."}}]}\n\n`
- **AND** terminate the stream with `data: [DONE]\n\n` upon reaching stop sequence or max tokens

### Requirement: Advanced Sampling and Stop Sequence Control
The inference pipeline SHALL support configurable sampling parameters to control generation randomness and termination:
- Temperature scaling (with greedy argmax when $\le 10^{-4}$)
- Top-$K$ and Top-$P$ (nucleus) probability filtering
- Repetition penalty
- Stop tokens / custom stop word sequences

#### Scenario: Stop sequence trimming
- **WHEN** model generates `<|im_end|>` or any configured stop word sequence
- **THEN** generation SHALL terminate immediately with finish reason `stop`
- **AND** the stop token itself SHALL NOT be appended to user-visible content

### Requirement: Interactive Terminal CLI
The engine binary SHALL provide an interactive command-line interface (`titan chat`) for direct terminal conversation with live token-by-token streaming.

### Requirement: Batched Quantized GEMM Execution
The system SHALL provide native CUDA matrix multiplication kernels (`gemm_q4k`, `gemm_q6k`, `gemm_q80`) computing $Y = X W^T$ for batch sizes $M \ge 1$ without materializing uncompressed weights in VRAM.

#### Scenario: Batched GEMM numerical parity
- **WHEN** computing $Y = X W^T$ for activation matrix $X \in \mathbb{R}^{M \times K}$ ($M \in \{16, 64, 128, 256\}$) against quantized weight tensor $W$
- **THEN** output floats SHALL match CPU reference batched GEMM with maximum relative error $< 10^{-4}$
- **AND** cosine similarity SHALL exceed 0.9999

### Requirement: FlashAttention-2 Causal Prefill
The system SHALL provide a tiled CUDA FlashAttention-2 kernel computing causal self-attention and cross-attention over resident paged KV blocks with $O(S)$ intermediate VRAM usage.

#### Scenario: FlashAttention-2 causal parity
- **WHEN** executing causal prefill attention on sequence length $S \in [1, 2048]$
- **THEN** output attention vectors SHALL match CPU reference causal attention with cosine similarity $\ge 0.9999$
- **AND** peak temporary attention memory consumption SHALL remain bounded by shared-memory tile sizes ($B_r \times B_c$) without allocating $S \times S$ global memory matrices

### Requirement: Chunked Prefill Forward Pipeline
The `ForwardDriver` SHALL support chunked prefill evaluation, splitting long prompts into bounded token slices ($S_{\text{chunk}} \le \text{CHUNK\_SIZE}$) and computing full forward passes in parallel batches.

#### Scenario: Multi-token prompt evaluation
- **WHEN** evaluating a multi-token prompt of length $S$
- **THEN** prefill SHALL execute via batched GEMM and FlashAttention-2 chunks
- **AND** final output logits SHALL match single-token serial prefill with cosine similarity $\ge 0.997$
- **AND** Time To First Token (TTFT) SHALL achieve at least $5\times$ speedup on prompts $\ge 128$ tokens

### Requirement: Predictable throughput at scale
The system SHALL document real measured throughput per model scale, without theoretical promises lacking benchmarks.

#### Scenario: Dense 14B Q4 on PCIe x8
- **WHEN** generating text with a dense 14B Q4 (~8.5 GB) resident in RAM
- **THEN** expected throughput ≈ 1.4 tok/s (measured, not estimated)
- **AND** the first generated token is valid end-to-end (full layer topology)
