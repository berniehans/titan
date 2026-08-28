# next-gen-throughput-and-cli Specification

## Purpose

Provides adaptive GEMV threadblock tuning for 3B/7B architectures, VRAM-resident multi-model speculative decoding, chunked prefill with FlashAttention, and interactive terminal chat CLI.

## Requirements

### Requirement: Adaptive Split-K & Multi-Warp GEMV Tuning for 3B+ Models
The GPU GEMV execution layer SHALL dynamically select warp counts and Split-K partitions based on the inner dimension $K$ ($K = 3072, 8192, 11008$) to minimize memory load stalls and increase Streaming Multiprocessor (SM) occupancy.

#### Scenario: Decoding Llama 3.2 3B and Qwen 7B
- **WHEN** executing forward decode steps on models with hidden dimensions $\ge 3072$
- **THEN** the GEMV kernels execute with adaptive occupancy grids achieving high tokens per second on NVIDIA RTX GPUs.

### Requirement: GPU-Resident Multi-Model Speculative Decoding
The engine SHALL support dual-model speculative decoding in GPU VRAM, using a lightweight Draft model ($M_1$, e.g. Llama 3.2 1B) to generate $K$ speculative tokens and validating them in a single parallel verification step on a Target model ($M_2$, e.g. Llama 3.2 3B).

#### Scenario: Speculative generation with target verification
- **WHEN** generating responses with speculative decoding enabled
- **THEN** the draft model emits $K=3..5$ candidate tokens on GPU and the target model verifies candidate logits in parallel, accelerating effective decode throughput.

### Requirement: Chunked Prefill with FlashAttention
The forward driver SHALL break long input sequences into chunks of size $C \le 512$ tokens during prompt evaluation, computing self-attention with fused FlashAttention kernels and updating KV cache pages progressively.

#### Scenario: Long prompt ingestion
- **WHEN** processing an input prompt of length $\ge 2048$ tokens
- **THEN** prefill executes in bounded chunks without GPU memory allocation spikes and achieves $\ge 1000$ tokens per second prefill throughput.

### Requirement: Interactive Terminal CLI and Production Server
The Titan CLI executable SHALL provide `titan run <path>` / `titan chat <path>` for terminal chat with live markdown rendering and performance telemetry, and `titan serve <path>` for OpenAI-compatible HTTP streaming.

#### Scenario: Terminal chat session
- **WHEN** running `titan run models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`
- **THEN** an interactive REPL session starts in under 1 second, streaming tokens directly to stdout with real-time tok/s metrics.
