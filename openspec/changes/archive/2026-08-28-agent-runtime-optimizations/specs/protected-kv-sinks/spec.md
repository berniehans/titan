# protected-kv-sinks Delta Specification

## Purpose

Provides Attention Sinks retention and protected sliding-window KV management to prevent VRAM Out-Of-Memory (OOM) failures during long multi-turn agent sessions.

## Requirements

### Requirement: Attention Sinks Preservation
The PagedAttention kernel and KV scheduler SHALL permanently retain the first $K_{\text{sink}} = 4$ initial sequence tokens (*Attention Sinks*) across all context rollouts to prevent softmax score divergence.

#### Scenario: Long-context sequence evaluation exceeding window size
- **WHEN** the token position $\text{pos}$ exceeds the configured sliding window length $W$
- **THEN** the attention kernel accumulates attention over the sink tokens $[0, 4)$ and the active rolling window $[\text{pos} - W, \text{pos}]$, ignoring intermediate evicted tokens.

### Requirement: Protected Sliding Window & Memory Eviction
The KV-cache manager SHALL evict or offload intermediate tool execution logs outside the active window once memory consumption exceeds the VRAM guard watermark ($90\%$ of usable capacity).

#### Scenario: High-volume tool execution output
- **WHEN** an agent executes multiple commands producing large output logs
- **THEN** unpinned intermediate blocks are systematically reclaimed, keeping total GPU VRAM allocation bounded within $4.8\text{ GB}$ on the 6 GB RTX 3060.
