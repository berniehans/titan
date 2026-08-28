# Architecture Design: Multi-Slot Continuous Batching & Unified CLI

## 1. Multi-Slot Virtual Agent Scheduler

```
                        CONTINUOUS BATCHING SCHEDULER
                        
     Incoming Agent Requests (HTTP / OpenAI Wire Format / CLI)
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │       Lock-Free Async Request Mailbox        │
        └──────────────────────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
     [ Active Slot 0 ]  [ Active Slot 1 ]  [ Active Slot 2 ]  (Slots 0..N-1)
       Session #101       Session #102       Session #103
            │                  │                  │
            └──────────────────┼──────────────────┘
                               ▼
     ┌─────────────────────────────────────────────────────────┐
     │      Batched Forward Step / CUDA Graph Replay (B=N)     │
     │      (Batched QKV, FlashDecoding Split-KV, SwiGLU)      │
     └─────────────────────────────────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
      Sampled Tok 0      Sampled Tok 1      Sampled Tok 2
            │                  │                  │
            ▼                  ▼                  ▼
      SSE Stream #1      SSE Stream #2      SSE Stream #3
```

- **Slot Lifecycle:** Slots transition between `Idle`, `Prefilling`, `Decoding`, and `Complete`.
- **Dynamic Slot Reallocation:** When a slot completes (via `<|im_end|>` or `max_tokens`), its Paged KV blocks are returned to the block pool or retained in the Radix prefix cache for immediate multi-turn reuse.

## 2. Chunked Prefill

When prompt length $L > 512$:
1. The scheduler partitions the prompt tokens into slices $T_0, T_1, \dots, T_{k-1}$ of size 512.
2. In each iteration, 1 slice $T_i$ is evaluated on GPU while active decode slots advance 1 token.
3. Once $T_{k-1}$ is processed, the request transitions to `Decoding` state.

## 3. Interactive CLI (`engine-cli`)

- Binary: `titan`
- Commands:
  - `titan serve -m <model.gguf> -p <port> -c <ctx> --slots <n>`
  - `titan chat -m <model.gguf>`
  - `titan bench -m <model.gguf>`
  - `titan agent [--hermes-dir <path>]`
