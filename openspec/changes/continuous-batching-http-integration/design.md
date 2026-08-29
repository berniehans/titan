# Design: Continuous Batching HTTP Integration

## Architecture
1. **Axum HTTP Handler:** Parses `ChatCompletionRequest`, creates `(mpsc::UnboundedSender, oneshot::Sender)`, and enqueues `GenerationJob` to `ContinuousBatchManager`.
2. **Background Engine Loop:** 
   - Wakes up on new requests or pending active slots.
   - Admits requests into free GPU slots (up to `max_slots`).
   - Runs batched forward decode step on GPU.
   - Pushes token deltas to client `mpsc` channels for live SSE streaming.
   - Retires finished requests and frees KV blocks.
