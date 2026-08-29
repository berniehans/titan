## 1. Actor Architecture & Queue Wiring

- [x] 1.1 Implement background scheduler task loop in `engine-server/src/server.rs`.
- [x] 1.2 Replace serialized mutex lock in `/v1/chat/completions` with asynchronous job submission.

## 2. Streaming & SSE Integration

- [x] 2.1 Pipe `mpsc::UnboundedReceiver` streams into Axum `Sse<impl Stream<Item = Result<Event, Infallible>>>`.
- [x] 2.2 Validate concurrent generation with multi-client integration test.
