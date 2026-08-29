## 1. Actor Architecture & Queue Wiring

- [ ] 1.1 Implement background scheduler task loop in `engine-server/src/server.rs`.
- [ ] 1.2 Replace serialized mutex lock in `/v1/chat/completions` with asynchronous job submission.

## 2. Streaming & SSE Integration

- [ ] 2.1 Pipe `mpsc::UnboundedReceiver` streams into Axum `Sse<impl Stream<Item = Result<Event, Infallible>>>`.
- [ ] 2.2 Validate concurrent generation with multi-client integration test.
