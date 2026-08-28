## 1. Multi-Slot Continuous Batch Scheduler (`engine-server` & `engine-core`)

- [ ] 1.1 Implement dynamic multi-slot allocation and state machine in `engine-server/src/scheduler.rs`.
- [ ] 1.2 Implement batched decode step in `engine-core/src/forward_driver.rs` for batch sizes $B \in \{1, 2, 4\}$.
- [ ] 1.3 Add multi-slot concurrency test in `engine-server/tests/multi_slot_concurrency_test.rs`.

## 2. Chunked Prefill & Radix Prefix Reuse (`engine-core`)

- [ ] 2.1 Implement `ForwardDriver::chunked_prefill()` in `engine-core/src/forward_driver.rs`.
- [ ] 2.2 Add chunked prefill jitter verification test in `engine-server/tests/chunked_prefill_bench.rs`.

## 3. High-Performance Unified CLI (`engine-cli`)

- [ ] 3.1 Create `engine/engine-cli` crate in workspace with `clap` CLI parser.
- [ ] 3.2 Implement `titan serve`, `titan chat`, `titan bench`, and `titan agent` commands.
- [ ] 3.3 Add real-time ANSI terminal streaming and GPU metrics rendering.
