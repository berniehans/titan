## 1. Multi-Model Driver Allocation & Context Sizing

- [ ] 1.1 Support dual-model concurrent residency in ForwardDriver without VRAM fragmentation.
- [ ] 1.2 Implement batched speculative candidate verification in ForwardDriver::verify_speculative_batch.

## 2. Fast Speculative Sampling & KV Synchronization

- [ ] 2.1 Implement SpeculativeOrchestrator supporting greedy and stochastic rejection sampling.
- [ ] 2.2 Implement virtual BlockTable branch and fast rollback upon rejection.

## 3. End-to-End Benchmark & Throughput Validation

- [ ] 3.1 Verify speculative output token identity against standalone 3B target in speculative_speedup_bench.
- [ ] 3.2 Measure end-to-end decode throughput and verify $\ge 120\text{ tok/s}$.
