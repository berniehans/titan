## 1. Radix Prefix Cache Engine Integration

- [ ] 1.1 Integrate `RadixTree` into `ForwardDriver` in `engine-core/src/forward_driver.rs`.
- [ ] 1.2 Implement prefix match check in `ForwardDriver::prefill` and suffix-only execution.
- [ ] 1.3 Add multi-turn agent cache hit verification test in `engine-server/tests/radix_prefix_cache_test.rs`.
- [ ] 1.4 Benchmark TTFT speedup with and without prefix cache.
