# Proposal: Multi-Slot Continuous Batching, Chunked Prefill & Unified Interactive CLI

## Motivation

As local autonomous agents (e.g. **Hermes Agent**, autonomous tool-calling loops) scale up, single-sequence execution forces concurrent subagents to serialize their inference requests behind global mutexes. Furthermore, large agent system prompts (containing thousands of tokens of OpenAPI tool schemas) introduce TTFT prefill latency spikes that starve active streaming decodes.

This change introduces:
1. **Iteration-Level Continuous Batching & Virtual Slot Scheduler:** Parallelizes up to 4–8 concurrent agent requests across GPU SMs with dynamic slot assignment, non-blocking asynchronous response streaming, and batched CUDA Graph dispatch.
2. **Chunked Prefill & Radix Prefix Eviction:** Interleaves long prompt prefill slices ($B_{\text{chunk}} = 512$) with active decode iterations, ensuring steady $<10\text{ ms}$ decode pacing.
3. **Interactive Terminal CLI (`engine-cli`):** Provides a zero-dependency CLI binary (`titan`) supporting `titan serve`, `titan chat`, `titan bench`, and `titan agent` with real-time GPU telemetry meters.

## Performance Goals & Targets

- **Multi-Agent Concurrency:** Serve 4 concurrent agent streams simultaneously on RTX 3060 Mobile with aggregate throughput $>200\text{ tok/s}$.
- **Decode Jitter with Prefill:** Maintain $<12\text{ ms}$ decode step time even when a new 2,048-token agent request begins prefilling in the background.
- **Prefix Cache Latency:** 0 ms TTFT on repeated agent tool definitions and system prompts.
