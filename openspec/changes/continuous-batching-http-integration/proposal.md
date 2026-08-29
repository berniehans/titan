# Proposal: Continuous Batching HTTP Integration

## Summary
Wire the `ContinuousBatchManager` actor into the Axum HTTP `/v1/chat/completions` server loop, enabling up to 4–8 concurrent client/agent requests to be evaluated and streamed concurrently on GPU without mutex blocking.

## Motivation
Local autonomous agent setups (e.g. Hermes Agent with parallel subagents) trigger concurrent LLM calls. A single-request mutex serializes these calls, introducing massive queuing delays. Dynamic continuous batching advances all active request slots simultaneously on GPU.
