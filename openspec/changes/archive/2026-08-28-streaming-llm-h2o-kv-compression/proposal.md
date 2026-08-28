# Proposal: Dynamic KV Cache Compression with StreamingLLM Attention Sinks & H2O Eviction for Infinite Context in 6 GB VRAM

## Motivation

In local consumer GPU environments (such as RTX 3060 Mobile with 6 GB VRAM), long-running autonomous agent conversations, code generation, and multi-turn loops exceed the available KV cache memory capacity as sequence length grows beyond 4,096 - 8,192 tokens.

This change implements:
1. **StreamingLLM Attention Sinks:** Pins initial $S=4$ initial tokens (absorbing softmax mass) to prevent perplexity collapse.
2. **Recent Rolling Window:** Retains the latest $W$ tokens for immediate conversational context.
3. **H2O Dynamic Heavy-Hitter Eviction:** Selectively evicts non-critical intermediate KV blocks when physical pool budget is exhausted.
4. **Bounded VRAM Footprint:** Ensures KV cache memory usage is $\mathcal{O}(1)$ with respect to sequence length, allowing infinite generation without OOM.
