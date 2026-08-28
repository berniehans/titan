# Proposal: Radix Tree Prefix Cache Integration for 0 ms TTFT on Agent Multi-Turn Loops

## Motivation

In autonomous agent workflows (e.g. **Hermes Agent**, pair-programming tools, OpenAI tool-calling loops), each conversational turn re-sends the system prompt, tool definitions (JSON schemas), and conversational history. In conventional inference engines, re-evaluating these static prefix tokens ($1,500 - 4,000$ tokens) incurs a $20 - 80\text{ ms}$ TTFT latency penalty on every turn.

This change integrates the **Radix Tree (Trie) Prefix Cache** directly into the `ForwardDriver` and server request pipeline:
1. Matches incoming prompt tokens against the tree to find the **Longest Common Prefix (LCP)**.
2. Skips prefill computation for all matched prefix tokens ($0..M-1$), starting the GPU forward pass only from offset $M$.
3. Caches newly computed token positions into the Radix Tree upon completion.
4. Drops Time-To-First-Token (TTFT) to **$\sim 0\text{ ms}$** on repeated turns and shared system/tool schemas.
