# Proposal: Grammar-Guided Constrained Decoding & Logit Masking for 100% Guaranteed JSON Tool Calling

## Motivation

Autonomous agent tool-calling loops (Hermes Agent, LangChain, Cursor) depend on receiving strictly valid structured JSON adhering to an OpenAPI function schema (`{"name": "...", "arguments": {...}}`). Unconstrained generation can produce syntax errors (unbalanced braces, unquoted keys, trailing commas, missing colons), requiring expensive retries or causing tool execution failures.

This change introduces:
1. **Grammar-Guided Token Masking Engine:** A deterministic state-machine parser in `engine-core/src/grammar.rs` tracking JSON and tool call states.
2. **GPU Logit Masking Kernel:** A fast CUDA kernel (`apply_logit_mask_kernel`) masking invalid tokens directly in VRAM prior to sampling.
3. **OpenAI `response_format: { "type": "json_object" }` & `tools` integration:** Automatically activates the JSON schema constraint when tools are specified.
