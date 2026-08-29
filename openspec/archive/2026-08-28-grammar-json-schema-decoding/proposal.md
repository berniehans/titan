# Proposal: Grammar-Constrained JSON Schema Decoding in OpenAI API

## Summary
Wire the grammar-constrained decoding engine (`engine_core::Grammar` and logit masking) to the OpenAI HTTP API `response_format: { "type": "json_object" | "json_schema" }` parameter, guaranteeing 100% syntactically valid JSON outputs for tool calling.

## Motivation
Autonomous agents like Hermes rely heavily on structured tool calls and JSON schemas. If a model outputs malformed JSON or invalid types, agent execution halts. Grammar-constrained logit masking eliminates invalid token transitions on the GPU.
