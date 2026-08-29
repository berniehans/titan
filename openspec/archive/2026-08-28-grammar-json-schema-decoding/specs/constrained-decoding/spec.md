# constrained-decoding Delta Specification

## Purpose
Exposes grammar-constrained logit masking over the OpenAI HTTP API for structured JSON generation.

## Requirements

### Requirement: OpenAI response_format JSON Enforcement
The API server SHALL parse `response_format` in `ChatCompletionRequest` and configure grammar logit masking during token sampling.

#### Scenario: Tool call JSON generation
- **WHEN** client sends a request with `response_format: {"type": "json_object"}`
- **THEN** logits of tokens violating JSON grammar are masked to $-\infty$
- **AND** the completed response parses successfully with standard JSON parsers (`serde_json::from_str`).
