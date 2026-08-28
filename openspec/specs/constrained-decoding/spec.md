# constrained-decoding Delta Specification

## Purpose
Enforces 100% syntactically valid JSON and OpenAPI tool call formatting using grammar-guided logit masking.

## Requirements
### Requirement: JSON Schema Enforcement
The engine SHALL mask all vocabulary tokens that would violate the active JSON grammar state.

#### Scenario: Tool Call Generation
- **WHEN** the agent model generates `<tool_call>` arguments
- **THEN** the emitted output parses as valid JSON with 100% reliability.
