# radix-prefix-cache Delta Specification

## Purpose
Enables zero-latency prompt prefill bypassing for multi-turn agent conversations by matching Longest Common Prefix in a token Radix tree.

## Requirements
### Requirement: Zero-Prefill on Matching Prefix
The engine SHALL bypass prefill computation for tokens matched in the Radix prefix tree.

#### Scenario: Agent Multi-Turn Cache Hit
- **WHEN** Turn 2 arrives with identical system prompt + tool definitions as Turn 1
- **THEN** prefill latency for the shared prefix is 0 ms.
