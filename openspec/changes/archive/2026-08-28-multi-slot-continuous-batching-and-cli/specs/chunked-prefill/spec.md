# chunked-prefill Delta Specification

## Purpose
Slices large prompt prefills into discrete chunks to prevent decode latency stalls.

## Requirements
### Requirement: Prompt Chunking
The engine SHALL split prompts larger than $B_{\text{chunk}} = 512$ tokens into discrete passes.

#### Scenario: Prefill during Active Decode
- **WHEN** a new 2,048-token request enters during active decodes
- **THEN** decode step latency remains $<12\text{ ms}$.
