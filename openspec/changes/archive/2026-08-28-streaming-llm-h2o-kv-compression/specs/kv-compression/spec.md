# kv-compression Delta Specification

## Purpose
Provides constant-memory streaming KV cache management using StreamingLLM Attention Sinks and H2O block recycling.

## Requirements
### Requirement: Constant Memory Infinite Streaming
The engine SHALL maintain generation without out-of-memory errors past physical cache capacity by evicting non-sink blocks.

#### Scenario: 1,000+ Token Stream in 256-Token Budget
- **WHEN** generating sequences longer than the physical block pool budget
- **THEN** attention sinks (initial 4 tokens) and rolling window are preserved, and perplexity remains stable.
