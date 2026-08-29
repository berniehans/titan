# Proposal: FP8 KV-Cache Quantization

## Summary
Implement FP8 (`E4M3`/`E5M2`) quantization for paged Key and Value cache pools, doubling context window capacity under the 6 GB VRAM budget (from 2048 to 4096+ tokens) and reducing memory bandwidth traffic in FlashDecoding.

## Motivation
At 2048 context length on Llama 3.2 3B, FP16 KV pairs consume significant bandwidth and memory. Storing keys and values in 8-bit floats reduces KV pool footprint by 50% with negligible perplexity degradation (< 0.05).
