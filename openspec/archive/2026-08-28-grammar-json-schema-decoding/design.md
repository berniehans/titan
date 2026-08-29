# Design: Grammar-Constrained JSON Schema Decoding

## Architecture
1. **Schema Parsing:** When `response_format` specifies `json_object` or `json_schema`, a `Grammar` state machine is instantiated.
2. **Logit Masking:** In `Sampler::sample`, if grammar constraint is active, `gpu_logit_mask` sets non-allowed token logits to $-\infty$.
3. **State Transition:** Upon sampling token $t$, the grammar transitions to `next_state(t)`.
