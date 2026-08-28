## 1. Grammar State Machine & Logit Masking Engine

- [ ] 1.1 Implement `JsonGrammar` state machine in `engine-core/src/grammar.rs`.
- [ ] 1.2 Implement `Sampler::sample_constrained` with logit masking.
- [ ] 1.3 Add `apply_logit_mask_kernel` in `engine-cuda/kernels/norm_rope.cu`.
- [ ] 1.4 Add integration test in `engine-server/tests/grammar_constrained_tool_call_test.rs`.
