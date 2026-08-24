# Tasks: 6.6-single-layer-parity

> Execute via bot coder. Strict TDD. One commit per task group. GPU tests: `#[ignore]` + NVRTC PATH trick.

## 1. RED: single-layer wiring
- [ ] 1.1 Failing test: run ONE block over the streamed pipeline (norm→QKV GEMV→RoPE→paged append+attn→out GEMV→residual→norm→SwiGLU→down→residual) on the fixture
- [ ] 1.2 Assert cos-sim > 0.999 vs golden L0 (from 6.1)
- [ ] 1.3 Verify it FAILS (compound drift expected before debug)

## 2. GREEN: wire the block (engine-core/src + engine-cuda)
- [ ] 2.1 Wire MultiFormatGEMV (6.3) for QKV, out, SwiGLU down projections
- [ ] 2.2 Wire norm_rope.cu (6.4) RMSNorm+residual+RoPE at the correct state
- [ ] 2.3 Wire PagedAttention + paged KV append (6.5) into the attention slot
- [ ] 2.4 Residual stream wiring + bias handling
- [ ] 2.5 Re-run parity test; if failing, dump stage traces (2.6)

## 3. Compound-drift bisect (only if Gate fails)
- [ ] 3.1 Dump per-stage outputs (norm state, QKV, attn, out, SwiGLU, down, residual) to temp trace
- [ ] 3.2 Compare each stage vs expected values (CPU bank/known math); fix top-drift stage
- [ ] 3.3 Re-run Gate test

## 4. Gate
- [ ] 4.1 Full suite green
- [ ] 4.2 Gate sealed: cos-sim > 0.999 AND rel-L2 < 1e-3 vs golden L0