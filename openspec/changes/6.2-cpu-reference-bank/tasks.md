# Tasks: 6.2-cpu-reference-bank

> Execute via bot coder. Strict TDD. One commit per task group.

## 1. Synthetic GGUF + loader (engine-io)
- [ ] 1.1 Failing test: engine-io loads the synthetic minimal `.gguf` (tensor names mirroring llama.cpp fixture), resolves tensor→shape+type
- [ ] 1.2 Build the synthetic GGUF under `tests/fixtures/synthetic/` (2 layers, Q4_K_M + Q8_0 + F32 embed/head)
- [ ] 1.3 Test metadata + type resolution round-trip on the synthetic

## 2. CPU forward bank (engine-core/src/forward_cpu.rs)
- [ ] 2.1 Failing test: RMSNorm reference vs hand-computed vector (rms_eps placement per ggml.c)
- [ ] 2.2 Implement RMSNorm + residual helper + RoPE (Qwen3 NeoX partial) references
- [ ] 2.3 Failing test: quantized matmul (dequant→dot) returns expected FP32 on controlled block
- [ ] 2.4 Implement dequant (Q4_K_M, Q8_0, F16) generic dot on the CPU bank
- [ ] 2.5 Failing test: SwiGLU + single-layer stack + logits over known constants
- [ ] 2.6 Implement layer stack: RMSNorm→QKV→RoPE→attn→out→residual→norm→SwiGLU→down→residual

## 3. Bit-exactness + cross-validation
- [ ] 3.1 Failing test: bank output vs expected reference (bit-exact synthetic logits)
- [ ] 3.2 Verify PASS (exit 0)
- [ ] 3.3 Failing test: bank layer-0 vs golden L0 cos-sim ≥ 0.9999 (fixture from 6.1)
- [ ] 3.4 Record numbers; verify PASS

## 4. Gate
- [ ] 4.1 Full suite green (CPU 6.1 + 6.2)
- [ ] 4.2 Gate sealed: bit-exact + L0 cos-sim recorded