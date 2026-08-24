# Tasks: 6.3-gemv-multiformat

> Execute via bot coder. Strict TDD. One commit per task group. GPU tests: `#[ignore]` + `%LOCALAPPDATA%/Temp` PATH trick for NVRTC.

## 1. CPU harness + RED
- [ ] 1.1 Failing test: CPU dequant→dot (from 6.2 bank) over controlled Q4_K_M block == expected floats
- [ ] 1.2 Failing GPU test (`#[ignore]`): Q4_K_M block dot on device vs CPU dot
- [ ] 1.3 Failing GPU test (`#[ignore]`): Q8_0 block dot parity
- [ ] 1.4 Failing GPU test (`#[ignore]`): F16 (non-quant) dot parity

## 2. Kernel — GREEN (engine-cuda)
- [ ] 2.1 Implement `gemv_q4k.cu` with top-of-source port comment:
  `// Port of llama.cpp vcdotq.cuh::vec_dot_q4_K_q8_K (ggml/src/ggml-cuda/vecdotq.cuh @ cb1adf8)`
- [ ] 2.2 Add Q8_0 and F16 paths in the same translation unit (layout per ggml quant conventions)
- [ ] 2.3 Wrapper `MultiFormatGEMV` (RAII, NVRTC launch); emits only needed tile, no layer FP16 materialization
- [ ] 2.4 Verify PASS on local GPU (3/3 formats)

## 3. Real-tensor parity (teacher-forced)
- [ ] 3.1 Failing test (`#[ignore]`): real fixture embedding-row GEMV rel-L2 < 1e-3 vs CPU forward bank
- [ ] 3.2 Same for output/logit head (Q8_0 or F16 per realistic GGUF); rel-L2 < 1e-3
- [ ] 3.3 Failing test (`#[ignore]`): real first-block Q4 attention-weight matmul vs llama golden L0 (cos-sim ≥ 0.999, rel-L2 < 1e-3)
- [ ] 3.4 Record numbers; verify PASS

## 4. Gate
- [ ] 4.1 Full suite green (clippy -D warnings, CPU suite, GPU `--ignored`)
- [ ] 4.2 Gate sealed: rel-L2 < 1e-3 vs both references on all 3 formats