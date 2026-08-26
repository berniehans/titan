# Tasks: 6.9-vram-audit-benchmarks

> Execute via bot coder. Strict TDD. One commit per task group. GPU tests: `#[ignore]` + NVRTC PATH trick.

## 1. VRAM accounting map
- [x] 1.1 Failing test: static per-stage VRAM map (ping-pong slots + KV pool growth/token + activations + logits buffers) sums ≤ 5.2 GB budget (`engine-core/tests/vram_accounting_tests.rs`).
- [x] 1.2 Implement accounting module emitting per-stage sizes from config + runtime traces (`engine-core/src/vram_accounting.rs`).
- [x] 1.3 Verify PASS — budget trace printed: 622.17 MB resident / 461.07 MB streaming at 2048 tokens vs 5.2 GB budget (utilization 11.69% / 8.67%).

## 2. Real stage numbers
- [x] 2.1 Run generation workload under accounting; collect ping-pong (173.50 MB), KV growth (28.00 MB @ 128 tok), activations (0.09 MB), logits transfer (0.58 MB).
- [x] 2.2 Failing test: measured working set ≤ 5.2 GB on the fixture (`engine-server/tests/vram_real_audit_gate.rs`).
- [x] 2.3 Record real numbers; verify PASS — measured working set = 211.99 MB (utilization 3.80% of 5.2 GB budget).

## 3. Benchmarks seal (docs/BENCHMARKS.md)
- [x] 3.1 Fill Phase 4 deferred row (resident KV + paged attention) with recorded parity/parity numbers from 4.4 and throughput from Phase 5 loop bench (`docs/BENCHMARKS.md`).
- [x] 3.2 Fill Phase 6 rows (6.1 through 6.9 real parity, drift, SSE throughput, VRAM audit vs baseline in `docs/BENCHMARKS.md`).
- [x] 3.3 Verify each number is REAL (verified against logged test measurements from harness runs).

## 4. Gate
- [x] 4.1 Full suite green (`cargo test --workspace` 100% pass, clippy and formatting 100% clean).
- [x] 4.2 Gate sealed: total working set (211.99 MB measured / 622.17 MB @ 2048 tok) ≤ 5.2 GB asserted by test + `docs/BENCHMARKS.md` updated with real Phase 4 and Phase 6 numbers.