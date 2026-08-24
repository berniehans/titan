# Tasks: 6.9-vram-audit-benchmarks

> Execute via bot coder. Strict TDD. One commit per task group. GPU tests: `#[ignore]` + NVRTC PATH trick.

## 1. VRAM accounting map
- [ ] 1.1 Failing test: static per-stage VRAM map (ping-pong slots + KV pool growth/token + activations + logits buffers) sums ≤ 5.2 GB budget
- [ ] 1.2 Implement accounting module emitting per-stage sizes from config + runtime traces
- [ ] 1.3 Verify PASS (budget trace printed: per stage and aggregate)

## 2. Real stage numbers
- [ ] 2.1 Run generation workload under accounting; collect ping-pong, KV growth/token, activation cliffs, logits transfer bytes
- [ ] 2.2 Failing test: measured working set ≤ 5.2 GB on the fixture
- [ ] 2.3 Record real numbers; verify PASS

## 3. Benchmarks seal (docs/BENCHMARKS.md)
- [ ] 3.1 Fill Phase 4 deferred row (resident KV + paged attention) with recorded parity/parity numbers from 4.4
- [ ] 3.2 Fill Phase 6 rows (6.3-6.8 real parity, drift, throughput vs baseline)
- [ ] 3.3 Verify each number is REAL (from a logged measurement, not a placeholder)

## 4. Gate
- [ ] 4.1 Full suite green
- [ ] 4.2 Gate sealed: total ≤ 5.2 GB asserted by test + BENCHMARKS updated with real Phase 4/6 numbers