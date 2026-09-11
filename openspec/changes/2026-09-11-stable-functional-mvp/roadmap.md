# Roadmap: Stable Functional MVP

## Guardrails

- Implement code only through `coder`/Agy using TDD RED → GREEN.
- Preserve the dirty tree and all historical artifacts; no reset, restore, clean, deletion, staging, commit, or push.
- Do not modify strict release thresholds, Rust/CUDA runtime code, defaults, or the production envelope.
- Execute one gate at a time and stop dependent work at the first failed or blocked prerequisite.
- Use native `C:/...` paths, fresh run IDs, separate raw logs, and SHA-256 manifests for every long run.

## Stages

### Stage 0 — Dependency and evidence freeze

Reconcile P1.R, P2.3, F8, production-envelope, and grammar contracts against live Git/OpenSpec output. Verify exact input files and hashes. Record the known absence of F8-to-P5 Titan build binding as a blocking evidence question, not as an assumption.

Exit: OpenSpec validates; ownership has no overlap; dirty-tree and evidence snapshot are persisted.

### Stage 1 — Contract tests (RED)

Through `coder`/Agy, add synthetic tests for valid functional acceptance with low performance, every required blocking failure, unknown-failure fail-closed handling, semantic operational cases, advisory-only ratios/regression, mandatory safety flags, and promotion/RC denial.

Exit: tests fail only because MVP implementation is absent, not because of a broken harness.

### Stage 2 — Minimal MVP validator (GREEN)

Implement `tools/mvp_gate.py` and `tools/test_mvp_gate.py` within the explicit boundary in `design.md`. Reuse P1.R validation primitives where stable; do not parse a strict report and remove failures. Validate hashes, identities, route, clean benchmark structure, operational manifest, advisory allowlist, and decision schema.

Exit: Stage 1 tests pass and the strict release validator's semantics are unchanged.

### Stage 3 — Evidence manifests

Create a correctness/build binding manifest only from independently verifiable source/build/run facts. Create a normalized operational manifest from real raw logs and machine-readable evidence. If either cannot be established, classify the corresponding gate blocked and do not fabricate a verified manifest.

Exit: both manifests are either verified or the decision is explicitly blocked/not exercised.

### Stage 4 — Real integration evaluation

Run the MVP validator against the P5 clean artifact and F8 evidence. Run negative checks against F3 diagnostic evidence and synthetic malformed fixtures. Use the aligned P2 artifact only for optional advisory regression. A fresh GPU benchmark is conditional and authorized only if existing evidence cannot satisfy identity or operational coverage.

Exit: a machine-readable MVP decision exists; strict release status is unchanged.

### Stage 5 — Documentation and bundle

Only after Stage 4, update this change's research/tasks from real output and create a local bundle with decision, input hashes, reproduction commands, limitations, rollback to F32, and performance advisory. If blocked, create a blocker bundle rather than an accepted-looking MVP bundle.
