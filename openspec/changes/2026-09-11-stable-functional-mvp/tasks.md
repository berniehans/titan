# Tasks: Stable Functional MVP

> Scope is limited to the functional-MVP policy validator, its tests, and evidence manifests. Strict release validation, producer schema alignment, Rust/CUDA runtime, defaults, historical artifacts, commit, and push remain out of scope.

## 0. Dependency and contract freeze

- [x] 0.1 Reconcile live Git/OpenSpec counts and preserve the dirty-tree snapshot.
- [x] 0.2 Verify P1.R and P2.3 ownership and decisive test evidence.
- [x] 0.3 Freeze the required MVP input/output schemas and status vocabulary.
- [x] 0.4 Freeze the explicit advisory allowlist and unknown-failure fail-closed rule.
- [x] 0.5 Validate this change with `openspec validate --all` (`40 passed, 0 failed`).

## 1. TDD RED validator tests

- [x] 1.1 Valid F8/build binding/clean benchmark/operational manifest with low ratio yields accepted-with-advisory.
- [x] 1.2 Missing, corrupt, or SHA-mismatched F8 blocks correctness/evidence.
- [x] 1.3 F8 model, route, model-hash, or benchmark-build mismatch blocks evidence.
- [x] 1.4 Missing or unverifiable F8-to-Titan-build binding blocks evidence.
- [x] 1.5 Missing, skipped, failed, timed-out, or cleanup-failing operational case blocks operation.
- [x] 1.6 Malformed grammar output or schema mismatch blocks operation.
- [x] 1.7 Diagnostic instrumentation, recursive telemetry, missing selector fields, changed dispatch, or invalid Graph observation blocks evidence.
- [x] 1.8 Incomplete models, repetitions, conditions, tokens, prompt identities, raw-log hashes, or summary statistics block evidence.
- [x] 1.9 Ratio/regression/reference incomparability is advisory only.
- [x] 1.10 Unknown failure class blocks; only the explicit advisory allowlist is downgradeable.
- [x] 1.11 Every decision denies promotion and RC generation.

## 2. Minimal implementation

- [x] 2.1 Implement the CLI and deterministic decision schema in `tools/mvp_gate.py`.
- [x] 2.2 Load and hash-verify all required input files and linked raw logs.
- [x] 2.3 Validate F8-to-benchmark model/path/hash/route identity and correctness/build binding.
- [x] 2.4 Validate the real producer artifact through the shared P1.R adapter/helpers.
- [x] 2.5 Validate the operational manifest semantically, including grammar and lifecycle cases.
- [x] 2.6 Compute advisory ratios/regression without applying strict rejection thresholds.
- [x] 2.7 Emit fail-closed statuses and finite JSON with promotion/RC denial.

## 3. Verification

- [x] 3.1 Run focused MVP unit tests with a positive executed count (`25 passed`).
- [x] 3.2 Run the complete Python tools suite (`88 passed, 8 subtests passed` in the current live run; the earlier MVP execution record reported 84).
- [x] 3.3 Run `py_compile` and verify no strict validator regression.
- [x] 3.4 Inspect changed files and the exact diff from the master session.

## 4. Real evidence binding

- [x] 4.1 Create and verify a correctness/build binding manifest or record `mvp_blocked_evidence` — **RECORDED BLOCKED**: Binding manifest created at `local-artifacts/reviews/mvp-correctness-binding-20260911T162326Z.json`; F8 execution lacks direct Titan build/source identity binding.
- [x] 4.2 Create and verify a machine-readable operational evidence manifest or record `mvp_blocked_operational` — verified from existing real logs.
- [x] 4.3 Evaluate the exact P5 clean artifact with the exact F8 evidence and all input hashes — **EVALUATED (mvp_blocked_evidence)**: Evaluated via `tools/mvp_gate.py`; blocked fail-closed on unverified build binding and missing model hashes; decision persisted at `local-artifacts/reviews/mvp-decision-20260911T162326Z.json`.
- [x] 4.4 Confirm F3 diagnostic evidence is rejected by the clean MVP profile — `mvp_blocked_evidence`.
- [x] 4.5 Confirm the strict release envelope and historical artifacts remain unchanged.

## 5. Reconciliation and handoff

- [x] 5.1 Persist the final MVP decision and raw verification log (`mvp_blocked_evidence`).
- [x] 5.2 Update this change's research log and task states from machine-readable output.
- [x] 5.3 Create the local blocker bundle with reproduction commands and limitations.
- [x] 5.4 Run final applicable software gates and `openspec validate --all` — fmt/check/clippy/workspace tests exit 0; `openspec validate --all` reports `40 passed, 0 failed`.
