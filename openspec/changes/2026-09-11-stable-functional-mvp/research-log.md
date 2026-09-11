# Research Log: Stable Functional MVP

## Initial live evidence

- Live checkout is intentionally dirty at `e15fa907f0ccad0aee75b4ac85d0ec08411fb675`.
- The live repository has 23 active OpenSpec changes and `openspec validate --all` currently passes 39 items.
- The full Python tools suite currently passes 59 tests; the strict release-gate subset passes 55 tests.
- P1.R and P2.3 are active predecessors with implementation evidence in the dirty tree but open task checklists. Their scopes must be reconciled before this change's implementation.
- F8.R1.P is independently verified for five models under F32, batch 1, resident KV, Graph=false, with selectors absent and production dispatch unchanged.
- The P5 clean benchmark is complete and cleanly instrumented, while the strict release gate is rejected on performance/regression gates.

## Evidence limitations

- The F8 matrix and external evidence manifest do not contain a Titan build/source identity matching the P5 benchmark's Titan build hash. This is a blocking evidence gap for MVP acceptance until independently proven.
- Existing operational evidence verifies Resident/Streaming/N-gram and grammar behavior in separate artifacts/logs, but no consolidated machine-readable MVP operational manifest exists yet.
- The F3 artifact uses diagnostic attribution and is not eligible as clean throughput evidence.

## Decisions

1. Keep throughput `correctness.status = "not_evaluated"`; bind correctness externally.
2. Keep `tools/release_gate.py` strict and unchanged in threshold semantics.
3. Treat ratio, regression, reference instability, `ffn_sync`, and Nsight limitations as advisory/deferred only after all functional evidence passes.
4. Treat missing build binding or operational normalization as a blocker, not as acceptance.
5. Do not run a fresh GPU benchmark unless existing evidence cannot satisfy a required identity or operational case.

## Execution record — 2026-09-11T162326Z

- M0 preflight: verified. Artifact:
  `local-artifacts/reviews/mvp-preflight-20260911T162326Z.json`.
  The live snapshot recorded 23 tracked status entries, 54 untracked status
  entries, 199 recursively untracked files, 23 active OpenSpec changes, and
  359/516 mechanical tasks checked. All nine preflight commands exited 0.
- P1.R verification: focused 55 tests and full 59-test suite passed from the
  master session. Raw log:
  `local-artifacts/reviews/m1-p1r-validator-20260911T162326Z.raw.log`.
- P2.3 verification: fmt, check, clippy, and producer target passed; the target
  reported 33 passed, 0 failed, 1 ignored benchmark. Raw log:
  `local-artifacts/reviews/m1-p23-producer-20260911T162326Z.raw.log`.
- MVP OpenSpec successor: `openspec validate --all` returned `40 passed, 0
  failed (40 items)` after creation of this change.
- MVP implementation: `tools/mvp_gate.py` and `tools/test_mvp_gate.py` were
  created through Agy/Coder. The focused suite passed 25 tests; the full tools
  suite passed 84 tests; `py_compile` passed. Master verification log:
  `local-artifacts/reviews/m2-master-hardening-verification-20260911T162326Z.raw.log`.
- Operational normalization: verified manifest created at
  `local-artifacts/reviews/mvp-operational-evidence-20260911T162326Z.json`,
  covering seven required cases and linking real raw logs by SHA-256.
- Correctness/build binding: deliberately `blocked`, not `verified`, at
  `local-artifacts/reviews/mvp-correctness-binding-20260911T162326Z.json`.
  The F8 raw/runner evidence does not identify the Titan binary/source build;
  the P5 freeze identifies its own benchmark binary but does not prove F8 ran
  with that binary.
- Real P5 evaluation: `mvp_blocked_evidence`, exit 1. Decision:
  `local-artifacts/reviews/mvp-decision-20260911T162326Z.json`. Blocking
  failures are `build_binding_unverified` and `model_hash_missing`; the
  operation gate passed and ratio/regression remained advisory.
- F3 negative evaluation: `mvp_blocked_evidence`, with clean-mode and
  diagnostic-instrumentation failures. Decision:
  `local-artifacts/reviews/mvp-f3-negative-20260911T162326Z.json`.
- Synthetic missing-selector negative: `mvp_blocked_evidence`, with
  `candidate_isolation_violation`. Decision:
  `local-artifacts/reviews/mvp-negative-missing-selector-decision-20260911T162326Z.json`.
- Blocker bundle:
  `local-artifacts/bundles/stable-functional-mvp-blocked-20260911T162326Z/`.

## Final status

The functional-MVP implementation and fail-closed tests are verified, but this
iteration does **not** produce an accepted MVP. The final decision is
`mvp_blocked_evidence`. No fresh GPU rerun was justified: the missing
F8-to-build association and missing per-result model hashes are evidence/schema
gaps, and the existing historical artifacts must not be rewritten. The strict
release envelope remains `not_accepted`, with promotion and RC generation
denied.

## Final software-gate reconciliation

- Final `cargo fmt --manifest-path engine/Cargo.toml --all -- --check`: exit 0.
- Final `cargo check --manifest-path engine/Cargo.toml --workspace`: exit 0.
- Final `cargo clippy --manifest-path engine/Cargo.toml --workspace --all-targets -- -D warnings`: exit 0.
- An initial `cargo test --manifest-path engine/Cargo.toml --workspace` returned
  exit 101 because the existing source-contract test scanned synthetic fixtures
  and produced a false positive. The test-only correction was delegated to Agy
  and limited to `engine/engine-server/tests/full_forward_causal_bound.rs`; the
  producer/runtime was not changed. The exact RED and GREEN evidence is in
  `local-artifacts/reviews/causal-test-false-positive-fix-20260911T162326Z.raw.log`.
- Final `cargo test --manifest-path engine/Cargo.toml --workspace`: exit 0 after
  that correction. Final cargo log:
  `local-artifacts/reviews/m5-final-cargo-gates-after-test-fix-20260911T162326Z.raw.log`.
- Final Python tooling suite: 84 tests passed. Final OpenSpec validation:
  `40 passed, 0 failed (40 items)`.

The MVP implementation remains verified within its authorized scope and the
repository-wide software gate is green after correcting the false-positive test
contract. The final MVP decision remains blocked by evidence: the F8 execution
does not directly identify the P5 Titan build, and the historical P5 artifact
lacks per-result model hashes. The next executable gate requires an explicitly
authorized producer/evidence successor or a fresh paired F8/P5 execution that
records those identities; do not rewrite historical artifacts or accept based
on the current blocked binding.

## Documentation reconciliation — 2026-09-11

All repository-facing documentation (`README.md`, `docs/MVP.md`, `docs/ARCHITECTURE.md`, `docs/BENCHMARKS.md`, `docs/TESTING.md`, `docs/WORKSPACE_STATE.md`, `docs/TITAN_PROJECT_DIAGNOSIS_AND_ROADMAP.md`) has been updated and reconciled:
- Prominent status block positioned near top of all relevant docs stating functional MVP is `mvp_blocked_evidence` (NOT accepted) and strict release is `not_accepted` / `rejected`.
- Clear distinction that 0.95x ratio vs llama.cpp is an advisory diagnostic for functional MVP, not an acceptance gate.
- Full details on exact blockers (`build_binding_unverified` and `model_hash_missing`).
- Verified facts reconciled against the current live run: 25 MVP unit tests, 88 full-suite tests plus 8 subtests, 40/40 OpenSpec, operational E2E modes verified, grammar JSON `{city: Tokyo}`, 30 clean benchmark rows, clean telemetry, and strict gate aggregate ratio 0.5286115177544821. The earlier MVP execution record reported 84 full-suite tests; that historical count is retained in the execution record above.
- Clear demarcation of local-only assets (`local-artifacts/`, `.hermes/`, `models/`, `target/`) as non-publishable repository content.
