# Change: Stable Functional MVP

## Why

Titan has an independently verified F32 correctness matrix and a clean five-model throughput artifact, but the strict production gate rejects the current run on performance and regression thresholds. The project needs a separate functional-MVP decision that preserves those strict results while accepting only independently verified correctness, operational behavior, clean evidence, and safety invariants.

The existing `release_gate.py` and producer-provenance successors own strict release validation and producer schema alignment. This change owns only the functional-MVP policy boundary and its decision manifest; it must not duplicate or weaken those upstream contracts.

## What Changes

- Add a separate functional-MVP validator and tests under `tools/`.
- Bind the external F8.R1.P correctness artifact to the exact benchmark model identities, route, and Titan build/source identity.
- Validate a machine-readable operational evidence manifest for Resident JSON, Streaming SSE/N-gram, grammar-constrained JSON, readiness, bounded execution, cleanup, and negative lifecycle cases.
- Validate the clean benchmark artifact and preserve `correctness.status = "not_evaluated"`.
- Report ratio, regression, reference incomparability, causal-frontier limitations, and Nsight limitations as advisory/deferred dimensions.
- Emit a distinct MVP decision status with `promotion_authorized = false` and `rc_generated = false`.

## Non-goals

- Do not modify `tools/release_gate.py` strict threshold semantics.
- Do not modify CUDA, Rust runtime dispatch, kernels, defaults, or candidate selectors.
- Do not rewrite `docs/RELEASE_ENVELOPE_V1.json` or historical artifacts.
- Do not promote production, generate an RC, publish, commit, or push.
- Do not infer F8-to-build identity from a dirty-tree HEAD or from a status string.
- Do not treat synthetic validator fixtures as production evidence.

## Dependencies

- `2026-09-11-release-evidence-binding` owns strict external correctness binding and release validation.
- `2026-09-12-producer-provenance-alignment` owns benchmark producer provenance fields.
- `2026-09-10-f32-five-model-correctness` owns F8 correctness evidence.
- `2026-09-09-production-envelope` and `2026-09-09-grammar-json-conformance` own operational behavior evidence.

## Impact

Expected implementation scope is limited to `tools/mvp_gate.py`, `tools/test_mvp_gate.py`, and MVP evidence manifests under `local-artifacts/reviews/`. Existing release and benchmark artifacts remain immutable inputs. If build binding or operational normalization cannot be verified, the MVP result is blocked rather than accepted.
