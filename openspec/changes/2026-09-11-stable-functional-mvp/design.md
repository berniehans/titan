# Design: Stable Functional MVP

## Boundary

The MVP gate is a policy evaluator over independently produced evidence. It is not a replacement for `tools/release_gate.py`, and it must not accept by deleting strict ratio or regression failures from a strict report. It may reuse validated helper functions through explicit imports/interfaces.

## Inputs

Required inputs:

1. Clean benchmark JSON from the real producer.
2. External F8.R1.P correctness JSON plus expected SHA-256.
3. A correctness/build binding manifest proving that F8 and the benchmark exercise the same verified Titan build/source identity.
4. A machine-readable operational evidence manifest with raw-log paths and SHA-256 values.

Optional input:

- A compatible three-repetition regression baseline. Missing or incompatible regression evidence is advisory `not_evaluated`/`incomparable`, never an invented zero.

All paths are absolute at execution time. Every referenced file must exist, parse, and match its recorded digest.

## Required blocking gates

The gate blocks unless all of these pass:

- F8 status is `verified`, with exactly five allowlisted model identities and hashes.
- F8 route is F32, batch 1, resident KV, Graph=false, with selectors absent and production dispatch unchanged.
- F8 model/path/hash identities match the benchmark artifact.
- The correctness/build binding proves the exact Titan binary/source/build identity used by the benchmark; a dirty HEAD is not proof.
- The benchmark is schema-valid, complete, finite, and reconciled: 5 models × 3 repetitions × cold/warm × 41 generated tokens.
- Clean provenance is explicit: `instrumentation_mode=clean_throughput`, no recursive diagnostic telemetry, selectors absent, `candidate_authorization=false`, `production_dispatch_changed=false`, and observed Graph disabled with zero capture/replay.
- The operational manifest proves real execution, zero skipped required cases, semantic grammar assertions, readiness, bounded completion, and cleanup.
- Unknown failure classes are blocking.

## Advisory allowlist

Only the following are advisory: Titan/llama.cpp ratios, per-model ratio failures, compatible regression measurements, historical baseline incomparability, unstable reference samples, unavailable `ffn_sync`, asymmetric reference stage coverage, and blocked Nsight counters. Any missing evidence, schema error, provenance error, operational failure, telemetry contamination, selector violation, or build-binding failure remains blocking.

## Statuses and exit behavior

The output decision JSON uses only:

```text
mvp_accepted_functional
mvp_accepted_with_performance_advisory
mvp_blocked_correctness
mvp_blocked_operational
mvp_blocked_evidence
mvp_not_exercised
```

Accepted functional decisions always contain `promotion_authorized=false` and `rc_generated=false`. Blocked/not-exercised decisions return a nonzero process exit. Strict release status is copied only as an input reference and is never changed.

## Decision manifest

The output schema includes `schema_version`, `decision_id`, `status`, `release_approval`, promotion/RC flags, input path/hash records, per-gate verdicts, blocking failures, advisory values, exact model/build identities, operational case results, and reproduction metadata. The manifest is written to a new collision-safe path and never overwrites historical decisions.
