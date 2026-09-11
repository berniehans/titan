# Specification: Stable Functional MVP

## Purpose

Define a separate, fail-closed functional-MVP decision for Titan that requires independent correctness, operational execution, clean evidence, and safety provenance without weakening strict release/performance acceptance.

## ADDED Requirements

### Requirement: Independent correctness and build binding

The MVP validator SHALL load external F8.R1.P evidence, verify its SHA-256 digest and exact five-model F32 route, and SHALL verify that the evidence is bound to the exact Titan build/source identity used by the evaluated benchmark.

#### Scenario: Fully bound F8 and benchmark
- **WHEN** F8 is `verified`, its five model/path/hash identities and route match the benchmark
- **AND** the correctness/build binding proves the exact Titan build/source identity
- **THEN** the correctness and evidence gates SHALL pass.

#### Scenario: Missing build binding
- **WHEN** F8 correctness is valid but no independently verifiable binding to the benchmark Titan build exists
- **THEN** the MVP status SHALL be `mvp_blocked_evidence`
- **AND** the validator SHALL NOT infer identity from a dirty HEAD or a status string.

### Requirement: Clean benchmark integrity

The MVP validator SHALL require a complete real-producer artifact with five models, three repetitions, cold and warm conditions, exactly 41 generated tokens, finite metrics, reconciled summaries, clean instrumentation, explicit selector provenance, unchanged production dispatch, and observed Graph disabled with zero capture/replay.

#### Scenario: Low-throughput clean artifact
- **WHEN** all functional benchmark invariants pass
- **AND** Titan/llama.cpp ratio or regression thresholds fail
- **THEN** those failures SHALL be advisory and SHALL NOT alone block functional MVP acceptance.

#### Scenario: Diagnostic or incomplete artifact
- **WHEN** diagnostic telemetry, missing provenance, malformed samples, invalid Graph state, or incomplete structure is present
- **THEN** the MVP validator SHALL block and SHALL NOT downgrade the failure to advisory.

### Requirement: Operational evidence manifest

The MVP validator SHALL require a machine-readable, hash-linked manifest proving real execution of Resident JSON, Streaming SSE/N-gram, grammar-constrained JSON, readiness, bounded completion, cleanup, and required negative lifecycle cases.

#### Scenario: Semantically verified operations
- **WHEN** each required case has an executed command, exit code, fixture identity, raw-log path/hash, zero skipped cases, and its declared semantic assertions pass
- **THEN** the operational gate SHALL pass.

#### Scenario: Missing or skipped operation
- **WHEN** a required case is missing, skipped, timed out, truncated, or fails cleanup
- **THEN** the MVP status SHALL be `mvp_blocked_operational`
- **AND** a raw log's mere existence SHALL NOT count as execution.

### Requirement: Semantic grammar validation

The MVP validator SHALL require complete grammar state, successful JSON parsing, exactly one `city` key, and the value `Tokyo` for the declared grammar case.

#### Scenario: Valid constrained JSON
- **WHEN** the grammar case reaches the explicit complete state
- **AND** serde parsing succeeds with exactly `{ "city": "Tokyo" }`
- **THEN** the grammar case SHALL pass.

#### Scenario: Incomplete or malformed JSON
- **WHEN** grammar completion or semantic JSON assertions fail
- **THEN** the operational gate SHALL block.

### Requirement: Explicit advisory allowlist

The MVP validator SHALL downgrade only ratio, regression, reference-comparability, unavailable causal-frontier, and unavailable Nsight findings to advisory/deferred fields. Unknown or missing failure classes SHALL block.

#### Scenario: Strict release rejection only on advisory fields
- **WHEN** the strict release gate rejects only absolute ratio or regression gates
- **AND** all functional MVP gates pass
- **THEN** the MVP status SHALL be `mvp_accepted_with_performance_advisory`.

#### Scenario: Unknown failure
- **WHEN** a failure is not in the explicit advisory allowlist
- **THEN** the MVP validator SHALL block closed and SHALL NOT accept.

### Requirement: Non-promotion decision

Every MVP decision SHALL be distinct from strict release acceptance and SHALL deny promotion and RC generation.

#### Scenario: Functional MVP acceptance
- **WHEN** all required functional gates pass
- **THEN** the decision SHALL use an allowed MVP accepted status
- **AND** `promotion_authorized` SHALL be `false`
- **AND** `rc_generated` SHALL be `false`.
