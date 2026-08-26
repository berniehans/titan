# Delta Specification: Phase 12 — Speculative Decoding Engine (Draft Verification & Multi-Token Speculation)

## ADDED Requirements

### Requirement: Speculative Multi-Token Verification
The engine SHALL support validating $K \ge 2$ proposed candidate tokens simultaneously in a single batched target forward pass ($M = K$).

#### Scenario: Greedy speculative verification
- **WHEN** $K$ candidate tokens are proposed for positions $t \dots t+K-1$
- **THEN** target model SHALL compute logits for all candidate positions concurrently
- **AND** emit all consecutively matching candidate tokens plus one bonus token
- **AND** the emitted token sequence SHALL be identical to standard serial autoregressive decoding

### Requirement: Rejection Sampling & Distribution Equivalence
The speculative engine SHALL implement exact distribution-preserving rejection sampling for stochastic generation ($\text{temperature} > 0$).

#### Scenario: Probabilistic sampling equivalence
- **WHEN** generating text with speculative verification enabled at temperature $T > 0$
- **THEN** sample probabilities across accepted and bonus tokens SHALL match the target model's theoretical output distribution without bias

### Requirement: Context N-Gram Draft Proposer
The engine SHALL provide an ultra-low-latency context n-gram proposer (`NgramDraftProposer`) capable of proposing candidate tokens with $< 0.1\text{ ms}$ host overhead and 0 extra model parameters.

#### Scenario: Recurring phrase speculation
- **WHEN** current token history matches a previously generated or prompt n-gram
- **THEN** proposer SHALL output candidate continuation tokens of length $K \in [2, 5]$
