# titan-cli Delta Specification

## Purpose
Specifies release documentation and reproducible benchmarking suite.

## Requirements

### Requirement: Comprehensive README & Quickstart Guide
The repository SHALL provide a root `README.md` containing installation steps, CLI usage examples, architecture overview, and verified performance benchmark comparisons.

#### Scenario: Developer builds and runs Titan from source
- **WHEN** developer runs `cargo build --release` and `titan chat -m <model.gguf>`
- **THEN** engine starts up with full diagnostics and real-time token telemetry.
