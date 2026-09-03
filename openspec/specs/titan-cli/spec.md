# titan-cli Delta Specification

## Purpose
Provides unified command-line binary `titan` with interactive chat, benchmarking, and Hermes Agent auto-detection.

## Requirements
### Requirement: Subcommands
The CLI SHALL implement `serve`, `chat`, `bench`, and `agent` subcommands.

#### Scenario: Recognized top-level subcommands
- **WHEN** the user invokes `titan --help`
- **THEN** the help output SHALL list `serve`, `chat`, `bench`, and `agent`
- **AND** invoking each listed subcommand SHALL be routed to its corresponding execution path
