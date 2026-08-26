You are implementing Task 1 (workspace scaffolding) of OpenSpec change "bootstrap-f0-f1" for a Rust LLM inference engine. The repo root is C:/Users/niber/AppData/Local/hermes/workspace/motor-llm-rust-cuda.

Create the following files EXACTLY. Do not build or run anything — just create files. Be precise; this is Cargo 1.96 ("new Cargo" with rust-toolchain conventions), not classic cargo.toml.

--- MANDATORY CONVENTIONS (from project constitution) ---
- Rust stable via engine/rust-toolchain.toml
- Cargo workspace root = engine/ (virtual manifest — NO [package] in it). Members = the 5 crates, each in its own subdir under engine/.
- Shared deps declared in [workspace.dependencies]; each crate inherits the ones it needs via its own package-level deps with workspace=true.
- `cargo clippy -- -D warnings` clean on every commit.
- Errors with thiserror in libs, anyhow in bins. NO unwrap() outside tests.
- unsafe ONLY in engine-cuda/engine-io, always with `// SAFETY:` comment.

=== FILE 1: engine/Cargo.toml (workspace root) ===
Use this EXACT validated structure. cudarc 0.12.2 is YANKED on crates.io, so pin cudarc 0.12.1 with the cuda-12000 feature:

```toml
[workspace]
members = ["engine-api", "engine-core", "engine-io", "engine-cuda", "engine-kvcache"]
resolver = "3"

[workspace.dependencies]
cudarc = { version = "0.12.1", default-features = false, features = ["cuda-12000"] }
tokio = "1.45.3"
axum = "0.10.1"
anyhow = "1.9.0"
thiserror = "1.2.5"
tracing = "0.11.0"
```
(Note: pick the [workspace] and [workspace.dependencies] tables. The exact dep versions should be sensible latest-stable for each crate; if a specific version I gave is unknown/yanked, use `cargo add` semantics and pick the latest 0.x stable. DO keep cudarc at 0.12.1 with default-features=false and features=["cuda-12000"] — that specific combination was validated to compile.)

## FILE 2: engine/rust-toolchain.toml ===
The constitution requires Rust stable. Create:

```toml
[toolchain]
channel = "stable"
```

## FILES 3-7: the 5 crates ===
For each crate create the directory and a Cargo.toml of the form:

```toml
[package]
name = "<name>"
version = "0.1.0"
edition = "2024"
```

Then add the src/lib.rs with a minimal public function (so the lib is non-empty and clippy-clean). E.g. for engine-api: `pub fn version() -> &'static str { "0.1.0" }`. Use an appropriate minimal public symbol per crate name. Do NOT add dependencies to crates yet (tasks 3-5 add them with TDD). But DO add the shared cudarc dependency to engine-cuda only, marked as workspace-inherited:

engine-cuda/Cargo.toml:
```toml
[package]
name = "engine-cuda"
version = "0.1.0"
edition = "2024"

[dependencies]
cudarc.workspace = true
```

For the other crates do NOT add [dependencies] yet.

The 5 crates: engine-api, engine-core, engine-io, engine-cuda, engine-kvcache. Each under engine/<name>/ with src/lib.rs.

## FILE 8: CI at .github/workflows/ci.yml ===
GitHub Actions, job "bootstrap", on push + pull_request. Steps: checkout, install Rust (use actions by posting stable via a toolchain install — but since this uses rust-toolchain.toml, install rust stable via the standard actions/setup-rust approach), then run in engine/:
- `cargo fmt --check`  (fmt linter check)
- `cargo clippy -- -D warnings`
- `cargo test -w`      (workspace-wide tests)
GPU tests: crates that need GPU mark their tests with #[ignore] so they do NOT run on CI without GPU. Do not run on GPU in CI.

Implement CI generically — if exact engine-cargo CLI commands are uncertain, use the canonical cargo 2.x equivalents: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test` (workspace operates on all members by default).

## FILE 9: .gitignore at repo root ===
Ignore: the engine `target/` build output, all. Windows/editor noise, PNG/GD etc. Standard Rust + Windows:
```
# OS
Thumbs.db
Desktop.ini
# Editors
.vscode/
.idea/
*.swp
# Rust / Cargo
engine/target/
# downloads / fixtures
testdata/*.gguf
testdata/.cache/
```
IMPORTANT: do NOT ignore openspec/ or the crates source.

## IMPORTANT IMPLEMENTATION NOTES
- Create directories as needed. Write files with correct content. Keep everything minimal and clean.
- Do NOT run cargo commands — the orchestrator verifies. Just create the files.
- Do NOT touch .agent/, openspec/, testdata/, tools/, or any existing files.
- GitHub Actions YAML must be valid (4-space indentation, no tabs).