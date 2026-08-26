You are implementing Task 2 (fixture download script) of OpenSpec change "bootstrap-f0-f1". Repo root: C:/Users/niber/AppData/Local/hermes/workspace/motor-llm-rust-cuda.

Context: The GGUF fixture is Qwen3-0.6B-Q4_K_M, ~378MB, from the huggingface repo **unsloth/Qwen3-0.6B-GGUF** (file `Qwen3-0.6B-Q4_K_M.gguf`). This repo is verified to resolve correctly via curl -L. The file is already downloaded and verified at `testdata/Qwen3-0.6B-Q4_K_M.gguf` with:
- size: 396705472 bytes
- sha256: ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a
(source: https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf)

IMPORTANT repo ground truth: the resolver `huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf` returns HTTP 404 (broken xet object pointer, "Entry not found"); the `Qwen/Qwen3-0.6B-GGUF` official repo only has Q8_0, not Q4_K_M. So the ONLY working source is unsloth. The script MUST default to unsloth, with a clear comment noting the bartowski/official repos issue. Optionally support a mirror override via env var.

Create these files:

## FILE 1: tools/download_fixture.sh
A bash (git-bash POSIX compatible, runs on Win + Linux) download script with the following behavior:
- Target dir: testdata/ under repo root (resolve repo root from script location via `dirname`).
- Target file: testdata/Qwen3-0.6B-Q4_K_M.gguf
- URL (default): https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf  (overridable via env FIXTURE_URL)
- Expected SHA256 const: ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a
- Expected size: 396705472
- IDEMPOTENT: if the target file already exists AND its sha256 matches the expected constant, print a short "already present, checksum OK" message and exit 0 WITHOUT re-downloading. If it exists but checksum mismatches, re-download (replace).
- Uses curl (follow redirects with -L). On download, compute sha256 (via `sha256sum` on Linux/bash; on Windows git-bash sha256sum exists — if not found fall back to `python -c hashlib.sha256`).
- After successful download, update/write `testdata/CHECKSUMS.md` with the filename, size, and sha256 (journal entry, idempotent — overwrite the file fresh each run with the entry).
- Print clear PASS/FAIL. Non-zero exit on failure (curl error, size mismatch, or checksum mismatch).
- Must NOT require the huggingface_hub python lib (that's only for the manual bootstrap). Pure curl + optional sha256sum.
- bash strict mode: start with `#!/usr/bin/env bash` and `set -euo pipefail`.

## FILE 2: testdata/CHECKSUMS.md
Markdown with a small table or list: file | size (bytes) | sha256. Include exactly:
- Qwen3-0.6B-Q4_K_M.gguf  | 396705472 | ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a
Also add a short note that the official bartowski/Official-Qwen repos do NOT serve this file (404) and that unsloth is the pinned mirror source.

## NOTES
- Do NOT actually run the download (already done). Just create the two files.
- Keep the script clean and clippy-irrelevant (it's bash). Single quotes in the script where interpolation would break.
- Ensure the script works when invoked from any CWD (use repo root resolution).
- Do NOT modify testdata/Qwen3-0.6B-Q4_K_M.gguf (it is verified good).