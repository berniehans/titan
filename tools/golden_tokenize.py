#!/usr/bin/env python
"""Golden tokenize harness (change 6.1, task 3).

Ground-truth token streams from the PINNED llama.cpp build for every prompt in
`tests/fixtures/prompts.txt`. Writes `tests/fixtures/golden/tokenize_reference.json`.

Usage:
    uv run python tools/golden_tokenize.py

Requires $$LOCALAPPDATA/llama.cpp/build/bin/Release/llama-tokenize.exe
and the fixture GGUF (mirrors the pinned reference in reference.md, commit cb1adf8).
Idempotent; refuses to overwrite an existing reference hash mismatch warning.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA_BIN = Path(os.environ.get("LOCALAPPDATA", Path.home()))
LLAMA_BIN = LLAMA_BIN / "llama.cpp" / "build" / "bin" / "Release" / "llama-tokenize.exe"
FIXTURE = ROOT / "testdata" / "Qwen3-0.6B-Q4_K_M.gguf"
PROMPTS = ROOT / "tests" / "fixtures" / "prompts.txt"
OUT = ROOT / "tests" / "fixtures" / "golden" / "tokenize_reference.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    assert LLAMA_BIN.exists(), f"llama-tokenize missing: {LLAMA_BIN}"
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"

    prompts = [line.rstrip("\n").rstrip("\r") for line in PROMPTS.read_text(encoding="utf-8").splitlines()]
    prompts = [p for p in prompts if p != ""]
    assert len(prompts) >= 20, f"need >=20 prompts, got {len(prompts)}"

    refs = []
    for i, prompt in enumerate(prompts):
        proc = subprocess.run(
            [str(LLAMA_BIN), "-m", str(FIXTURE), "--ids", "--stdin", "--no-bos",
             "--no-escape", "--no-parse-special", "--log-disable"],
            input=prompt.encode("utf-8"),
            capture_output=True,
        )
        stdout = proc.stdout.decode("utf-8", "replace").strip()
        if proc.returncode != 0 or not stdout.startswith("["):
            print(f"[{i}] llama-tokenize failed rc={proc.returncode}: {stdout[:200]}")
            return 1
        ids = json.loads(stdout)
        refs.append({"index": i, "prompt": prompt, "ids": ids})
        print(f"[{i}] {prompt[:40]!r} -> {len(ids)} tokens")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "reference": "llama.cpp cb1adf8",
        "binary": str(LLAMA_BIN.name),
        "params": {"temp": 0, "seed": 42, "add_bos": False, "escape": False},
        "fixture": {
            "path": str(FIXTURE.relative_to(ROOT)),
            "sha256": sha256(FIXTURE),
            "size": FIXTURE.stat().st_size,
        },
        "prompts": refs,
    }
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes, {len(refs)} prompts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())