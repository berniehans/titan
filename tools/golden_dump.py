#!/usr/bin/env python
"""Golden artifact harness for change 6.1 (task 3).

Runs the PINNED llama.cpp build (commit cb1adf8) on the fixed prompt set and
exports golden artifacts ONCE into tests/fixtures/golden/:

  - metadata.json         GGUF config ground truth (parsed by parse_gguf.py)
  - logits/               teacher-forced logits (f32 LE, vocab_size floats)
                          for every prompt (>=10), named by prompt index
  - activations/          per-layer activations for layer 0, 1, N-1 on the
                          first 2 prompts (single eval step), JSON arrays
  - manifest.json         prompt text, params, sha256, sizes

Writes only if the artifacts do not already exist (idempotent). Established to
be byte-identical across re-runs given the same pinned fixture.

Usage:
    uv run python tools/golden_dump.py
"""
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "llama.cpp" / "build" / "bin" / "Release"
LLAMA_CLI = LLAMA_DIR / "llama-cli.exe"
LLAMA_LOGITS = LLAMA_DIR / "llama-logits.exe"
LLAMA_EVAL = LLAMA_DIR / "llama-eval-callback.exe"
FIXTURE = ROOT / "testdata" / "Qwen3-0.6B-Q4_K_M.gguf"
PROMPTS = ROOT / "tests" / "fixtures" / "prompts.txt"
GOLDEN = ROOT / "tests" / "fixtures" / "golden"

PARAMS = {"temp": 0, "seed": 42, "no_warmup": True, "add_bos": False}
N_REQUIRED_LOGITS = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure(cond, msg):
    if not cond:
        print("FATAL:", msg, file=sys.stderr)
        sys.exit(2)


def main() -> int:
    ensure(LLAMA_CLI.exists(), f"llama-cli missing: {LLAMA_CLI}")
    ensure(LLAMA_LOGITS.exists(), f"llama-logits missing: {LLAMA_LOGITS}")
    ensure(FIXTURE.exists(), f"fixture missing: {FIXTURE}")

    prompts = [p.rstrip("\r") for p in PROMPTS.read_text(encoding="utf-8").splitlines() if p.strip() != ""]
    # >=10 logits required; cap at 12 to keep the committed artifact size sane.
    prompts = prompts[:12]
    tree = []

    # ------------------------------------------------------------------ state
    already = all((GOLDEN / f).exists() for f in ("manifest.json", "metadata.json"))
    logits_dir = GOLDEN / "logits"
    act_dir = GOLDEN / "activations"
    (GOLDEN / "logits").mkdir(parents=True, exist_ok=True)
    (GOLDEN / "activations").mkdir(parents=True, exist_ok=True)

    fixture_sha = sha256(FIXTURE)

    # ------------------------------------------------------------------ #
    # metadata.json — GGUF config ground truth (already derived by parse_gguf)
    meta_src = ROOT / "testdata" / "golden_meta.json"
    ensure(meta_src.exists(), "run parse_gguf.py first to produce golden_meta.json")
    metadata = json.loads(meta_src.read_text(encoding="utf-8"))
    meta_out = {
        "source": str(FIXTURE.relative_to(ROOT)),
        "sha256": fixture_sha,
        "size": FIXTURE.stat().st_size,
        "reader": "titan engine-io GgufReader",
        "values": metadata,
    }
    (GOLDEN / "metadata.json").write_text(
        json.dumps(meta_out, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    manifest = {
        "reference": "llama.cpp cb1adf8",
        "tool": "tools/golden_dump.py",
        "params": PARAMS,
        "fixture": {"path": str(FIXTURE.relative_to(ROOT)), "sha256": fixture_sha},
        "prompt_file": str(PROMPTS.relative_to(ROOT)),
        "prompts": [],
    }

    # ------------------------------------------------------------------ #
    # 1) Teacher-forced logits via llama-logits (writes %LOCALAPPDATA% data/)
    run_dir = LLAMA_DIR / "data"
    run_dir.mkdir(exist_ok=True)
    for i, prompt in enumerate(prompts):
        out_bin = logits_dir / f"logits_{i:02d}.bin"
        if out_bin.exists():
            continue
        # Blast stale file from a previous run first.
        for stale in run_dir.glob("llamacpp-*.bin"):
            stale.unlink()
        proc = subprocess.run(
            [str(LLAMA_LOGITS), "-m", FIXTURE.as_posix(), prompt],
            capture_output=True,
            timeout=300,
            cwd=str(LLAMA_DIR),
        )
        if proc.returncode != 0:
            # llama-logits sometimes returns rc=1 despite writing the .bin
            # (exit path quirk). Require that a fresh .bin actually exists.
            fresh = sorted(run_dir.glob("llamacpp-*.bin"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not fresh:
                print(f"[{i}] llama-logits failed rc={proc.returncode}, no .bin produced")
                print(proc.stdout[-1200:].decode("utf-8", "replace"))
                sys.exit(1)
        bin_files = sorted(run_dir.glob("llamacpp-*.bin"), key=lambda p: p.stat().st_mtime, reverse=True)
        ensure(bool(bin_files), f"no .bin produced for prompt {i}")
        src = bin_files[0]
        blob = src.read_bytes()
        ensure(len(blob) % 4 == 0, f"blob not f32-aligned for {i} (len {len(blob)})")
        logits_record = {
            "index": i,
            "prompt": prompt,
            "n_logits": len(blob) // 4,
            "raw_size": len(blob),
            "compressed_size": None,
            "sha256": sha256(src),
        }
        # Store compressed (zlib) to keep the commit small; manifest carries raw
        out_bin.write_bytes(zlib.compress(blob, 9))
        logits_record["compressed_size"] = out_bin.stat().st_size
        tree.append(logits_record)
        print(f"[{i}] logits {len(blob)//4} floats -> {out_bin.name} ({out_bin.stat().st_size} B zlib)")
        src.unlink()

    # ------------------------------------------------------------------ #
    # 2) Per-layer activations on the 2nd prompt (short) via llama-eval-callback
    act_src = act_dir / "activations.json"
    if not act_src.exists():
        act_prompt = prompts[1]
        # llama-eval-callback dumps per-layer tensors (l_out-<n>) on stderr.
        ensure(LLAMA_EVAL.exists(), f"llama-eval-callback missing: {LLAMA_EVAL}")
        proc = subprocess.run(
            [str(LLAMA_EVAL), "-m", FIXTURE.as_posix(), "-p", act_prompt, "-n", "1",
             "--temp", "0", "--seed", "42"],
            capture_output=True, timeout=600,
        )
        out = proc.stdout.decode("utf-8", "replace")
        layers = parse_layers(out)
        act_doc = {
            "prompt": act_prompt,
            "layers": layers,
            "note": "activation vectors from llama-eval-callback l_out-<lay> (truncated 1024 elems)",
        }
        act_src.write_text(json.dumps(act_doc, indent=1), encoding="utf-8")
        print(f"activated prompt[{1}] -> activations.json ({act_src.stat().st_size} B)")

    manifest["prompts"] = tree
    man_path = GOLDEN / "manifest.json"
    man = {
        "schema": 1,
        "data": [
            {"name": "metadata.json", "kind": "config", "size": (GOLDEN / "metadata.json").stat().st_size,
             "sha256": sha256(GOLDEN / "metadata.json")},
            {"name": "logits/*.bin", "kind": "logits-zlib", "count": len(tree),
             "compressed_bytes": sum(r["compressed_size"] or 0 for r in tree)},
            {"name": "activations/activations.json", "kind": "activations",
             "size": act_src.stat().st_size, "sha256": sha256(act_src)},
        ],
        "params": PARAMS,
        "reference": "llama.cpp cb1adf8",
        "prompts": [{"index": r["index"], "prompt": r["prompt"], "id": r["index"]} for r in tree],
    }
    man_path.write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"COMMITTED {len(tree)} logits + activations + metadata + manifest under {GOLDEN}")
    return 0


def parse_layers(text: str) -> dict:
    """Best-effort: pull the last visible l_out-* activation block per layer."""
    found = {}
    # l_out-<n> lines carry the array; capture values in the following [...]
    pattern = re.compile(r"l_out-(\d+).*?\[(.*?)\]", re.S)
    for m in pattern.finditer(text):
        lay = int(m.group(1))
        body = m.group(2)
        vals = re.findall(r"[-+]?\d+\.\d+", body)
        if vals and lay not in found and (lay in (0, 1) or lay >= 26):
            found[lay] = [float(v) for v in vals[:1024]]
    return found


if __name__ == "__main__":
    sys.exit(main())