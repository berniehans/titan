from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).with_name("checkpoint_matrix.py")


def artifact(model="Qwen3", path="models/q4.gguf", *, titan=100.0, llama=None, config=None, metadata=None):
    cfg = {"generated_tokens": 41, "temperature": 0.0, "repetitions": 3}
    cfg.update(config or {})
    result = {
        "model": model, "model_path": path, "repetitions": 3,
        "statistics": {"cold": {"titan_decode_tok_s": {"samples": 3, "median": titan, "stddev": 2}},
                        "warm": {"titan_decode_tok_s": {"samples": 3, "median": titan + 1, "stddev": 2}}},
        "runs": [{"prompts": [{"cache_condition": "cold", "titan": {"generated_tokens": 41}},
                                 {"cache_condition": "warm", "titan": {"generated_tokens": 41}}]}] * 3,
    }
    if llama is not None:
        for condition, value in (("cold", llama), ("warm", llama + 1)):
            result["statistics"][condition]["llama_decode_tok_s"] = {"samples": 3, "median": value, "stddev": 2}
    data = {"schema_version": 1, "configuration": cfg, "results": [result]}
    if metadata:
        data.update(metadata)
    return data


def run(tmp_path, *files):
    output = tmp_path / "matrix.json"
    proc = subprocess.run([sys.executable, str(TOOL), *sum((["--artifact", str(f)] for f in files), []), "--output", str(output)], capture_output=True, text=True)
    return proc, json.loads(output.read_text()) if output.exists() else None


def test_valid_mixed_artifacts_and_classification(tmp_path):
    historical = tmp_path / "historical.json"; historical.write_text(json.dumps(artifact(titan=90, llama=100)))
    q8 = tmp_path / "current-q8.json"; q8.write_text(json.dumps(artifact(titan=110, metadata={"path": "q8", "git": {"commit": "abc"}})))
    f32 = tmp_path / "current-f32.json"; f32.write_text(json.dumps(artifact(titan=80, path="testdata/Qwen3-f32.gguf", metadata={"path": "fp32"})))
    proc, report = run(tmp_path, historical, q8, f32)
    assert proc.returncode == 0
    assert {row["path_classification"] for row in report["rows"]} >= {"historical Q8", "current Q8", "current FP32"}
    assert all("source_artifact" in row and "comparability_reason" in row for row in report["rows"])


def test_incompatible_config_and_missing_identity_are_not_comparable(tmp_path):
    a = tmp_path / "a.json"; a.write_text(json.dumps(artifact()))
    b = tmp_path / "b.json"; b.write_text(json.dumps(artifact(config={"temperature": 0.7}, metadata={"model_identity": None})))
    proc, report = run(tmp_path, a, b)
    assert proc.returncode == 0
    assert any(row["ratio"] is None for row in report["rows"])
    assert any("temperature" in row["comparability_reason"] or "identity" in row["comparability_reason"] for row in report["rows"])


def test_malformed_artifact_is_reported_nonzero(tmp_path):
    bad = tmp_path / "bad.json"; bad.write_text("{")
    proc, report = run(tmp_path, bad)
    assert proc.returncode != 0
    assert report["status"] == "error"
    assert report["artifacts"][0]["error"]


def test_regression_suppressed_when_incompatible(tmp_path):
    old = tmp_path / "old.json"; old.write_text(json.dumps(artifact(titan=100, metadata={"build": "old"})))
    new = tmp_path / "new.json"; new.write_text(json.dumps(artifact(titan=80, metadata={"build": "new"})))
    proc, report = run(tmp_path, old, new)
    assert proc.returncode == 0
    assert report["compatibility_summary"]["regression"] is None
    assert "build" in report["compatibility_summary"]["regression_reason"]


if __name__ == "__main__":
    import tempfile
    tests = [test_valid_mixed_artifacts_and_classification,
             test_incompatible_config_and_missing_identity_are_not_comparable,
             test_malformed_artifact_is_reported_nonzero,
             test_regression_suppressed_when_incompatible]
    for test in tests:
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory))
    print(f"{len(tests)} passed, 0 failed")
