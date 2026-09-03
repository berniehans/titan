import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("release_gate.py")
MODELS = [
    "Qwen 2.5 1.5B Instruct",
    "Llama 3.2 1B Instruct",
    "Llama 3.2 3B Instruct",
    "DeepSeek-R1-Distill 1.5B",
    "Qwen3 0.6B Base/Chat",
]


def artifact(titan_value=100.0, llama_value=100.0, models=None, status="complete", repetitions=2, runs=None, config=None):
    models = MODELS if models is None else models
    runs = [{"repetition": i + 1, "value": 1.0} for i in range(repetitions)] if runs is None else runs
    results = []
    for model in models:
        results.append({
            "model": model,
            "repetitions": repetitions,
            "runs": runs,
            "statistics": {
                "cold": {
                    "llama_decode_tok_s": {"median": llama_value, "samples": 2},
                    "titan_decode_tok_s": {"median": titan_value, "samples": 2},
                },
                "warm": {
                    "llama_decode_tok_s": {"median": llama_value, "samples": 2},
                    "titan_decode_tok_s": {"median": titan_value, "samples": 2},
                },
            },
        })
    return {
        "schema_version": 1,
        "status": status,
        "configuration": config or {"generated_tokens": 10, "temperature": 0.0, "repetitions": repetitions},
        "results": results,
    }


class ReleaseGateTests(unittest.TestCase):
    def run_cli(self, baseline, current, regression=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("baseline.json", "current.json", "out.json")]
            paths[0].write_text(json.dumps(baseline), encoding="utf-8")
            paths[1].write_text(json.dumps(current), encoding="utf-8")
            command = [sys.executable, str(SCRIPT), "--baseline", str(paths[0]), "--current-titan", str(paths[1]), "--output", str(paths[2])]
            if regression is not None:
                regression_path = root / "regression.json"
                regression_path.write_text(json.dumps(regression), encoding="utf-8")
                command += ["--regression-baseline", str(regression_path)]
            result = subprocess.run(command, capture_output=True, text=True)
            output = json.loads(paths[2].read_text(encoding="utf-8"))
            return result, output

    def test_accepts_complete_fixture(self):
        result, output = self.run_cli(artifact(), artifact())
        self.assertEqual(result.returncode, 0)
        self.assertEqual(output["status"], "accepted")
        self.assertEqual(output["aggregate"]["overall_ratio"], 1.0)

    def test_missing_model_rejected(self):
        result, output = self.run_cli(artifact(models=MODELS[:-1]), artifact())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing_model", output["failed_gates"])

    def test_provisional_status_rejected(self):
        result, output = self.run_cli(artifact(status="provisional_partial"), artifact())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("artifact_status", output["failed_gates"])

    def test_ratio_below_threshold_rejected(self):
        result, output = self.run_cli(artifact(titan_value=94.0), artifact(titan_value=94.0))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("per_model_ratio", output["failed_gates"])

    def test_aggregate_below_threshold_rejected(self):
        result, output = self.run_cli(artifact(titan_value=94.0), artifact(titan_value=94.0))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("aggregate_ratio", output["failed_gates"])

    def test_repetition_mismatch_rejected(self):
        bad = artifact(runs=[{"repetition": 1}], repetitions=2)
        result, output = self.run_cli(bad, artifact())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("repetitions", " ".join(output["failed_gates"]))

    def test_config_mismatch_rejected(self):
        result, output = self.run_cli(artifact(), artifact(config={"generated_tokens": 11, "temperature": 0.0, "repetitions": 2}))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("configuration", " ".join(output["failed_gates"]))

    def test_regression_above_five_percent_rejected(self):
        result, output = self.run_cli(artifact(), artifact(titan_value=94.0), artifact(titan_value=100.0))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("regression", " ".join(output["failed_gates"]))

    def test_invalid_nonfinite_metric_rejected(self):
        current = artifact()
        current["results"][0]["statistics"]["cold"]["titan_decode_tok_s"]["median"] = float("nan")
        result, output = self.run_cli(artifact(), current)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("metric", " ".join(output["failed_gates"]))


if __name__ == "__main__":
    unittest.main()
