import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("compare_static_baseline.py")


def artifact(model="model-a", schema_version=1, repetitions=2, runs=None, llama=True, titan=True, config=None):
    if runs is None:
        runs = [{"repetition": i + 1} for i in range(repetitions)]
    statistics = {}
    for condition, llama_value, titan_value in (("cold", 100.0, 80.0), ("warm", 120.0, 90.0)):
        statistics[condition] = {}
        if llama:
            statistics[condition]["llama_decode_tok_s"] = {"median": llama_value, "samples": 2}
        if titan:
            statistics[condition]["titan_decode_tok_s"] = {"median": titan_value, "samples": 2}
    return {
        "schema_version": schema_version,
        "configuration": config or {"generated_tokens": 10, "temperature": 0.0, "repetitions": repetitions},
        "results": [{"model": model, "statistics": statistics, "repetitions": repetitions, "runs": runs}],
    }


class ComparatorTests(unittest.TestCase):
    def run_cli(self, baseline, current):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path, current_path, output_path = (root / name for name in ("base.json", "current.json", "out.json"))
            base_path.write_text(json.dumps(baseline), encoding="utf-8")
            current_path.write_text(json.dumps(current), encoding="utf-8")
            result = subprocess.run([sys.executable, str(SCRIPT), "--baseline", str(base_path), "--current-titan", str(current_path), "--output", str(output_path)], capture_output=True, text=True)
            output = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else None
            return result, output

    def test_valid_partial_comparison(self):
        result, output = self.run_cli(artifact("model-a"), artifact("model-a"))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(output["status"], "complete")
        comparison = output["comparisons"]["model-a"]["cold"]
        self.assertEqual(comparison["llama_median"], 100.0)
        self.assertEqual(comparison["titan_median"], 80.0)
        self.assertEqual(comparison["ratio"], 0.8)

    def test_missing_model_is_reported(self):
        baseline = artifact("model-a")
        baseline["results"].append(artifact("model-b")["results"][0])
        result, output = self.run_cli(baseline, artifact("model-a"))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(output["status"], "provisional_partial")
        self.assertEqual(output["missing_current_models"], ["model-b"])
        self.assertNotIn("model-b", output["comparisons"])

    def test_malformed_schema_and_config_fail(self):
        bad = artifact(schema_version=2, config={"generated_tokens": 9, "temperature": 0.0, "repetitions": 2})
        result, output = self.run_cli(bad, artifact())
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(output["incompatible_configuration_errors"])

    def test_repetition_mismatch_fails(self):
        result, output = self.run_cli(artifact(repetitions=2), artifact(repetitions=2, runs=[{"repetition": 1}]))
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(any("repetitions" in error for error in output["incompatible_configuration_errors"]))


if __name__ == "__main__":
    unittest.main()
