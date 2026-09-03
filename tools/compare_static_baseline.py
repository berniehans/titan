#!/usr/bin/env python3
"""Compare frozen llama.cpp medians with current Titan medians."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONDITION_NAMES = ("cold", "warm")
CONFIG_KEYS = ("generated_tokens", "temperature", "repetitions")


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return value


def validate_artifact(data: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{label}: schema_version must be {SCHEMA_VERSION}")
    configuration = data.get("configuration")
    if not isinstance(configuration, dict):
        return errors + [f"{label}: configuration must be an object"]
    for key in CONFIG_KEYS:
        if key not in configuration:
            errors.append(f"{label}: missing configuration.{key}")
    results = data.get("results")
    if not isinstance(results, list):
        return errors + [f"{label}: results must be an array"]
    for index, result in enumerate(results):
        if not isinstance(result, dict) or not isinstance(result.get("model"), str):
            errors.append(f"{label}: result {index} missing model identity")
            continue
        repetitions = result.get("repetitions")
        runs = result.get("runs")
        if not isinstance(runs, list) or repetitions != len(runs):
            errors.append(f"{label}: {result['model']} repetitions must equal len(runs)")
    return errors


def compare(baseline: dict[str, Any], current: dict[str, Any], baseline_path: str, current_path: str) -> tuple[dict[str, Any], int]:
    errors = validate_artifact(baseline, "baseline") + validate_artifact(current, "current-titan")
    baseline_config = baseline.get("configuration", {})
    current_config = current.get("configuration", {})
    for key in CONFIG_KEYS:
        if key in baseline_config and key in current_config and baseline_config[key] != current_config[key]:
            errors.append(f"configuration mismatch: {key} baseline={baseline_config[key]!r} current={current_config[key]!r}")
    base_results = {item["model"]: item for item in baseline.get("results", []) if isinstance(item, dict) and isinstance(item.get("model"), str)}
    current_results = {item["model"]: item for item in current.get("results", []) if isinstance(item, dict) and isinstance(item.get("model"), str)}
    missing = sorted(set(base_results) - set(current_results))
    comparisons: dict[str, Any] = {}
    for model in sorted(set(base_results) & set(current_results)):
        base_stats = base_results[model].get("statistics", {})
        titan_stats = current_results[model].get("statistics", {})
        model_output: dict[str, Any] = {}
        for condition in CONDITION_NAMES:
            llama_metric = base_stats.get(condition, {}).get("llama_decode_tok_s", {})
            titan_metric = titan_stats.get(condition, {}).get("titan_decode_tok_s", {})
            if not isinstance(llama_metric, dict) or not isinstance(titan_metric, dict) or "median" not in llama_metric or "median" not in titan_metric:
                errors.append(f"missing median for {model}/{condition}")
                continue
            llama = llama_metric["median"]
            titan = titan_metric["median"]
            if not isinstance(llama, (int, float)) or not isinstance(titan, (int, float)) or llama == 0:
                errors.append(f"invalid median for {model}/{condition}")
                continue
            comparisons.setdefault(model, {})[condition] = {"llama_median": llama, "titan_median": titan, "ratio": titan / llama, "delta_percentage": (titan - llama) / llama * 100, "llama_samples": llama_metric.get("samples"), "titan_samples": titan_metric.get("samples")}
    output = {"schema_version": SCHEMA_VERSION, "status": "provisional_partial" if missing else "complete", "baseline_artifact": baseline_path, "current_titan_artifact": current_path, "configuration": {key: baseline_config.get(key) for key in CONFIG_KEYS}, "comparisons": comparisons, "missing_current_models": missing, "incompatible_configuration_errors": errors}
    return output, 1 if errors or not comparisons else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current-titan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        output, code = compare(load_json(args.baseline), load_json(args.current_titan), args.baseline, args.current_titan)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        output, code = {"schema_version": SCHEMA_VERSION, "status": "provisional_partial", "baseline_artifact": args.baseline, "current_titan_artifact": args.current_titan, "comparisons": {}, "missing_current_models": [], "incompatible_configuration_errors": [str(exc)]}, 1
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
