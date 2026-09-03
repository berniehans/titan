#!/usr/bin/env python3
"""Strict, stdlib-only acceptance gate for Titan benchmark artifacts."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONDITIONS = ("cold", "warm")
CONFIG_KEYS = ("generated_tokens", "temperature", "repetitions")
DEFAULT_MODELS = (
    "Qwen 2.5 1.5B Instruct", "Llama 3.2 1B Instruct", "Llama 3.2 3B Instruct",
    "DeepSeek-R1-Distill 1.5B", "Qwen3 0.6B Base/Chat",
)
BLOCKED_STATUSES = {"provisional_partial", "blocked", "diagnostic_only"}


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return value


def finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def results_by_model(data: dict[str, Any], label: str, failures: list[str]) -> dict[str, dict[str, Any]]:
    if data.get("schema_version") != SCHEMA_VERSION:
        failures.append(f"{label}_schema")
    if data.get("status") in BLOCKED_STATUSES:
        failures.append("artifact_status")
    results = data.get("results")
    if not isinstance(results, list):
        failures.append(f"{label}_results")
        return {}
    found: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict) or not isinstance(item.get("model"), str):
            failures.append(f"{label}_model_identity")
            continue
        model = item["model"]
        found[model] = item
        runs = item.get("runs")
        if not isinstance(runs, list) or item.get("repetitions") != len(runs):
            failures.append(f"{label}_repetitions")
    return found


def metric(item: dict[str, Any], condition: str, key: str, failures: list[str], model: str) -> float | None:
    try:
        value = item["statistics"][condition][key]["median"]
        samples = item["statistics"][condition][key]["samples"]
    except (KeyError, TypeError):
        failures.append(f"metric_{model}_{condition}")
        return None
    if not finite_positive(value) or not isinstance(samples, int) or samples <= 0:
        failures.append(f"metric_{model}_{condition}")
        return None
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Accept or reject a complete Titan benchmark checkpoint.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current-titan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--required-model", action="append", dest="required_models")
    parser.add_argument("--regression-baseline")
    args = parser.parse_args()
    required = args.required_models or list(DEFAULT_MODELS)
    failures: list[str] = []
    paths = {"baseline": args.baseline, "current_titan": args.current_titan}
    try:
        baseline = load_json(args.baseline)
        current = load_json(args.current_titan)
        regression = load_json(args.regression_baseline) if args.regression_baseline else None
        if regression is not None:
            paths["regression_baseline"] = args.regression_baseline
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        failures.append("input_artifact")
        baseline = current = {}
        regression = None
        paths["error"] = str(exc)
    base_items = results_by_model(baseline, "baseline", failures)
    current_items = results_by_model(current, "current", failures)
    for model in required:
        if model not in base_items or model not in current_items:
            failures.append("missing_model")
    for key in CONFIG_KEYS:
        if baseline.get("configuration", {}).get(key) != current.get("configuration", {}).get(key):
            failures.append(f"configuration_{key}")
    comparisons: dict[str, Any] = {}
    ratios: dict[str, list[float]] = {condition: [] for condition in CONDITIONS}
    for model in required:
        if model not in base_items or model not in current_items:
            continue
        comparisons[model] = {}
        for condition in CONDITIONS:
            llama = metric(base_items[model], condition, "llama_decode_tok_s", failures, model)
            titan = metric(current_items[model], condition, "titan_decode_tok_s", failures, model)
            if llama is None or titan is None:
                continue
            ratio = titan / llama
            ratios[condition].append(ratio)
            comparisons[model][condition] = {"llama_median": llama, "titan_median": titan, "ratio": ratio}
            if ratio < 0.95:
                failures.append("per_model_ratio")
    aggregate = {"cold_ratio": statistics.median(ratios["cold"]) if ratios["cold"] else None, "warm_ratio": statistics.median(ratios["warm"]) if ratios["warm"] else None, "overall_ratio": statistics.median(ratios["cold"] + ratios["warm"]) if ratios["cold"] + ratios["warm"] else None, "method": "median of available per-model cold/warm ratios; overall median across both conditions"}
    if aggregate["cold_ratio"] is None or aggregate["warm_ratio"] is None or aggregate["overall_ratio"] is None or aggregate["overall_ratio"] < 0.95:
        failures.append("aggregate_ratio")
    regression_compatible = (regression is not None and regression.get("schema_version") == SCHEMA_VERSION and regression.get("status") not in BLOCKED_STATUSES and regression.get("configuration") == current.get("configuration"))
    if regression_compatible:
        old = results_by_model(regression, "regression", failures)
        for model in required:
            if model not in old or model not in current_items:
                continue
            for condition in CONDITIONS:
                previous = metric(old[model], condition, "titan_decode_tok_s", failures, model)
                now = metric(current_items[model], condition, "titan_decode_tok_s", failures, model)
                if previous and now < previous * 0.95:
                    failures.append("regression")

    report = {"schema_version": SCHEMA_VERSION, "status": "accepted" if not failures else "rejected", "failed_gates": sorted(set(failures)), "required_models": required, "comparisons": comparisons, "aggregate": aggregate, "input_paths": paths}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
