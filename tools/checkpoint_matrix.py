#!/usr/bin/env python3
"""Build a conservative compatibility matrix from benchmark JSON artifacts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONFIG_KEYS = ("generated_tokens", "temperature", "repetitions", "cache_condition", "build")

def load(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("top-level JSON must be an object")
    return value

def classify(path: str, data: dict[str, Any], result: dict[str, Any]) -> str:
    text = (path + " " + json.dumps(data.get("metadata", {}))).lower()
    if "static" in text: return "static llama.cpp"
    if "fresh" in text or "llama.cpp" in text: return "fresh llama.cpp"
    if "f32" in text or "fp32" in text or "float32" in text: return "current FP32"
    if "rerun" in text or "historical" in text: return "historical Q8"
    return "current Q8" if "q8" in text or "q4" in str(result.get("model_path", "")).lower() else "current Q8"

def identity(data: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    cfg = data.get("configuration") if isinstance(data.get("configuration"), dict) else {}
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    model = meta.get("model_identity", result.get("model"))
    return {"model": model, "model_path": result.get("model_path"),
            "generated_tokens": cfg.get("generated_tokens"), "temperature": cfg.get("temperature"),
            "repetitions": result.get("repetitions", cfg.get("repetitions")),
            "build": meta.get("build", data.get("build", data.get("git"))),
            "cache_condition": None}

def metric(result: dict[str, Any], condition: str, engine: str) -> tuple[Any, int | None, dict[str, Any]]:
    stats = result.get("statistics", {}).get(condition, {})
    name = f"{engine}_decode_tok_s"
    item = stats.get(name, {}) if isinstance(stats, dict) else {}
    if not isinstance(item, dict): item = {}
    return item.get("median"), item.get("samples"), item

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "status": "diagnostic", "artifacts": [], "rows": [], "compatibility_summary": {}, "missing_fields": []}
    loaded: list[tuple[str, dict[str, Any]]] = []
    failed = False
    for raw in args.artifact:
        entry = {"source_artifact": raw}
        try:
            data = load(raw)
            if not isinstance(data.get("results"), list): raise ValueError("results must be an array")
            entry["status"] = "parsed"; loaded.append((raw, data))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            entry["status"] = "error"; entry["error"] = f"{raw}: {exc}"; failed = True
        report["artifacts"].append(entry)
    for path, data in loaded:
        for result in data["results"]:
            if not isinstance(result, dict):
                report["missing_fields"].append(f"{path}: result is not an object"); continue
            ident = identity(data, result); cls = classify(path, data, result)
            for condition in ("cold", "warm"):
                titan, tsamples, tstats = metric(result, condition, "titan")
                llama, lsamples, _ = metric(result, condition, "llama")
                reason = []
                for key in ("model", "model_path", "generated_tokens", "temperature", "repetitions", "build"):
                    if ident.get(key) is None: reason.append(f"missing {key}"); report["missing_fields"].append(f"{path}: {key}")
                row = {"source_artifact": path, "model": result.get("model"), "condition": condition,
                       "titan_metric": titan, "llama_metric": llama, "ratio": None,
                       "repetitions": ident.get("repetitions"), "serialized_sample_count": tsamples,
                       "llama_serialized_sample_count": lsamples, "path_classification": cls,
                       "graph_path_metadata": {k: data[k] for k in ("graph", "path", "dispatch") if k in data},
                       "statistics": {"titan": tstats}, "comparability_reason": "; ".join(reason) or "provisional until paired with compatible artifact"}
                report["rows"].append(row)
    configs = []
    for _, data in loaded:
        cfg = data.get("configuration", {})
        configs.append(tuple(cfg.get(k) for k in ("generated_tokens", "temperature", "repetitions")))
    if len(set(configs)) > 1:
        for row in report["rows"]:
            row["ratio"] = None
            row["comparability_reason"] = "incompatible configuration: generated_tokens, temperature or repetitions"
    # Pair only exact identities; historical/current labels remain visible and never silently merge.
    rows = report["rows"]
    for row in rows:
        for other in rows:
            if row is other or row["model"] != other["model"] or row["condition"] != other["condition"]: continue
            if row["titan_metric"] is None or other["llama_metric"] is None: continue
            if row["comparability_reason"] != "provisional until paired with compatible artifact": continue
            row["ratio"] = row["titan_metric"] / other["llama_metric"]
            row["comparability_reason"] = "compatible model/configuration identity"
            break
    current = [r for r in rows if r["path_classification"].startswith("current") and r["titan_metric"] is not None]
    historical = [r for r in rows if r["path_classification"] == "historical Q8" and r["model"]]
    compatible = current and historical and all(r["comparability_reason"] == "compatible model/configuration identity" for r in current)
    reason = "no compatible current-vs-historical checkpoint identity"
    if current and historical or len(loaded) > 1:
        reason = "build identity differs or is missing"
    report["compatibility_summary"] = {"comparable_rows": sum(r["ratio"] is not None for r in rows), "regression": None,
        "regression_reason": reason if not compatible else "not calculated"}
    report["status"] = "error" if failed else "diagnostic"
    try:
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except OSError: return 1
    print(json.dumps(report, indent=2)); return 1 if failed else 0

if __name__ == "__main__": sys.exit(main())
