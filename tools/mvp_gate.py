#!/usr/bin/env python3
"""Stable Functional MVP acceptance gate for Titan.

Evaluates clean benchmark artifacts, external F8 correctness evidence,
correctness-to-build binding, and operational evidence under an advisory-only
performance profile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

# Ensure tools directory is in sys.path for local module imports
TOOLS_DIR = Path(__file__).parent.resolve()
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from release_gate import (
    SCHEMA_VERSION,
    CONDITIONS,
    ENGINES,
    CONFIG_KEYS,
    OPTIONAL_CONFIG_KEYS,
    DEFAULT_MODELS,
    SUMMARY_KEYS,
    IDENTITY_RESULT_FIELDS,
    IDENTITY_PROMPT_FIELDS,
    WORKLOAD_FIELDS,
    load_json,
    finite_number,
    finite_positive,
    positive_int,
    contains_key,
    check_telemetry_contamination,
    validate_artifact,
    compare_configurations,
    compare_identities,
    metric,
    ProducerArtifactAdapter,
    load_and_verify_correctness_evidence,
)

# Declared MVP statuses
MVP_STATUS_ACCEPTED_FUNCTIONAL = "mvp_accepted_functional"
MVP_STATUS_ACCEPTED_WITH_PERFORMANCE_ADVISORY = "mvp_accepted_with_performance_advisory"
MVP_STATUS_BLOCKED_CORRECTNESS = "mvp_blocked_correctness"
MVP_STATUS_BLOCKED_OPERATIONAL = "mvp_blocked_operational"
MVP_STATUS_BLOCKED_EVIDENCE = "mvp_blocked_evidence"
MVP_STATUS_NOT_EXERCISED = "mvp_not_exercised"

ALLOWED_MVP_STATUSES = {
    MVP_STATUS_ACCEPTED_FUNCTIONAL,
    MVP_STATUS_ACCEPTED_WITH_PERFORMANCE_ADVISORY,
    MVP_STATUS_BLOCKED_CORRECTNESS,
    MVP_STATUS_BLOCKED_OPERATIONAL,
    MVP_STATUS_BLOCKED_EVIDENCE,
    MVP_STATUS_NOT_EXERCISED,
}

# Explicit advisory allowlist: ONLY these failures may be downgraded from blocking to advisory
ADVISORY_ALLOWLIST = {
    "per_model_ratio",
    "aggregate_ratio",
    "regression",
    "regression_failed",
    "regression_not_evaluated",
    "regression_incomparable",
    "unstable_reference",
    "ffn_sync_unavailable",
    "ffn_sync_missing_frontier",
    "asymmetric_reference_stage_coverage",
    "nsight_counters_blocked",
    "causal_frontier_diagnostic_incomplete",
}

REQUIRED_OPERATIONAL_CASES = (
    "resident_json",
    "streaming_sse",
    "grammar_constrained_json",
    "readiness",
    "bounded_completion",
    "cleanup",
    "negative_lifecycle",
)

BENCHMARK_STRUCTURE_FAILURES = {
    "benchmark_missing",
    "benchmark_invalid",
    "benchmark_schema",
    "artifact_status",
    "benchmark_results",
    "benchmark_model_identity",
    "missing_model",
    "extra_model",
    "duplicate_model",
    "configuration",
    "workload_compatibility",
    "engine_identity",
    "repetitions",
    "repetitions_mismatch",
    "generated_tokens_mismatch",
    "decode_path_invalid",
    "benchmark_claims_passed_correctness",
    "summary_statistics",
    "sample_count",
    "raw_samples",
    "model_identity_mismatch",
    "model_path_mismatch",
    "model_hash_missing",
    "model_hash_mismatch",
}

CLEAN_EXECUTION_SAFETY_FAILURES = {
    "observed_graph_active",
    "observed_graph_missing",
    "diagnostic_telemetry_contamination",
    "diagnostic_instrumentation",
    "candidate_isolation_violation",
    "clean_mode_unverified",
    "instrumentation_mode_missing",
}


def is_benchmark_structure_failure(failure: str) -> bool:
    """Returns True if the failure code pertains to benchmark structure, models, summaries, or configuration."""
    if failure in BENCHMARK_STRUCTURE_FAILURES:
        return True
    if failure.startswith("raw_metric_") or failure.startswith("metric_"):
        return True
    return False


def is_clean_execution_safety_failure(failure: str) -> bool:
    """Returns True if the failure code pertains to clean execution safety, telemetry, candidate isolation, or graphs."""
    return failure in CLEAN_EXECUTION_SAFETY_FAILURES


def is_advisory_failure(failure: str) -> bool:
    """Returns True if and only if the failure code is on the explicit advisory allowlist."""
    return failure in ADVISORY_ALLOWLIST


def compute_file_sha256(path: Path | str) -> str:
    """Computes SHA-256 digest of a file."""
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def normalize_path(path_str: str) -> str:
    """Strips Windows extended path prefix and normalizes separators."""
    clean = path_str.replace("\\\\?\\", "").replace("\\?\\", "")
    return clean.replace("\\", "/")


def validate_observed_graph(bench_data: dict[str, Any], failures: list[str]) -> bool:
    """Verifies that observed Graph execution is explicitly disabled with zero counts."""
    valid = True
    config = bench_data.get("configuration")
    if not isinstance(config, dict):
        failures.append("observed_graph_missing")
        return False

    obs_graph = config.get("observed_graph")
    if not isinstance(obs_graph, dict):
        failures.append("observed_graph_missing")
        valid = False
    else:
        if (
            obs_graph.get("status") != "disabled"
            or obs_graph.get("requested") is not False
            or obs_graph.get("capture_count") != 0
            or obs_graph.get("replay_count") != 0
        ):
            failures.append("observed_graph_active")
            valid = False

    # Also check per-prompt graph observations in results
    results = bench_data.get("results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            runs = item.get("runs")
            if isinstance(runs, list):
                for run in runs:
                    if not isinstance(run, dict):
                        continue
                    prompts = run.get("prompts")
                    if isinstance(prompts, list):
                        for p in prompts:
                            if not isinstance(p, dict):
                                continue
                            if p.get("graph") is True:
                                failures.append("observed_graph_active")
                                valid = False
                            if p.get("observed_graph_status") not in (None, "disabled"):
                                failures.append("observed_graph_active")
                                valid = False
                            gs = p.get("graph_state")
                            if isinstance(gs, dict):
                                if (
                                    gs.get("status") != "disabled"
                                    or gs.get("requested") is not False
                                    or gs.get("capture_count") != 0
                                    or gs.get("replay_count") != 0
                                ):
                                    failures.append("observed_graph_active")
                                    valid = False
    return valid


def validate_correctness_build_binding(
    binding_data: dict[str, Any],
    f8_data: dict[str, Any],
    f8_sha256: str,
    bench_data: dict[str, Any],
    bench_sha256: str,
    failures: list[str],
) -> dict[str, Any]:
    """Validates that the binding manifest ties F8 correctness to the exact benchmark build identity."""
    binding_info = {}
    if not isinstance(binding_data, dict):
        failures.append("correctness_binding_invalid")
        return binding_info

    if binding_data.get("status") != "verified":
        failures.append("build_binding_unverified")

    # Check F8 reference in binding
    f8_ref = binding_data.get("f8_evidence")
    if not isinstance(f8_ref, dict):
        failures.append("build_binding_f8_missing")
    else:
        ref_f8_sha = f8_ref.get("sha256")
        if not ref_f8_sha or ref_f8_sha.strip().lower() != f8_sha256.strip().lower():
            failures.append("build_binding_f8_sha256_mismatch")

    # Check benchmark reference in binding
    bench_ref = binding_data.get("benchmark")
    if not isinstance(bench_ref, dict):
        failures.append("build_binding_benchmark_missing")
    else:
        ref_bench_sha = bench_ref.get("sha256")
        if not ref_bench_sha or ref_bench_sha.strip().lower() != bench_sha256.strip().lower():
            failures.append("build_binding_benchmark_sha256_mismatch")

    # Check binding block
    b_block = binding_data.get("binding")
    if not isinstance(b_block, dict):
        failures.append("build_binding_missing")
    else:
        binding_status = b_block.get("status")
        if binding_status != "verified":
            failures.append("build_binding_unverified")

        titan_build = b_block.get("titan_build")
        bench_titan_identity = bench_data.get("engine_identity", {}).get("titan", {})
        bench_titan_build = bench_titan_identity.get("build") if isinstance(bench_titan_identity, dict) else None

        if not titan_build or not bench_titan_build or str(titan_build).strip() != str(bench_titan_build).strip():
            failures.append("build_binding_mismatch")
        else:
            tb_lower = str(titan_build).lower()
            if any(p in tb_lower for p in ("unreported", "unknown", "placeholder", "unverified")):
                failures.append("build_binding_unverified")

        proof_kind = b_block.get("proof_kind") or b_block.get("verification_method") or b_block.get("relationship")
        if not proof_kind or not str(proof_kind).strip():
            failures.append("build_binding_proof_missing")

        binding_info = {
            "status": binding_status,
            "titan_build": titan_build,
            "proof_kind": proof_kind,
            "titan_source_identity": b_block.get("titan_source_identity"),
        }

    return binding_info


def validate_operational_manifest(
    op_data: dict[str, Any],
    failures: list[str],
) -> list[dict[str, Any]]:
    """Semantically validates the operational evidence manifest across all required cases."""
    case_results: list[dict[str, Any]] = []
    if not isinstance(op_data, dict):
        failures.append("operational_evidence_invalid")
        return case_results

    if op_data.get("status") != "verified":
        failures.append("operational_evidence_unverified")

    cases_raw = op_data.get("cases")
    cases_dict: dict[str, dict[str, Any]] = {}
    if isinstance(cases_raw, dict):
        cases_dict = cases_raw
    elif isinstance(cases_raw, list):
        for item in cases_raw:
            if isinstance(item, dict):
                cid = item.get("case_id") or item.get("name")
                if cid:
                    cases_dict[str(cid)] = item
    else:
        failures.append("operational_cases_missing")
        return case_results

    for req_case in REQUIRED_OPERATIONAL_CASES:
        case = cases_dict.get(req_case)
        if case is None:
            failures.append(f"operational_case_{req_case}_missing")
            continue

        case_record: dict[str, Any] = {"case_id": req_case}
        # Command
        cmd = case.get("command")
        if not cmd or not str(cmd).strip():
            failures.append(f"operational_case_{req_case}_invalid")
        case_record["command"] = cmd

        # Exit code
        exit_code = case.get("exit_code")
        if exit_code != 0:
            failures.append(f"operational_case_{req_case}_nonzero_exit")
        case_record["exit_code"] = exit_code

        # Execution counts
        executed = case.get("executed")
        failed = case.get("failed")
        skipped = case.get("skipped")
        if not positive_int(executed) or executed < 1:
            failures.append(f"operational_case_{req_case}_not_executed")
        if failed != 0:
            failures.append(f"operational_case_{req_case}_failed")
        if skipped != 0:
            failures.append(f"operational_case_{req_case}_skipped")
        case_record["executed"] = executed
        case_record["failed"] = failed
        case_record["skipped"] = skipped

        # Fixture identity
        fixture = case.get("fixture")
        if not fixture:
            failures.append(f"operational_case_{req_case}_fixture_missing")
        case_record["fixture"] = fixture

        # Raw log path & SHA-256
        rl_path_str: str | None = None
        rl_sha: str | None = None
        raw_log_info = case.get("raw_log")
        if isinstance(raw_log_info, dict):
            rl_path_str = raw_log_info.get("path")
            rl_sha = raw_log_info.get("sha256")
        elif isinstance(raw_log_info, str):
            rl_path_str = raw_log_info
            rl_sha = case.get("raw_log_sha256") or case.get("sha256")
        else:
            rl_path_str = case.get("raw_log_path")
            rl_sha = case.get("raw_log_sha256")

        if not rl_sha or not str(rl_sha).strip():
            failures.append(f"operational_case_{req_case}_raw_log_sha_missing")

        if not rl_path_str:
            failures.append(f"operational_case_{req_case}_raw_log_missing")
        else:
            clean_rl_path = normalize_path(rl_path_str)
            rl_path = Path(clean_rl_path)
            if not rl_path.is_file() or rl_path.stat().st_size == 0:
                failures.append(f"operational_case_{req_case}_raw_log_invalid")
            elif rl_sha and str(rl_sha).strip():
                actual_sha = hashlib.sha256(rl_path.read_bytes()).hexdigest()
                if actual_sha.lower() != str(rl_sha).strip().lower():
                    failures.append(f"operational_case_{req_case}_raw_log_sha_mismatch")
        case_record["raw_log_path"] = rl_path_str
        case_record["raw_log_sha256"] = rl_sha

        # Semantic assertions
        assertions = case.get("semantic_assertions")
        if not isinstance(assertions, dict):
            failures.append(f"operational_case_{req_case}_assertions_missing")
        else:
            case_record["semantic_assertions"] = assertions
            if req_case == "grammar_constrained_json":
                # Grammar complete + serde parse success + exactly {"city": "Tokyo"}
                gc = assertions.get("grammar_complete")
                serde_ok = assertions.get("serde_parse_success") or assertions.get("serde_parsed")
                output_json = assertions.get("output_json") or assertions.get("parsed_json")
                if gc is not True or serde_ok is not True or output_json != {"city": "Tokyo"}:
                    failures.append("grammar_semantic_mismatch")

        case_results.append(case_record)

    return case_results


def cross_check_f8_and_benchmark_models(
    f8_data: dict[str, Any],
    bench_data: dict[str, Any],
    failures: list[str],
) -> list[dict[str, Any]]:
    """Verifies that F8 models and benchmark models match in identity, route, and paths."""
    f8_models_list = f8_data.get("models", [])
    bench_results = bench_data.get("results", [])

    f8_by_name: dict[str, dict[str, Any]] = {}
    if isinstance(f8_models_list, list):
        for m in f8_models_list:
            if isinstance(m, dict):
                name = m.get("model_name") or m.get("model")
                if name:
                    f8_by_name[name] = m

    bench_by_name: dict[str, dict[str, Any]] = {}
    if isinstance(bench_results, list):
        for r in bench_results:
            if isinstance(r, dict):
                name = r.get("model")
                if name:
                    bench_by_name[name] = r

    model_identities: list[dict[str, Any]] = []
    for model_name in DEFAULT_MODELS:
        f8_m = f8_by_name.get(model_name)
        b_m = bench_by_name.get(model_name)
        if f8_m is None or b_m is None:
            failures.append("model_identity_mismatch")
            continue

        f8_path = normalize_path(str(f8_m.get("model_path", "")))
        bench_path = normalize_path(str(b_m.get("model_path", "")))

        # Compare file basenames
        if Path(f8_path).name != Path(bench_path).name:
            failures.append("model_path_mismatch")

        # Compare hashes
        f8_hash = f8_m.get("computed_sha256") or f8_m.get("model_hash") or f8_m.get("model_sha256")
        bench_hash = b_m.get("model_hash") or b_m.get("model_sha256")
        if not f8_hash or not str(f8_hash).strip() or not bench_hash or not str(bench_hash).strip():
            failures.append("model_hash_missing")
        elif str(f8_hash).strip().lower() != str(bench_hash).strip().lower():
            failures.append("model_hash_mismatch")

        model_identities.append({
            "model": model_name,
            "f8_path": f8_path,
            "benchmark_path": bench_path,
            "f8_hash": f8_hash,
            "benchmark_hash": bench_hash,
        })

    return model_identities


def evaluate_mvp(
    benchmark_path: Path | str,
    correctness_evidence_path: Path | str,
    correctness_sha256: str | None,
    correctness_binding_path: Path | str,
    operational_evidence_path: Path | str,
    regression_baseline_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> tuple[dict[str, Any], int]:
    """Core evaluation function for the Stable Functional MVP gate."""
    # Resolve all paths to absolute
    bench_p = Path(benchmark_path).resolve()
    f8_p = Path(correctness_evidence_path).resolve()
    binding_p = Path(correctness_binding_path).resolve()
    op_p = Path(operational_evidence_path).resolve()
    reg_p = Path(regression_baseline_path).resolve() if regression_baseline_path else None
    out_p = Path(output_path).resolve() if output_path else None

    input_paths: dict[str, str] = {
        "benchmark": str(bench_p),
        "correctness_evidence": str(f8_p),
        "correctness_binding": str(binding_p),
        "operational_evidence": str(op_p),
    }
    if reg_p:
        input_paths["regression_baseline"] = str(reg_p)
    if out_p:
        input_paths["output"] = str(out_p)

    input_hashes: dict[str, str] = {}
    correctness_failures: list[str] = []
    evidence_failures: list[str] = []
    operational_failures: list[str] = []
    advisory_warnings: list[str] = []

    # 1. Load benchmark
    bench_data: dict[str, Any] = {}
    if not bench_p.is_file():
        evidence_failures.append("benchmark_missing")
    else:
        try:
            bench_data = load_json(str(bench_p))
            input_hashes["benchmark_sha256"] = compute_file_sha256(bench_p)
        except Exception:
            evidence_failures.append("benchmark_invalid")

    # 2. Load and verify F8 correctness evidence
    f8_data: dict[str, Any] = {}
    f8_valid = False
    if not correctness_sha256 or not str(correctness_sha256).strip():
        correctness_failures.append("correctness_evidence_invalid")

    if not f8_p.is_file():
        correctness_failures.append("correctness_evidence_missing")
    else:
        try:
            f8_data = load_json(str(f8_p))
            actual_f8_sha = compute_file_sha256(f8_p)
            input_hashes["correctness_evidence_sha256"] = actual_f8_sha
            if correctness_sha256 and str(correctness_sha256).strip():
                f8_valid = load_and_verify_correctness_evidence(
                    str(f8_p), expected_sha256=correctness_sha256, failures=correctness_failures
                )
            else:
                load_and_verify_correctness_evidence(
                    str(f8_p), expected_sha256=None, failures=correctness_failures
                )
                f8_valid = False
        except Exception:
            correctness_failures.append("correctness_evidence_invalid")

    # 3. Load correctness-to-build binding
    binding_data: dict[str, Any] = {}
    if not binding_p.is_file():
        evidence_failures.append("correctness_binding_missing")
    else:
        try:
            binding_data = load_json(str(binding_p))
            input_hashes["correctness_binding_sha256"] = compute_file_sha256(binding_p)
        except Exception:
            evidence_failures.append("correctness_binding_invalid")

    # 4. Load operational evidence
    op_data: dict[str, Any] = {}
    if not op_p.is_file():
        operational_failures.append("operational_evidence_missing")
    else:
        try:
            op_data = load_json(str(op_p))
            input_hashes["operational_evidence_sha256"] = compute_file_sha256(op_p)
        except Exception:
            operational_failures.append("operational_evidence_invalid")

    # 5. Load regression baseline if provided
    reg_data: dict[str, Any] | None = None
    if reg_p:
        if not reg_p.is_file():
            advisory_warnings.append("regression_incomparable")
        else:
            try:
                reg_data = load_json(str(reg_p))
                input_hashes["regression_baseline_sha256"] = compute_file_sha256(reg_p)
            except Exception:
                advisory_warnings.append("regression_incomparable")

    # Perform structural validation on benchmark
    bench_view = None
    if bench_data:
        bench_view = validate_artifact(bench_data, "benchmark", evidence_failures)
        validate_observed_graph(bench_data, evidence_failures)

        # Check clean producer requirements
        cfg = bench_data.get("configuration", {})
        if cfg.get("generated_tokens") != 41:
            evidence_failures.append("generated_tokens_mismatch")
        if cfg.get("repetitions") != 3:
            evidence_failures.append("repetitions_mismatch")
        if str(cfg.get("decode_path", "")).lower() != "f32":
            evidence_failures.append("decode_path_invalid")

        # Correctness status inside throughput benchmark MUST be 'not_evaluated'
        corr = bench_data.get("correctness")
        if not isinstance(corr, dict) or corr.get("status") != "not_evaluated":
            evidence_failures.append("benchmark_claims_passed_correctness")

    # Cross-check F8 and benchmark models
    model_identities: list[dict[str, Any]] = []
    if f8_data and bench_data:
        model_identities = cross_check_f8_and_benchmark_models(f8_data, bench_data, evidence_failures)

    # Validate binding manifest
    binding_info: dict[str, Any] = {}
    if binding_data and f8_data and bench_data:
        binding_info = validate_correctness_build_binding(
            binding_data=binding_data,
            f8_data=f8_data,
            f8_sha256=input_hashes.get("correctness_evidence_sha256", ""),
            bench_data=bench_data,
            bench_sha256=input_hashes.get("benchmark_sha256", ""),
            failures=evidence_failures,
        )

    # Validate operational manifest
    operational_case_results: list[dict[str, Any]] = []
    if op_data:
        operational_case_results = validate_operational_manifest(op_data, operational_failures)

    # Compute advisory performance ratios
    comparisons: dict[str, Any] = {}
    ratios: dict[str, list[float]] = {condition: [] for condition in CONDITIONS}
    aggregate: dict[str, Any] = {
        "cold_ratio": None,
        "warm_ratio": None,
        "overall_ratio": None,
        "method": "median of available per-model cold/warm ratios; overall median across both conditions",
    }

    if bench_view and bench_view.valid:
        for model in DEFAULT_MODELS:
            item = bench_view.items.get(model)
            if item is None:
                continue
            comparisons[model] = {}
            for condition in CONDITIONS:
                llama = metric(item, condition, SUMMARY_KEYS["llama"], evidence_failures, model)
                titan = metric(item, condition, SUMMARY_KEYS["titan"], evidence_failures, model)
                if llama is not None and titan is not None:
                    ratio = titan / llama
                    ratios[condition].append(ratio)
                    comparisons[model][condition] = {
                        "llama_median": llama,
                        "titan_median": titan,
                        "ratio": ratio,
                    }
                    if ratio < 0.95:
                        advisory_warnings.append("per_model_ratio")

        complete_ratios = all(len(ratios[cond]) == len(DEFAULT_MODELS) for cond in CONDITIONS)
        if complete_ratios:
            aggregate["cold_ratio"] = float(statistics.median(ratios["cold"]))
            aggregate["warm_ratio"] = float(statistics.median(ratios["warm"]))
            all_ratios = ratios["cold"] + ratios["warm"]
            aggregate["overall_ratio"] = float(statistics.median(all_ratios))
            if aggregate["overall_ratio"] < 0.95:
                advisory_warnings.append("aggregate_ratio")
        else:
            advisory_warnings.append("aggregate_ratio")

    # Compute advisory regression
    regression_state = "not_evaluated"
    if reg_data is None:
        if reg_p is None:
            advisory_warnings.append("regression_not_evaluated")
        else:
            regression_state = "incomparable"
            advisory_warnings.append("regression_incomparable")
    else:
        reg_failures: list[str] = []
        reg_view = validate_artifact(reg_data, "regression", reg_failures)
        if not reg_view.valid or not bench_view or not bench_view.valid:
            regression_state = "incomparable"
            advisory_warnings.append("regression_incomparable")
        elif not compare_configurations(reg_view, bench_view, reg_failures) or not compare_identities(
            reg_view, bench_view, reg_failures
        ):
            regression_state = "incomparable"
            advisory_warnings.append("regression_incomparable")
        else:
            reg_failed = False
            for model in DEFAULT_MODELS:
                old_item = reg_view.items.get(model)
                now_item = bench_view.items.get(model)
                if old_item is None or now_item is None:
                    reg_failed = True
                    continue
                for condition in CONDITIONS:
                    prev = metric(old_item, condition, SUMMARY_KEYS["titan"], reg_failures, model)
                    cur = metric(now_item, condition, SUMMARY_KEYS["titan"], reg_failures, model)
                    if prev is not None and cur is not None and cur < prev * 0.95:
                        reg_failed = True
                        advisory_warnings.append("regression")
                        advisory_warnings.append("regression_failed")
            regression_state = "failed" if reg_failed else "passed"

    # Classify failures through the strict advisory allowlist
    raw_all_failures = sorted(set(correctness_failures + evidence_failures + operational_failures))
    blocking_failures: list[str] = []
    for f in raw_all_failures:
        if is_advisory_failure(f):
            advisory_warnings.append(f)
        else:
            blocking_failures.append(f)

    blocking_failures = sorted(set(blocking_failures))
    advisory_warnings = sorted(set(advisory_warnings))

    # Gate verdicts
    gate_verdicts: dict[str, str] = {
        "correctness": "failed" if any(f in blocking_failures for f in correctness_failures) or not f8_valid else "passed",
        "correctness_binding": "failed" if any("binding" in f for f in blocking_failures) or not binding_data else "passed",
        "operational": "failed" if any(f in blocking_failures for f in operational_failures) or not op_data else "passed",
        "benchmark_structure": "failed" if (
            not bench_data
            or bench_view is None
            or any(is_benchmark_structure_failure(f) for f in blocking_failures)
        ) else "passed",
        "clean_execution_safety": "failed" if any(
            is_clean_execution_safety_failure(f) for f in blocking_failures
        ) else "passed",
        "ratio_advisory": "advisory_warning" if "aggregate_ratio" in advisory_warnings or "per_model_ratio" in advisory_warnings else "passed",
        "regression_advisory": regression_state,
    }

    # Status classification
    if gate_verdicts["correctness"] == "failed":
        status = MVP_STATUS_BLOCKED_CORRECTNESS
        exit_code = 1
    elif gate_verdicts["operational"] == "failed":
        status = MVP_STATUS_BLOCKED_OPERATIONAL
        exit_code = 1
    elif blocking_failures or gate_verdicts["correctness_binding"] == "failed" or gate_verdicts["benchmark_structure"] == "failed" or gate_verdicts["clean_execution_safety"] == "failed":
        status = MVP_STATUS_BLOCKED_EVIDENCE
        exit_code = 1
    else:
        # All functional gates passed!
        if advisory_warnings:
            status = MVP_STATUS_ACCEPTED_WITH_PERFORMANCE_ADVISORY
        else:
            status = MVP_STATUS_ACCEPTED_FUNCTIONAL
        exit_code = 0

    decision_id = f"mvp-decision-{hashlib.sha256(str(bench_p).encode()).hexdigest()[:16]}"
    titan_engine = bench_data.get("engine_identity", {}).get("titan", {}) if isinstance(bench_data, dict) else {}

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": decision_id,
        "status": status,
        "release_approval": "not_granted",
        "promotion_authorized": False,
        "rc_generated": False,
        "input_paths": input_paths,
        "input_hashes": input_hashes,
        "gate_verdicts": gate_verdicts,
        "blocking_failures": blocking_failures,
        "advisory_warnings": advisory_warnings,
        "advisories": {
            "aggregate": aggregate,
            "comparisons": comparisons,
            "regression": {
                "status": regression_state,
            },
            "deferred_limitations": [
                "ffn_sync unavailable / missing real frontier",
                "asymmetric llama.cpp stage coverage",
                "Nsight counters ERR_NVGPUCTRPERM",
            ],
        },
        "identities": {
            "models": model_identities,
            "engine_identity": bench_data.get("engine_identity"),
            "titan_build": titan_engine.get("build") if isinstance(titan_engine, dict) else None,
            "binding": binding_info,
        },
        "operational_case_results": operational_case_results,
        "reproduction_metadata": {
            "command": sys.argv,
            "cwd": str(Path.cwd()),
        },
    }

    if out_p:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    return report, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Titan Stable Functional MVP Gate")
    parser.add_argument("--benchmark", required=True, help="Clean benchmark artifact JSON")
    parser.add_argument("--correctness-evidence", required=True, help="External F8 correctness evidence JSON")
    parser.add_argument("--correctness-sha256", required=True, help="Expected SHA-256 for correctness evidence")
    parser.add_argument("--correctness-binding", required=True, help="Manifest binding F8 to benchmark Titan build")
    parser.add_argument("--operational-evidence", required=True, help="Operational evidence manifest JSON")
    parser.add_argument("--regression-baseline", help="Optional regression baseline benchmark JSON")
    parser.add_argument("--output", required=True, help="Output decision JSON path")

    args = parser.parse_args(argv)

    report, exit_code = evaluate_mvp(
        benchmark_path=args.benchmark,
        correctness_evidence_path=args.correctness_evidence,
        correctness_sha256=args.correctness_sha256,
        correctness_binding_path=args.correctness_binding,
        operational_evidence_path=args.operational_evidence,
        regression_baseline_path=args.regression_baseline,
        output_path=args.output,
    )

    print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
