#!/usr/bin/env python3
"""Comprehensive synthetic unit and CLI tests for tools/mvp_gate.py."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

# Ensure tools directory is on sys.path
TOOLS_DIR = Path(__file__).parent.resolve()
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

try:
    import release_gate
except ImportError:
    release_gate = None

try:
    import mvp_gate
except ImportError:
    mvp_gate = None

DEFAULT_MODELS = (
    "Qwen 2.5 1.5B Instruct",
    "Llama 3.2 1B Instruct",
    "Llama 3.2 3B Instruct",
    "DeepSeek-R1-Distill 1.5B",
    "Qwen3 0.6B Base/Chat",
)

MODEL_PATHS = {
    "Qwen 2.5 1.5B Instruct": "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "Llama 3.2 1B Instruct": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "Llama 3.2 3B Instruct": "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "DeepSeek-R1-Distill 1.5B": "models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
    "Qwen3 0.6B Base/Chat": "testdata/Qwen3-0.6B-Q4_K_M.gguf",
}

MODEL_HASHES = {
    "Qwen 2.5 1.5B Instruct": "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
    "Llama 3.2 1B Instruct": "6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83",
    "Llama 3.2 3B Instruct": "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff",
    "DeepSeek-R1-Distill 1.5B": "1741e5b2d062b07acf048bf0d2c514dadf2a48f94e2b4aa0cfe069af3838ee2f",
    "Qwen3 0.6B Base/Chat": "ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


class MvpGateTestBase(unittest.TestCase):
    """Test harness creating minimal valid synthetic fixtures."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create synthetic raw log files
        self.f8_raw_log = self.root / "f8_test.raw.log"
        self.f8_raw_log.write_text("F8 test raw log content\n", encoding="utf-8")

        self.op_resident_raw_log = self.root / "resident.raw.log"
        self.op_resident_raw_log.write_text("Resident test log: ok\n", encoding="utf-8")

        self.op_streaming_raw_log = self.root / "streaming.raw.log"
        self.op_streaming_raw_log.write_text("Streaming test log: ok\n", encoding="utf-8")

        self.op_grammar_raw_log = self.root / "grammar.raw.log"
        self.op_grammar_raw_log.write_text(
            "[Grammar] Reached complete JSON document state at step 7\n"
            '{ "city": "Tokyo" }\n'
            'Object {"city": String("Tokyo")}\n'
            "test result: ok\n",
            encoding="utf-8",
        )

        self.op_readiness_raw_log = self.root / "readiness.raw.log"
        self.op_readiness_raw_log.write_text("Readiness test log: ok\n", encoding="utf-8")

        self.op_bounded_raw_log = self.root / "bounded.raw.log"
        self.op_bounded_raw_log.write_text("Bounded test log: ok\n", encoding="utf-8")

        self.op_cleanup_raw_log = self.root / "cleanup.raw.log"
        self.op_cleanup_raw_log.write_text("Cleanup test log: ok\n", encoding="utf-8")

        self.op_negative_raw_log = self.root / "negative.raw.log"
        self.op_negative_raw_log.write_text(
            "test lifecycle_tests::missing_fixture_is_explicitly_blocked ... ok\n"
            "test lifecycle_tests::readiness_timeout_is_bounded ... ok\n"
            "test lifecycle_tests::no_gpu_ready_server_can_be_cleaned_up ... ok\n",
            encoding="utf-8",
        )

        self.titan_build = "sha256:bc92211356bb0eb87ddd59e252d9d4497f8669d7257ac0133459040ce4eb131c"
        self.llama_build = "sha256:0da24b1e082e87df7db915d0e0e5d162eebe423e25b687d73475e6d357d5b92a"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_synthetic_f8(self, **overrides: Any) -> dict[str, Any]:
        models_list = []
        for name in DEFAULT_MODELS:
            models_list.append({
                "model_id": name.lower().replace(" ", "-"),
                "model_name": name,
                "model_path": MODEL_PATHS[name],
                "computed_sha256": MODEL_HASHES[name],
                "status": "verified",
            })
        base: dict[str, Any] = {
            "schema_version": "1.0.0",
            "manifest_id": "titan-f32-five-model-correctness-v1",
            "status": "verified",
            "raw_log_path": str(self.f8_raw_log),
            "production_dispatch_changed": False,
            "candidate_selectors_absent": True,
            "active_candidate_selectors": [],
            "route_configuration": {
                "input_format": "F32",
                "batch_size": 1,
                "cache_condition": "resident",
                "graph": False,
            },
            "models": models_list,
        }
        base.update(overrides)
        return base

    def make_synthetic_benchmark(
        self, titan_speed: float = 20.0, llama_speed: float = 40.0, **overrides: Any
    ) -> dict[str, Any]:
        """Creates a clean synthetic benchmark (defaults to ~0.50x ratio, low throughput)."""
        results = []
        for name in DEFAULT_MODELS:
            runs = []
            for rep in (1, 2, 3):
                prompts = [
                    {
                        "prompt_index": 1,
                        "cache_condition": "cold",
                        "prompt_tokens": 32,
                        "llama": {"decode_tok_s": llama_speed, "generated_tokens": 41},
                        "titan": {"decode_tok_s": titan_speed, "generated_tokens": 41},
                    },
                    {
                        "prompt_index": 2,
                        "cache_condition": "warm",
                        "prompt_tokens": 32,
                        "llama": {"decode_tok_s": llama_speed, "generated_tokens": 41},
                        "titan": {"decode_tok_s": titan_speed, "generated_tokens": 41},
                    },
                ]
                runs.append({"repetition": rep, "prompts": prompts})
            stats = {
                "cold": {
                    "llama_decode_tok_s": {
                        "samples": 3,
                        "median": float(llama_speed),
                        "variance": 0.0,
                        "stddev": 0.0,
                    },
                    "titan_decode_tok_s": {
                        "samples": 3,
                        "median": float(titan_speed),
                        "variance": 0.0,
                        "stddev": 0.0,
                    },
                },
                "warm": {
                    "llama_decode_tok_s": {
                        "samples": 3,
                        "median": float(llama_speed),
                        "variance": 0.0,
                        "stddev": 0.0,
                    },
                    "titan_decode_tok_s": {
                        "samples": 3,
                        "median": float(titan_speed),
                        "variance": 0.0,
                        "stddev": 0.0,
                    },
                },
            }
            results.append({
                "model": name,
                "model_path": MODEL_PATHS[name],
                "model_hash": MODEL_HASHES[name],
                "repetitions": 3,
                "runs": runs,
                "statistics": stats,
            })

        base: dict[str, Any] = {
            "schema_version": 2,
            "status": "complete",
            "instrumentation_mode": "clean_throughput",
            "candidate_authorization": False,
            "production_dispatch_changed": False,
            "candidate_selectors_absent": True,
            "active_candidate_selectors": [],
            "configuration": {
                "generated_tokens": 41,
                "temperature": 0.0,
                "repetitions": 3,
                "cuda_graphs": False,
                "decode_path": "f32",
                "observed_graph": {
                    "requested": False,
                    "status": "disabled",
                    "capture_count": 0,
                    "replay_count": 0,
                },
            },
            "workload_compatibility": {
                "workload_id": "multi_model_head_to_head_v2",
                "model_suite": list(DEFAULT_MODELS),
                "prompt_set": "release-prompts-v1",
                "conditions": [
                    {"prompt_index": 1, "cache_condition": "cold"},
                    {"prompt_index": 2, "cache_condition": "warm"},
                ],
                "generated_tokens": 41,
                "temperature": 0.0,
                "repetitions": 3,
            },
            "engine_identity": {
                "llama": {"name": "llama.cpp", "build": self.llama_build},
                "titan": {"name": "Titan", "build": self.titan_build},
            },
            "correctness": {
                "status": "not_evaluated",
                "reason": "throughput benchmark does not evaluate generation correctness",
                "required_independent_evidence": "F8.R1.P",
            },
            "results": results,
        }
        base.update(overrides)
        return base

    def make_synthetic_binding(
        self, f8_path: Path, benchmark_path: Path, **overrides: Any
    ) -> dict[str, Any]:
        models_record = {name: {"path": MODEL_PATHS[name], "sha256": MODEL_HASHES[name]} for name in DEFAULT_MODELS}
        base: dict[str, Any] = {
            "schema_version": 1,
            "status": "verified",
            "f8_evidence": {
                "path": str(f8_path.resolve()),
                "sha256": sha256_file(f8_path),
                "models": models_record,
            },
            "benchmark": {
                "path": str(benchmark_path.resolve()),
                "sha256": sha256_file(benchmark_path),
                "engine_identity": {
                    "titan": {"name": "Titan", "build": self.titan_build},
                },
            },
            "binding": {
                "status": "verified",
                "titan_build": self.titan_build,
                "titan_source_identity": "git:e15fa907f0ccad0aee75b4ac85d0ec08411fb675",
                "proof_kind": "binary_source_identity_match",
                "verification_method": "independent_build_provenance",
            },
        }
        base.update(overrides)
        return base

    def make_synthetic_operational(self, **overrides: Any) -> dict[str, Any]:
        cases: dict[str, Any] = {
            "resident_json": {
                "command": "cargo test --test e2e_unified_modes_gate resident",
                "exit_code": 0,
                "executed": 1,
                "failed": 0,
                "skipped": 0,
                "fixture": {"path": "testdata/Qwen3-0.6B-Q4_K_M.gguf", "sha256": MODEL_HASHES["Qwen3 0.6B Base/Chat"]},
                "raw_log": {
                    "path": str(self.op_resident_raw_log.resolve()),
                    "sha256": sha256_file(self.op_resident_raw_log),
                },
                "semantic_assertions": {"status": "verified", "engine_mode": "Resident"},
            },
            "streaming_sse": {
                "command": "cargo test --test e2e_unified_modes_gate streaming",
                "exit_code": 0,
                "executed": 1,
                "failed": 0,
                "skipped": 0,
                "fixture": {"path": "testdata/Qwen3-0.6B-Q4_K_M.gguf", "sha256": MODEL_HASHES["Qwen3 0.6B Base/Chat"]},
                "raw_log": {
                    "path": str(self.op_streaming_raw_log.resolve()),
                    "sha256": sha256_file(self.op_streaming_raw_log),
                },
                "semantic_assertions": {"status": "verified", "engine_mode": "Streaming", "speculative_ngram": True},
            },
            "grammar_constrained_json": {
                "command": "cargo test --test grammar_constrained_tool_call_test",
                "exit_code": 0,
                "executed": 1,
                "failed": 0,
                "skipped": 0,
                "fixture": {"path": "testdata/Qwen3-0.6B-Q4_K_M.gguf", "sha256": MODEL_HASHES["Qwen3 0.6B Base/Chat"]},
                "raw_log": {
                    "path": str(self.op_grammar_raw_log.resolve()),
                    "sha256": sha256_file(self.op_grammar_raw_log),
                },
                "semantic_assertions": {
                    "grammar_complete": True,
                    "serde_parse_success": True,
                    "output_json": {"city": "Tokyo"},
                },
            },
            "readiness": {
                "command": "cargo test --test e2e_unified_modes_gate readiness",
                "exit_code": 0,
                "executed": 1,
                "failed": 0,
                "skipped": 0,
                "fixture": {"path": "testdata/Qwen3-0.6B-Q4_K_M.gguf", "sha256": MODEL_HASHES["Qwen3 0.6B Base/Chat"]},
                "raw_log": {
                    "path": str(self.op_readiness_raw_log.resolve()),
                    "sha256": sha256_file(self.op_readiness_raw_log),
                },
                "semantic_assertions": {"readiness_verified": True},
            },
            "bounded_completion": {
                "command": "cargo test --test e2e_unified_modes_gate bounded",
                "exit_code": 0,
                "executed": 1,
                "failed": 0,
                "skipped": 0,
                "fixture": {"path": "testdata/Qwen3-0.6B-Q4_K_M.gguf", "sha256": MODEL_HASHES["Qwen3 0.6B Base/Chat"]},
                "raw_log": {
                    "path": str(self.op_bounded_raw_log.resolve()),
                    "sha256": sha256_file(self.op_bounded_raw_log),
                },
                "semantic_assertions": {"bounded_completion_verified": True},
            },
            "cleanup": {
                "command": "cargo test --test e2e_unified_modes_gate cleanup",
                "exit_code": 0,
                "executed": 1,
                "failed": 0,
                "skipped": 0,
                "fixture": {"path": "testdata/Qwen3-0.6B-Q4_K_M.gguf", "sha256": MODEL_HASHES["Qwen3 0.6B Base/Chat"]},
                "raw_log": {
                    "path": str(self.op_cleanup_raw_log.resolve()),
                    "sha256": sha256_file(self.op_cleanup_raw_log),
                },
                "semantic_assertions": {"cleanup_verified": True},
            },
            "negative_lifecycle": {
                "command": "cargo test --test e2e_unified_modes_gate lifecycle_tests",
                "exit_code": 0,
                "executed": 3,
                "failed": 0,
                "skipped": 0,
                "fixture": {"path": "testdata/Qwen3-0.6B-Q4_K_M.gguf", "sha256": MODEL_HASHES["Qwen3 0.6B Base/Chat"]},
                "raw_log": {
                    "path": str(self.op_negative_raw_log.resolve()),
                    "sha256": sha256_file(self.op_negative_raw_log),
                },
                "semantic_assertions": {
                    "missing_fixture_blocked": True,
                    "readiness_timeout_bounded": True,
                    "no_gpu_server_cleanup": True,
                },
            },
        }
        base = {
            "schema_version": 1,
            "status": "verified",
            "cases": cases,
        }
        base.update(overrides)
        return base

    def write_json(self, path: Path, data: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path


class TestMvpGate(MvpGateTestBase):
    """RED contract tests for mvp_gate."""

    def test_01_valid_low_ratio_yields_accepted_with_performance_advisory(self) -> None:
        """Requirement 1.1: Valid functional evidence with low ratio yields accepted-with-advisory."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark(titan_speed=20.0, llama_speed=40.0))
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())
        out_file = self.root / "out.json"

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
            output_path=out_file,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_accepted_with_performance_advisory")
        self.assertFalse(decision["promotion_authorized"])
        self.assertFalse(decision["rc_generated"])
        self.assertEqual(decision["release_approval"], "not_granted")
        self.assertEqual(decision["gate_verdicts"]["correctness"], "passed")
        self.assertEqual(decision["gate_verdicts"]["correctness_binding"], "passed")
        self.assertEqual(decision["gate_verdicts"]["operational"], "passed")
        self.assertEqual(decision["gate_verdicts"]["benchmark_structure"], "passed")
        self.assertEqual(decision["gate_verdicts"]["clean_execution_safety"], "passed")
        self.assertIn("aggregate_ratio", decision["advisory_warnings"])
        self.assertEqual(decision["blocking_failures"], [])
        self.assertTrue(out_file.is_file())

    def test_02_valid_high_ratio_yields_accepted_functional(self) -> None:
        """Valid functional evidence with >= 0.95 ratio and no advisory yields mvp_accepted_functional."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark(titan_speed=42.0, llama_speed=40.0))
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())
        out_file = self.root / "out.json"

        # Provide a passing regression baseline
        reg_file = self.write_json(self.root / "reg.json", self.make_synthetic_benchmark(titan_speed=40.0, llama_speed=40.0))

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
            regression_baseline_path=reg_file,
            output_path=out_file,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_accepted_functional")
        self.assertFalse(decision["promotion_authorized"])
        self.assertFalse(decision["rc_generated"])
        self.assertEqual(decision["blocking_failures"], [])

    def test_03_missing_or_corrupt_or_sha_mismatched_f8_blocks_correctness(self) -> None:
        """Requirement 1.2: Missing, corrupt, or SHA-mismatched F8 blocks correctness."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark())
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        # Test 1: Missing F8 file
        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=self.root / "nonexistent_f8.json",
            correctness_sha256="badhash",
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_correctness")
        self.assertIn("correctness_evidence_missing", decision["blocking_failures"])

        # Test 2: SHA mismatch
        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256="0000000000000000000000000000000000000000000000000000000000000000",
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_correctness")
        self.assertIn("correctness_evidence_invalid", decision["blocking_failures"])

        # Test 3: Unverified status in F8
        bad_f8_file = self.write_json(self.root / "bad_f8.json", self.make_synthetic_f8(status="unverified"))
        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=bad_f8_file,
            correctness_sha256=sha256_file(bad_f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_correctness")

    def test_04_f8_route_or_model_mismatch_blocks_evidence(self) -> None:
        """Requirement 1.3: F8 route or model mismatch blocks evidence."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark())
        # F8 missing one model
        bad_f8_data = self.make_synthetic_f8()
        bad_f8_data["models"] = bad_f8_data["models"][:-1]
        bad_f8_file = self.write_json(self.root / "bad_models_f8.json", bad_f8_data)
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(bad_f8_file, bench_file))
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=bad_f8_file,
            correctness_sha256=sha256_file(bad_f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertIn(decision["status"], ("mvp_blocked_correctness", "mvp_blocked_evidence"))

    def test_05_missing_or_unverifiable_build_binding_blocks_evidence(self) -> None:
        """Requirement 1.4: Missing or unverifiable F8-to-Titan-build binding blocks evidence."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark())
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        # Test 1: Missing binding file
        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=self.root / "missing_binding.json",
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_evidence")
        self.assertIn("correctness_binding_missing", decision["blocking_failures"])

        # Test 2: Binding has mismatched titan_build
        bad_binding = self.make_synthetic_binding(f8_file, bench_file)
        bad_binding["binding"]["titan_build"] = "sha256:different_build_hash"
        bad_binding_file = self.write_json(self.root / "bad_binding.json", bad_binding)
        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=bad_binding_file,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_evidence")
        self.assertIn("build_binding_mismatch", decision["blocking_failures"])

        # Test 3: Binding status unverified
        unverified_binding = self.make_synthetic_binding(f8_file, bench_file)
        unverified_binding["binding"]["status"] = "unverified"
        unverified_binding_file = self.write_json(self.root / "unverified_binding.json", unverified_binding)
        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=unverified_binding_file,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_evidence")

    def test_06_missing_or_failed_or_skipped_operational_case_blocks_operation(self) -> None:
        """Requirement 1.5: Missing, skipped, failed, or cleanup-failing case blocks operation."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark())
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))

        # Test 1: Missing operational manifest
        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=self.root / "nonexistent_op.json",
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_operational")
        self.assertIn("operational_evidence_missing", decision["blocking_failures"])

        # Test 2: Case with skipped > 0
        op_data = self.make_synthetic_operational()
        op_data["cases"]["resident_json"]["skipped"] = 1
        bad_op_file = self.write_json(self.root / "skipped_op.json", op_data)
        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=bad_op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_operational")

        # Test 3: Case with nonzero exit code
        op_data = self.make_synthetic_operational()
        op_data["cases"]["resident_json"]["exit_code"] = 101
        bad_op_file = self.write_json(self.root / "exit_op.json", op_data)
        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=bad_op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_operational")

        # Test 4: Missing required case (e.g. streaming_sse omitted)
        op_data = self.make_synthetic_operational()
        del op_data["cases"]["streaming_sse"]
        bad_op_file = self.write_json(self.root / "missing_case_op.json", op_data)
        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=bad_op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_operational")

    def test_07_malformed_grammar_semantic_output_blocks_operation(self) -> None:
        """Requirement 1.6: Malformed grammar output or schema mismatch blocks operation."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark())
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))

        # Grammar output does NOT match exactly {"city": "Tokyo"}
        op_data = self.make_synthetic_operational()
        op_data["cases"]["grammar_constrained_json"]["semantic_assertions"]["output_json"] = {"city": "Kyoto"}
        bad_op_file = self.write_json(self.root / "bad_grammar_op.json", op_data)

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=bad_op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_operational")
        self.assertIn("grammar_semantic_mismatch", decision["blocking_failures"])

    def test_08_diagnostic_telemetry_or_selector_or_graph_active_blocks_evidence(self) -> None:
        """Requirement 1.7: Telemetry contamination, selector active, or graph active blocks evidence."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        # Test 1: Telemetry contamination in benchmark
        bad_bench = self.make_synthetic_benchmark()
        bad_bench["results"][0]["runs"][0]["prompts"][0]["telemetry"] = {"some": "data"}
        bench_file = self.write_json(self.root / "telemetry_bench.json", bad_bench)
        binding_file = self.write_json(self.root / "binding1.json", self.make_synthetic_binding(f8_file, bench_file))

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_evidence")
        self.assertTrue(
            "diagnostic_telemetry_contamination" in decision["blocking_failures"]
            or "diagnostic_instrumentation" in decision["blocking_failures"]
        )

        # Test 2: Active candidate selectors
        bad_bench2 = self.make_synthetic_benchmark(
            active_candidate_selectors=["candidate_q8_ffn"],
            candidate_selectors_absent=False,
        )
        bench_file2 = self.write_json(self.root / "selector_bench.json", bad_bench2)
        binding_file2 = self.write_json(self.root / "binding2.json", self.make_synthetic_binding(f8_file, bench_file2))

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file2,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file2,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_evidence")

        # Test 3: Observed Graph active / nonzero capture count
        bad_bench3 = self.make_synthetic_benchmark()
        bad_bench3["configuration"]["observed_graph"]["capture_count"] = 1
        bench_file3 = self.write_json(self.root / "graph_bench.json", bad_bench3)
        binding_file3 = self.write_json(self.root / "binding3.json", self.make_synthetic_binding(f8_file, bench_file3))

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file3,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file3,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_evidence")
        self.assertIn("observed_graph_active", decision["blocking_failures"])

    def test_09_incomplete_benchmark_structure_or_summary_mismatch_blocks(self) -> None:
        """Requirement 1.8: Incomplete models, repetitions, tokens, or summary mismatch blocks evidence."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        # Test 1: Generated tokens is 40 instead of 41
        bad_bench = self.make_synthetic_benchmark()
        bad_bench["configuration"]["generated_tokens"] = 40
        bench_file = self.write_json(self.root / "tokens_bench.json", bad_bench)
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_evidence")

        # Test 2: Summary statistics mismatch with raw samples
        bad_bench2 = self.make_synthetic_benchmark()
        bad_bench2["results"][0]["statistics"]["cold"]["titan_decode_tok_s"]["median"] = 999.9
        bench_file2 = self.write_json(self.root / "stats_bench.json", bad_bench2)
        binding_file2 = self.write_json(self.root / "binding2.json", self.make_synthetic_binding(f8_file, bench_file2))

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file2,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file2,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_evidence")

    def test_10_unknown_failure_blocks_fail_closed(self) -> None:
        """Requirement 1.10: Unknown failure class blocks; only explicit allowlist is advisory."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark())
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        # Test advisory allowlist mapping function directly
        self.assertTrue(mvp_gate.is_advisory_failure("per_model_ratio"))
        self.assertTrue(mvp_gate.is_advisory_failure("aggregate_ratio"))
        self.assertTrue(mvp_gate.is_advisory_failure("regression_failed"))
        self.assertTrue(mvp_gate.is_advisory_failure("regression_not_evaluated"))
        self.assertTrue(mvp_gate.is_advisory_failure("regression_incomparable"))

        self.assertFalse(mvp_gate.is_advisory_failure("unknown_error_key"))
        self.assertFalse(mvp_gate.is_advisory_failure("candidate_isolation_violation"))
        self.assertFalse(mvp_gate.is_advisory_failure("diagnostic_telemetry_contamination"))

    def test_11_cli_deterministic_execution_and_reproduction_metadata(self) -> None:
        """Requirement 3 & 9: Deterministic CLI exposes required arguments, emits finite JSON, denies promotion."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark(titan_speed=20.0, llama_speed=40.0))
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())
        out_file = self.root / "cli_out.json"

        cmd = [
            sys.executable,
            "-B",
            str(TOOLS_DIR / "mvp_gate.py"),
            "--benchmark",
            str(bench_file.resolve()),
            "--correctness-evidence",
            str(f8_file.resolve()),
            "--correctness-sha256",
            sha256_file(f8_file),
            "--correctness-binding",
            str(binding_file.resolve()),
            "--operational-evidence",
            str(op_file.resolve()),
            "--output",
            str(out_file.resolve()),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(TOOLS_DIR.parent))
        self.assertEqual(proc.returncode, 0, f"Process failed: {proc.stderr}\n{proc.stdout}")
        self.assertTrue(out_file.is_file())

        with open(out_file, encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["status"], "mvp_accepted_with_performance_advisory")
        self.assertFalse(data["promotion_authorized"])
        self.assertFalse(data["rc_generated"])
        self.assertEqual(data["release_approval"], "not_granted")
        # Ensure paths in output are absolute
        for key, path_str in data["input_paths"].items():
            self.assertTrue(os.path.isabs(path_str), f"Path {key}={path_str} is not absolute")
        # Ensure all input hashes are recorded
        self.assertIn("benchmark_sha256", data["input_hashes"])
        self.assertIn("correctness_evidence_sha256", data["input_hashes"])
        self.assertIn("correctness_binding_sha256", data["input_hashes"])
        self.assertIn("operational_evidence_sha256", data["input_hashes"])

    def test_12_regression_above_five_percent_is_advisory_only(self) -> None:
        """Requirement 8: Regression drop > 5% is advisory-only and does not reject functional MVP."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        # Current titan speed 30.0, baseline 40.0 -> drop is 25% > 5%
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark(titan_speed=30.0, llama_speed=40.0))
        reg_file = self.write_json(self.root / "reg.json", self.make_synthetic_benchmark(titan_speed=40.0, llama_speed=40.0))
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
            regression_baseline_path=reg_file,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_accepted_with_performance_advisory")
        self.assertIn("regression_failed", decision["advisory_warnings"])
        self.assertIn("regression", decision["advisory_warnings"])
        self.assertEqual(decision["blocking_failures"], [])
        self.assertEqual(decision["advisories"]["regression"]["status"], "failed")

    def test_13_incompatible_regression_is_advisory_incomparable(self) -> None:
        """Requirement 8: Incompatible regression baseline is advisory incomparable and does not block."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark(titan_speed=42.0, llama_speed=40.0))
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        # Baseline with generated_tokens=30 instead of 41
        bad_reg = self.make_synthetic_benchmark(titan_speed=40.0, llama_speed=40.0)
        bad_reg["configuration"]["generated_tokens"] = 30
        reg_file = self.write_json(self.root / "incompatible_reg.json", bad_reg)

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
            regression_baseline_path=reg_file,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_accepted_with_performance_advisory")
        self.assertIn("regression_incomparable", decision["advisory_warnings"])
        self.assertEqual(decision["blocking_failures"], [])
        self.assertEqual(decision["advisories"]["regression"]["status"], "incomparable")

    def test_14_internal_passed_correctness_status_in_benchmark_is_rejected(self) -> None:
        """Requirement 4: Benchmark claiming internal correctness.status='passed' is rejected."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bad_bench = self.make_synthetic_benchmark()
        bad_bench["correctness"]["status"] = "passed"
        bench_file = self.write_json(self.root / "bench.json", bad_bench)
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
        )

        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_evidence")
        self.assertIn("benchmark_claims_passed_correctness", decision["blocking_failures"])

    def test_15_candidate_isolation_violations_block_evidence(self) -> None:
        """Requirement 6: candidate_authorization=true and production_dispatch_changed=true block evidence."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        # Test candidate_authorization=True
        bad_bench1 = self.make_synthetic_benchmark(candidate_authorization=True)
        bench_file1 = self.write_json(self.root / "bench1.json", bad_bench1)
        binding_file1 = self.write_json(self.root / "binding1.json", self.make_synthetic_binding(f8_file, bench_file1))

        decision1, exit_code1 = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file1,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file1,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code1, 0)
        self.assertEqual(decision1["status"], "mvp_blocked_evidence")

        # Test production_dispatch_changed=True
        bad_bench2 = self.make_synthetic_benchmark(production_dispatch_changed=True)
        bench_file2 = self.write_json(self.root / "bench2.json", bad_bench2)
        binding_file2 = self.write_json(self.root / "binding2.json", self.make_synthetic_binding(f8_file, bench_file2))

        decision2, exit_code2 = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file2,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file2,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code2, 0)
        self.assertEqual(decision2["status"], "mvp_blocked_evidence")

    def test_16_operational_raw_log_missing_or_hash_mismatch_blocks(self) -> None:
        """Requirement 7: Operational raw log missing or SHA mismatch blocks operation."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark())
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))

        # Test raw log file missing
        op_data1 = self.make_synthetic_operational()
        op_data1["cases"]["resident_json"]["raw_log"]["path"] = str(self.root / "nonexistent.raw.log")
        op_file1 = self.write_json(self.root / "op1.json", op_data1)

        decision1, exit_code1 = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file1,
        )
        self.assertNotEqual(exit_code1, 0)
        self.assertEqual(decision1["status"], "mvp_blocked_operational")

        # Test raw log SHA mismatch
        op_data2 = self.make_synthetic_operational()
        op_data2["cases"]["resident_json"]["raw_log"]["sha256"] = "1111111111111111111111111111111111111111111111111111111111111111"
        op_file2 = self.write_json(self.root / "op2.json", op_data2)

        decision2, exit_code2 = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file2,
        )
        self.assertNotEqual(exit_code2, 0)
        self.assertEqual(decision2["status"], "mvp_blocked_operational")

    def test_17_build_binding_placeholder_provenance_blocks(self) -> None:
        """Requirement 5: Build binding with placeholder / unreported provenance blocks evidence."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bad_bench = self.make_synthetic_benchmark()
        bad_bench["engine_identity"]["titan"]["build"] = "placeholder_build"
        bench_file = self.write_json(self.root / "bench.json", bad_bench)

        binding_data = self.make_synthetic_binding(f8_file, bench_file)
        binding_data["binding"]["titan_build"] = "placeholder_build"
        binding_file = self.write_json(self.root / "binding.json", binding_data)
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_evidence")

    def test_18_f8_raw_log_missing_blocks_correctness(self) -> None:
        """Requirement 5: F8 evidence with missing raw log blocks correctness."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        bad_f8 = self.make_synthetic_f8(raw_log_path=str(self.root / "missing_f8.raw.log"))
        f8_file = self.write_json(self.root / "f8.json", bad_f8)
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark())
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_correctness")

    def test_19_f8_route_configuration_mismatch_blocks_correctness(self) -> None:
        """Requirement 5: F8 route with graph=True or input_format='Q8' blocks correctness."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        bad_route_f8 = self.make_synthetic_f8(
            route_configuration={
                "input_format": "Q8",
                "batch_size": 1,
                "cache_condition": "resident",
                "graph": False,
            }
        )
        f8_file = self.write_json(self.root / "f8.json", bad_route_f8)
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark())
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_correctness")

    def test_20_cli_blocked_exit_code_is_nonzero(self) -> None:
        """Requirement 9: Blocked decisions return nonzero process exit code."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark())
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))
        out_file = self.root / "cli_blocked_out.json"

        cmd = [
            sys.executable,
            "-B",
            str(TOOLS_DIR / "mvp_gate.py"),
            "--benchmark",
            str(bench_file.resolve()),
            "--correctness-evidence",
            str(f8_file.resolve()),
            "--correctness-sha256",
            sha256_file(f8_file),
            "--correctness-binding",
            str(binding_file.resolve()),
            "--operational-evidence",
            str(self.root / "nonexistent_op.json"),
            "--output",
            str(out_file.resolve()),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(TOOLS_DIR.parent))
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(out_file.is_file())
        with open(out_file, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["status"], "mvp_blocked_operational")
        self.assertFalse(data["promotion_authorized"])
        self.assertFalse(data["rc_generated"])

    def test_21_missing_or_blank_correctness_sha256_blocks_correctness(self) -> None:
        """Gap 1: Missing or blank --correctness-sha256 blocks correctness in evaluate_mvp and is required in CLI."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark())
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        for blank_sha in (None, "", "   "):
            decision, exit_code = mvp_gate.evaluate_mvp(
                benchmark_path=bench_file,
                correctness_evidence_path=f8_file,
                correctness_sha256=blank_sha,
                correctness_binding_path=binding_file,
                operational_evidence_path=op_file,
            )
            self.assertNotEqual(exit_code, 0, f"Expected non-zero exit for sha256={blank_sha!r}")
            self.assertEqual(decision["status"], "mvp_blocked_correctness")
            self.assertEqual(decision["gate_verdicts"]["correctness"], "failed")
            self.assertIn("correctness_evidence_invalid", decision["blocking_failures"])

        # CLI invocation without --correctness-sha256 must fail
        out_file = self.root / "cli_missing_sha_out.json"
        cmd = [
            sys.executable,
            "-B",
            str(TOOLS_DIR / "mvp_gate.py"),
            "--benchmark",
            str(bench_file.resolve()),
            "--correctness-evidence",
            str(f8_file.resolve()),
            "--correctness-binding",
            str(binding_file.resolve()),
            "--operational-evidence",
            str(op_file.resolve()),
            "--output",
            str(out_file.resolve()),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(TOOLS_DIR.parent))
        self.assertNotEqual(proc.returncode, 0, "CLI must reject missing --correctness-sha256")

    def test_22_operational_raw_log_missing_sha256_blocks_operation(self) -> None:
        """Gap 2: Missing operational raw log sha256 blocks operation with exact failure code."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bench_file = self.write_json(self.root / "bench.json", self.make_synthetic_benchmark())
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))

        op_data = self.make_synthetic_operational()
        del op_data["cases"]["resident_json"]["raw_log"]["sha256"]
        bad_op_file = self.write_json(self.root / "missing_rl_sha_op.json", op_data)

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=bad_op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_operational")
        self.assertEqual(decision["gate_verdicts"]["operational"], "failed")
        self.assertIn("operational_case_resident_json_raw_log_sha_missing", decision["blocking_failures"])

    def test_23_model_hash_missing_blocks_evidence_and_fails_benchmark_structure(self) -> None:
        """Gap 3: Missing benchmark model_hash blocks evidence and fails benchmark_structure gate."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bad_bench = self.make_synthetic_benchmark()
        # Remove model_hash from one model
        del bad_bench["results"][0]["model_hash"]
        bench_file = self.write_json(self.root / "missing_model_hash_bench.json", bad_bench)
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_evidence")
        self.assertEqual(decision["gate_verdicts"]["benchmark_structure"], "failed")
        self.assertIn("model_hash_missing", decision["blocking_failures"])

    def test_24_observed_graph_missing_requested_blocks_evidence_and_fails_safety(self) -> None:
        """Gap 4: Observed graph omitting requested (not exactly false) blocks evidence and fails clean_execution_safety."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        bad_bench = self.make_synthetic_benchmark()
        del bad_bench["configuration"]["observed_graph"]["requested"]
        bench_file = self.write_json(self.root / "missing_requested_graph_bench.json", bad_bench)
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, bench_file))
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        decision, exit_code = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(decision["status"], "mvp_blocked_evidence")
        self.assertEqual(decision["gate_verdicts"]["clean_execution_safety"], "failed")
        self.assertIn("observed_graph_active", decision["blocking_failures"])

    def test_25_honest_gate_verdicts_reflect_subgate_failures(self) -> None:
        """Gap 5: benchmark_structure and clean_execution_safety honestly reflect subgate failures."""
        self.assertIsNotNone(mvp_gate, "mvp_gate must be importable")
        f8_file = self.write_json(self.root / "f8.json", self.make_synthetic_f8())
        op_file = self.write_json(self.root / "op.json", self.make_synthetic_operational())

        # Subtest A: Missing benchmark file -> benchmark_structure must fail
        dummy_bench = self.write_json(self.root / "dummy_bench.json", self.make_synthetic_benchmark())
        binding_file = self.write_json(self.root / "binding.json", self.make_synthetic_binding(f8_file, dummy_bench))
        decision_missing, exit_missing = mvp_gate.evaluate_mvp(
            benchmark_path=self.root / "nonexistent_bench.json",
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file,
            operational_evidence_path=op_file,
        )
        self.assertEqual(decision_missing["gate_verdicts"]["benchmark_structure"], "failed")

        # Subtest B: Model path mismatch -> benchmark_structure must fail (not pass!)
        bad_bench_path = self.make_synthetic_benchmark()
        bad_bench_path["results"][0]["model_path"] = "models/completely-different-basename.gguf"
        bench_file_path = self.write_json(self.root / "bench_path.json", bad_bench_path)
        binding_file_path = self.write_json(self.root / "binding_path.json", self.make_synthetic_binding(f8_file, bench_file_path))
        decision_path, _ = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file_path,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file_path,
            operational_evidence_path=op_file,
        )
        self.assertEqual(decision_path["gate_verdicts"]["benchmark_structure"], "failed")

        # Subtest C: Candidate isolation violation -> clean_execution_safety must fail
        bad_bench_safety = self.make_synthetic_benchmark(candidate_authorization=True)
        bench_file_safety = self.write_json(self.root / "bench_safety.json", bad_bench_safety)
        binding_file_safety = self.write_json(self.root / "binding_safety.json", self.make_synthetic_binding(f8_file, bench_file_safety))
        decision_safety, _ = mvp_gate.evaluate_mvp(
            benchmark_path=bench_file_safety,
            correctness_evidence_path=f8_file,
            correctness_sha256=sha256_file(f8_file),
            correctness_binding_path=binding_file_safety,
            operational_evidence_path=op_file,
        )
        self.assertEqual(decision_safety["gate_verdicts"]["clean_execution_safety"], "failed")


if __name__ == "__main__":
    unittest.main()
