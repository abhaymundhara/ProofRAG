from __future__ import annotations

import argparse
import subprocess
import sys

from scripts.run_full_benchmark_pipeline import build_commands


def _args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "minirag_export": "experiments/results/full_minirag_export.jsonl",
        "output_dir": "experiments/results/full_benchmark",
        "lihua_qa_csv": "data/query_set.csv",
        "lihua_data_dir": "data/source",
        "reviewer": "Test Reviewer",
        "review_date": "2026-05-10",
        "benchmark_scope": "Full LiHua-World benchmark",
        "cost_notes": "latency not collected",
        "judge_config": "deterministic claim-level scorer",
        "spot_check_notes": "spot check complete",
        "known_limitations": "none",
        "follow_up": "none",
        "docker_evidence": "evidence/docker_build.txt",
        "ci_evidence": "evidence/github_actions_success.txt",
        "ci_url": "https://github.com/example/proofrag/actions/runs/1",
        "faithfulness_scorer": "claim",
        "judge_backend": "ollama",
        "judge_model": "qwen3.5:4b",
        "judge_base_url": "http://localhost:11434",
        "judge_api_key": None,
        "judge_timeout": 120,
        "judge_temperature": 0.0,
        "judge_max_tokens": 512,
        "judge_endpoint_mode": "chat",
        "claim_min_total": 100,
        "min_baseline_export_rows": 100,
        "claim_max_accuracy_drop": 0.05,
        "claim_min_precision_at_answered": 0.75,
        "claim_max_unsafe_allow_rate": 0.0,
        "claim_min_groundedness_delta": 0.10,
        "claim_max_unsupported_claim_ratio": 0.75,
        "claim_alpha": 0.05,
        "require_claim_significance": False,
        "dry_run": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_full_benchmark_pipeline_builds_ordered_artifact_commands():
    commands = build_commands(_args())

    assert [command[1] for command in commands] == [
        "scripts/run_lihua_eval.py",
        "scripts/run_ablation.py",
        "scripts/make_publication_tables.py",
        "scripts/score_faithfulness.py",
        "scripts/write_full_benchmark_review.py",
        "scripts/write_external_evidence_manifest.py",
        "scripts/check_completion_gates.py",
    ]
    gate_command = commands[-1]
    assert "--min-baseline-export-rows" in gate_command
    assert "100" in gate_command
    assert "--docker-evidence" in gate_command
    assert "--ci-evidence" in gate_command
    assert "--review-note" in gate_command
    assert "--qa-csv" in commands[0]
    assert "--source-resolution-json" in commands[0]
    assert "--run" in commands[1]
    assert "--ablation-json" in commands[2]
    assert "--lihua-qa-csv" in commands[4]
    assert "--qa-csv" not in commands[4]


def test_full_benchmark_pipeline_dry_run_prints_commands(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_full_benchmark_pipeline.py",
            "--minirag-export",
            "benchmarks/sample_minirag_export.jsonl",
            "--output-dir",
            str(tmp_path / "full"),
            "--reviewer",
            "Test Reviewer",
            "--review-date",
            "2026-05-10",
            "--dry-run",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "scripts/run_lihua_eval.py" in result.stdout
    assert "scripts/run_ablation.py" in result.stdout
    assert "scripts/make_publication_tables.py" in result.stdout
    assert "scripts/check_completion_gates.py" in result.stdout
    assert "Full benchmark artifacts written" in result.stdout
