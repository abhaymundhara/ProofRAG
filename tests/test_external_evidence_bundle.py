from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.check_completion_gates import build_completion_report
from scripts.init_external_evidence_bundle import build_bundle


def test_external_evidence_bundle_creates_invalid_templates(tmp_path: Path):
    report = build_bundle(tmp_path)

    assert report["status"] == "created"
    review = tmp_path / "full_benchmark_review.md.template"
    docker = tmp_path / "docker_build.txt.template"
    ci = tmp_path / "github_actions_success.txt.template"
    expected = tmp_path / "expected_paths.json"
    assert review.exists()
    assert docker.exists()
    assert ci.exists()
    assert expected.exists()
    expected_paths = json.loads(expected.read_text(encoding="utf-8"))
    assert expected_paths["review_note"] == "experiments/results/full_benchmark_review.md"
    assert "NOT VALID EVIDENCE" in docker.read_text(encoding="utf-8")
    assert "OWNER/REPO" in ci.read_text(encoding="utf-8")


def test_external_evidence_bundle_does_not_overwrite_by_default(tmp_path: Path):
    build_bundle(tmp_path)
    docker = tmp_path / "docker_build.txt.template"
    docker.write_text("custom\n", encoding="utf-8")

    report = build_bundle(tmp_path)

    assert str(docker) in report["skipped"]
    assert docker.read_text(encoding="utf-8") == "custom\n"


def test_external_evidence_templates_do_not_pass_completion_gate(tmp_path: Path):
    build_bundle(tmp_path)

    report = build_completion_report(
        argparse_namespace(
            tmp_path,
            review_note=str(tmp_path / "full_benchmark_review.md.template"),
            docker_evidence=str(tmp_path / "docker_build.txt.template"),
            ci_evidence=str(tmp_path / "github_actions_success.txt.template"),
        )
    )

    blocked = {gate["name"]: gate for gate in report["gates"]}
    assert blocked["reviewed_result_artifacts"]["passed"] is False
    assert blocked["docker_build_verified"]["passed"] is False
    assert blocked["remote_ci_verified"]["passed"] is False
    assert "template placeholders" in blocked["docker_build_verified"]["detail"]
    assert "template placeholders" in blocked["remote_ci_verified"]["detail"]


def test_external_evidence_bundle_cli_writes_report(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/init_external_evidence_bundle.py",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["status"] == "created"
    assert (tmp_path / "README.md").exists()


def argparse_namespace(tmp_path: Path, **overrides: object):
    import argparse

    defaults: dict[str, object] = {
        "lihua_qa_csv": None,
        "lihua_data_dir": None,
        "min_lihua_qa_rows": 100,
        "min_lihua_source_resolution": 0.90,
        "minirag_export": None,
        "min_baseline_export_rows": 100,
        "comparison_summary": None,
        "faithfulness_summary": None,
        "review_note": None,
        "claim_min_total": 100,
        "claim_max_accuracy_drop": 0.05,
        "claim_min_precision_at_answered": 0.75,
        "claim_max_unsafe_allow_rate": 0.0,
        "claim_min_groundedness_delta": 0.10,
        "claim_max_unsupported_claim_ratio": 0.75,
        "require_claim_significance": False,
        "claim_alpha": 0.05,
        "docker_evidence": None,
        "docker_build_tag": "proofrag:test",
        "docker_build_context": ".",
        "ci_evidence": None,
        "ci_url": None,
        "check_docker_daemon": False,
        "output_json": str(tmp_path / "report.json"),
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)
