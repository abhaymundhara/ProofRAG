from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.run_local_release_checks import ReleaseCheckResult, write_report


def test_release_checks_dry_run_lists_core_gates(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_local_release_checks.py",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads(result.stdout)
    names = {entry["name"] for entry in manifest}
    assert {
        "ruff",
        "mypy",
        "pytest",
        "toy_benchmark",
        "cli_hybrid_iterative_smoke",
        "reproduce_sample_artifacts",
        "package_build",
        "distribution_contents",
        "roadmap_artifact_matrix",
        "external_completion_gates",
    } <= names
    external = next(entry for entry in manifest if entry["name"] == "external_completion_gates")
    assert external["expected_exit_codes"] == [0, 1]


def test_release_checks_can_require_external_gates(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_local_release_checks.py",
            "--output-dir",
            str(tmp_path),
            "--require-external-gates",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads(result.stdout)
    external = next(entry for entry in manifest if entry["name"] == "external_completion_gates")
    assert external["required"] is True
    assert external["expected_exit_codes"] == [0]


def test_release_checks_passes_external_gate_artifact_args(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_local_release_checks.py",
            "--output-dir",
            str(tmp_path),
            "--lihua-qa-csv",
            "qa.csv",
            "--lihua-data-dir",
            "data",
            "--min-lihua-qa-rows",
            "10",
            "--minirag-export",
            "export.jsonl",
            "--min-baseline-export-rows",
            "10",
            "--comparison-summary",
            "comparison.json",
            "--faithfulness-summary",
            "faithfulness.json",
            "--review-note",
            "review.md",
            "--docker-evidence",
            "docker.txt",
            "--ci-url",
            "https://github.com/example/proofrag/actions/runs/1",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads(result.stdout)
    external = next(entry for entry in manifest if entry["name"] == "external_completion_gates")
    command = external["command"]
    assert "--lihua-qa-csv" in command
    assert "qa.csv" in command
    assert "--min-lihua-qa-rows" in command
    assert "10" in command
    assert "--minirag-export" in command
    assert "export.jsonl" in command
    assert "--docker-evidence" in command
    assert "docker.txt" in command
    assert "--ci-url" in command


def test_release_checks_passes_publication_claim_threshold_args(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_local_release_checks.py",
            "--output-dir",
            str(tmp_path),
            "--claim-comparison-summary",
            "comparison.json",
            "--claim-faithfulness-summary",
            "faithfulness.json",
            "--claim-min-total",
            "250",
            "--claim-max-accuracy-drop",
            "0.03",
            "--claim-min-precision-at-answered",
            "0.9",
            "--claim-max-unsafe-allow-rate",
            "0.0",
            "--claim-min-groundedness-delta",
            "0.2",
            "--claim-max-unsupported-claim-ratio",
            "0.5",
            "--claim-alpha",
            "0.01",
            "--require-significance",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads(result.stdout)
    claim = next(entry for entry in manifest if entry["name"] == "publication_claim_validation")
    command = claim["command"]
    assert "--comparison-summary" in command
    assert "comparison.json" in command
    assert "--min-total" in command
    assert "250" in command
    assert "--max-accuracy-drop" in command
    assert "0.03" in command
    assert "--min-precision-at-answered" in command
    assert "0.9" in command
    assert "--max-unsafe-allow-rate" in command
    assert "--min-groundedness-delta" in command
    assert "--max-unsupported-claim-ratio" in command
    assert "--alpha" in command
    assert "0.01" in command
    assert "--require-significance" in command

    external = next(entry for entry in manifest if entry["name"] == "external_completion_gates")
    external_command = external["command"]
    assert "--claim-min-total" in external_command
    assert "250" in external_command
    assert "--claim-min-groundedness-delta" in external_command
    assert "--require-claim-significance" in external_command


def test_release_report_marks_failed_results(tmp_path: Path):
    report = write_report(
        tmp_path / "release_checks.json",
        [
            ReleaseCheckResult(
                name="ruff",
                command=[sys.executable, "-m", "ruff"],
                status="passed",
                returncode=0,
            ),
            ReleaseCheckResult(
                name="pytest",
                command=[sys.executable, "-m", "pytest"],
                status="failed",
                returncode=1,
            ),
        ],
    )

    assert report["status"] == "failed"
    written = json.loads((tmp_path / "release_checks.json").read_text(encoding="utf-8"))
    assert written["results"][1]["name"] == "pytest"
