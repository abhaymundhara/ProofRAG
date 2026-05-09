from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.write_external_evidence_manifest import build_manifest


def _args(**overrides: object) -> argparse.Namespace:
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
        "docker_evidence": None,
        "ci_evidence": None,
        "ci_url": None,
        "check_docker_build": False,
        "docker_build_tag": "proofrag:completion-gate",
        "docker_build_context": ".",
        "claim_min_total": 100,
        "claim_max_accuracy_drop": 0.05,
        "claim_min_precision_at_answered": 0.75,
        "claim_max_unsafe_allow_rate": 0.0,
        "claim_min_groundedness_delta": 0.10,
        "claim_max_unsupported_claim_ratio": 0.75,
        "claim_alpha": 0.05,
        "require_claim_significance": False,
        "output_dir": "experiments/results/full_release_checks",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_manifest_marks_missing_evidence_without_claiming_readiness():
    manifest = build_manifest(_args())

    assert manifest["status"] == "missing_evidence"
    assert all(item["status"] == "missing" for item in manifest["requirements"])
    assert "scripts/check_completion_gates.py" in manifest["commands"]["completion_gate"]
    assert "--require-external-gates" in manifest["commands"]["full_release"]


def test_manifest_includes_all_supplied_external_paths():
    manifest = build_manifest(
        _args(
            lihua_qa_csv="data/query_set.csv",
            lihua_data_dir="data/source",
            minirag_export="exports/minirag.jsonl",
            comparison_summary="results/comparison_summary.json",
            faithfulness_summary="results/faithfulness_summary.json",
            review_note="results/review.md",
            docker_evidence="results/docker.txt",
            ci_evidence="results/ci.txt",
            ci_url="https://github.com/example/proofrag/actions/runs/123",
        )
    )

    assert manifest["status"] == "ready_to_validate"
    gate_shell = manifest["commands"]["completion_gate_shell"]
    release_shell = manifest["commands"]["full_release_shell"]
    assert "--lihua-qa-csv data/query_set.csv" in gate_shell
    assert "--min-lihua-source-resolution 0.9" in gate_shell
    assert "--ci-evidence results/ci.txt" in gate_shell
    assert "--ci-url https://github.com/example/proofrag/actions/runs/123" in gate_shell
    assert "--claim-min-total 100" in gate_shell
    assert "--claim-max-unsafe-allow-rate 0.0" in gate_shell
    assert "--claim-min-groundedness-delta 0.1" in gate_shell
    assert "--claim-max-unsupported-claim-ratio 0.75" in gate_shell
    assert "--claim-comparison-summary results/comparison_summary.json" in release_shell
    assert "--require-significance" in release_shell


def test_manifest_treats_ci_url_without_evidence_as_missing():
    manifest = build_manifest(
        _args(ci_url="https://github.com/example/proofrag/actions/runs/123")
    )

    ci_requirement = next(
        item for item in manifest["requirements"] if item["name"] == "remote_ci_verified"
    )
    assert manifest["status"] == "missing_evidence"
    assert ci_requirement["status"] == "missing"


def test_manifest_cli_writes_json_and_markdown(tmp_path: Path):
    output_json = tmp_path / "manifest.json"
    output_md = tmp_path / "manifest.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/write_external_evidence_manifest.py",
            "--lihua-qa-csv",
            "data/query_set.csv",
            "--lihua-data-dir",
            "data/source",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    manifest = json.loads(output_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "missing_evidence"
    markdown = output_md.read_text(encoding="utf-8")
    assert "ProofRAG External Evidence Manifest" in markdown
    assert "full_lihua_world_data" in markdown
