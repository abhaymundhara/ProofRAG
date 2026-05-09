from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.validate_publication_claims import validate_claims


def _comparison(
    *,
    total: int = 100,
    baseline_accuracy: float = 0.70,
    proofrag_accuracy: float = 0.72,
    precision_at_answered: float = 0.90,
    unsafe_allow_count: int = 0,
    p_value: float = 0.01,
) -> dict[str, object]:
    return {
        "baseline": {
            "method": "minirag",
            "total": total,
            "answered": total,
            "correct": int(total * baseline_accuracy),
            "abstained": 0,
            "accuracy": baseline_accuracy,
            "precision_at_answered": baseline_accuracy,
            "unsafe_allow_count": 0,
        },
        "proofrag": {
            "method": "minirag+proofrag",
            "total": total,
            "answered": int(total * 0.8),
            "correct": int(total * proofrag_accuracy),
            "abstained": total - int(total * 0.8),
            "accuracy": proofrag_accuracy,
            "precision_at_answered": precision_at_answered,
            "unsafe_allow_count": unsafe_allow_count,
        },
        "paired_answer_accuracy": {
            "baseline_wins": 8,
            "treatment_wins": 20,
            "ties": total - 28,
            "total": total,
            "treatment_win_rate_delta": proofrag_accuracy - baseline_accuracy,
            "exact_p_value": p_value,
        },
        "proofrag_error_analysis": {"total": 0, "buckets": []},
    }


def _faithfulness(
    *,
    total: int = 100,
    baseline_groundedness: float = 0.50,
    proofrag_groundedness: float = 0.72,
    baseline_unsupported: int = 100,
    proofrag_unsupported: int = 40,
) -> dict[str, object]:
    return {
        "summary": {
            "total": total,
            "baseline_mean_groundedness": baseline_groundedness,
            "proofrag_mean_groundedness": proofrag_groundedness,
            "baseline_unsupported_claims": baseline_unsupported,
            "proofrag_unsupported_claims": proofrag_unsupported,
        },
        "rows": [],
    }


def _write_artifacts(
    tmp_path: Path,
    comparison: dict[str, object],
    faithfulness: dict[str, object],
) -> tuple[Path, Path]:
    comparison_path = tmp_path / "comparison.json"
    faithfulness_path = tmp_path / "faithfulness.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    faithfulness_path.write_text(json.dumps(faithfulness), encoding="utf-8")
    return comparison_path, faithfulness_path


def _args(comparison_path: Path, faithfulness_path: Path, **overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "comparison_summary": str(comparison_path),
        "faithfulness_summary": str(faithfulness_path),
        "output_json": None,
        "min_total": 100,
        "max_accuracy_drop": 0.05,
        "min_precision_at_answered": 0.75,
        "max_unsafe_allow_rate": 0.0,
        "min_groundedness_delta": 0.10,
        "max_unsupported_claim_ratio": 0.75,
        "require_significance": True,
        "alpha": 0.05,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_publication_claim_validation_passes_for_strong_artifacts(tmp_path: Path):
    comparison_path, faithfulness_path = _write_artifacts(
        tmp_path,
        _comparison(),
        _faithfulness(),
    )

    report = validate_claims(_args(comparison_path, faithfulness_path))

    assert report["publication_claim_ready"] is True
    assert report["status"] == "ready"
    assert all(check["passed"] for check in report["checks"])


def test_publication_claim_validation_blocks_smoke_sized_artifacts(tmp_path: Path):
    comparison_path, faithfulness_path = _write_artifacts(
        tmp_path,
        _comparison(total=10, p_value=0.5),
        _faithfulness(total=10, baseline_groundedness=0.014, proofrag_groundedness=0.271),
    )

    report = validate_claims(_args(comparison_path, faithfulness_path))

    assert report["publication_claim_ready"] is False
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "sample_size" in failed
    assert "paired_significance" in failed


def test_publication_claim_cli_writes_blocked_report(tmp_path: Path):
    comparison_path, faithfulness_path = _write_artifacts(
        tmp_path,
        _comparison(unsafe_allow_count=3),
        _faithfulness(proofrag_groundedness=0.52, proofrag_unsupported=90),
    )
    output_path = tmp_path / "claim_report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_publication_claims.py",
            "--comparison-summary",
            str(comparison_path),
            "--faithfulness-summary",
            str(faithfulness_path),
            "--require-significance",
            "--output-json",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert {"unsafe_allow_rate", "groundedness_delta", "unsupported_claim_reduction"} <= failed


def test_publication_claim_cli_fails_closed_for_missing_artifacts(tmp_path: Path):
    output_path = tmp_path / "missing_report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_publication_claims.py",
            "--comparison-summary",
            str(tmp_path / "missing_comparison.json"),
            "--faithfulness-summary",
            str(tmp_path / "missing_faithfulness.json"),
            "--output-json",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["checks"][0]["name"] == "required_artifacts"
