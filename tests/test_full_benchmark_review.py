from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.write_full_benchmark_review import build_review


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    comparison = tmp_path / "comparison_summary.json"
    comparison.write_text(
        json.dumps(
            {
                "baseline": {
                    "total": 3,
                    "accuracy": 0.67,
                    "precision_at_answered": 0.67,
                },
                "proofrag": {
                    "total": 3,
                    "accuracy": 0.67,
                    "precision_at_answered": 1.0,
                    "unsafe_allow_count": 0,
                    "abstained": 1,
                },
                "paired_answer_accuracy": {
                    "exact_p_value": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    faithfulness = tmp_path / "faithfulness_summary.json"
    faithfulness.write_text(
        json.dumps(
            {
                "summary": {
                    "total": 3,
                    "baseline_mean_groundedness": 0.2,
                    "proofrag_mean_groundedness": 0.5,
                    "baseline_unsupported_claims": 8,
                    "proofrag_unsupported_claims": 4,
                }
            }
        ),
        encoding="utf-8",
    )
    export = tmp_path / "minirag_export.jsonl"
    export.write_text(
        Path("benchmarks/sample_minirag_export.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return comparison, faithfulness, export


def _args(tmp_path: Path) -> argparse.Namespace:
    comparison, faithfulness, export = _write_artifacts(tmp_path)
    return argparse.Namespace(
        comparison_summary=str(comparison),
        faithfulness_summary=str(faithfulness),
        minirag_export=str(export),
        output=str(tmp_path / "review.md"),
        reviewer="Test Reviewer",
        review_date="2026-05-10",
        benchmark_scope="LiHua-World smoke validation",
        lihua_qa_csv=None,
        lihua_data_dir=None,
        proofrag_results="experiments/results/full_proofrag_results.jsonl",
        rows_excluded="none",
        cost_notes="token counts not collected",
        judge_config="deterministic claim-level scorer",
        spot_check_notes="three rows checked for citations",
        known_limitations="smoke fixture only",
        follow_up="run full LiHua benchmark",
        claim_min_total=3,
        claim_max_accuracy_drop=0.05,
        claim_min_precision_at_answered=0.75,
        claim_max_unsafe_allow_rate=0.0,
        claim_min_groundedness_delta=0.10,
        claim_max_unsupported_claim_ratio=0.75,
        require_claim_significance=False,
        claim_alpha=0.05,
    )


def test_build_full_benchmark_review_from_artifacts(tmp_path: Path):
    review = build_review(_args(tmp_path))

    assert "Review Scope" in review
    assert "benchmark scope plus the comparison and faithfulness" in review
    assert "MiniRAG accuracy: 67.0%" in review
    assert "MiniRAG+ProofRAG groundedness: 50.0%" in review
    assert "Claim status: approved" in review
    assert "Duplicate row IDs found: none" in review
    assert "Rows with empty retrieved context: none" in review


def test_build_full_benchmark_review_resolves_lihua_sources(tmp_path: Path):
    args = _args(tmp_path)
    qa_csv = tmp_path / "query_set.csv"
    qa_csv.write_text(
        "Question,Gold Answer,Evidence,Type\n"
        "When did Li Hua check in?,5:30 PM,20260105_14:00,Single\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "LiHua-World"
    data_dir.mkdir()
    (data_dir / "20260105_1400.txt").write_text(
        "Time: 20260105_14:00\nLiHua checked in at 5:30 PM.\n",
        encoding="utf-8",
    )
    args.lihua_qa_csv = str(qa_csv)
    args.lihua_data_dir = str(data_dir)

    review = build_review(args)

    assert "Evidence-ID source-resolution rate: 1 QA rows; 1/1 evidence IDs resolved (100.0%)" in review


def test_full_benchmark_review_cli_writes_gate_valid_note(tmp_path: Path):
    args = _args(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/write_full_benchmark_review.py",
            "--comparison-summary",
            args.comparison_summary,
            "--faithfulness-summary",
            args.faithfulness_summary,
            "--minirag-export",
            args.minirag_export,
            "--output",
            args.output,
            "--reviewer",
            args.reviewer,
            "--review-date",
            args.review_date,
            "--benchmark-scope",
            args.benchmark_scope,
            "--claim-min-total",
            "3",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    review = Path(args.output).read_text(encoding="utf-8")
    assert "review covers the benchmark scope" in review.lower()
    assert "comparison and faithfulness" in review.lower()
