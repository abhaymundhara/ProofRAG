#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from proofrag.evaluation.comparison import load_jsonl
from proofrag.evaluation.plots import write_bar_chart_svg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize multiple ProofRAG ablation/result JSONL files."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named JSONL run. May be passed multiple times.",
    )
    parser.add_argument("--summary-json", required=True, help="Ablation JSON output.")
    parser.add_argument("--table-md", required=True, help="Ablation Markdown output.")
    parser.add_argument("--chart-svg", required=True, help="Ablation SVG chart output.")
    args = parser.parse_args()

    summaries = [_summarize_run(spec) for spec in args.run]
    summary = {"runs": summaries}

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    table_path = Path(args.table_md)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_ablation_markdown(summaries) + "\n", encoding="utf-8")

    write_bar_chart_svg(
        args.chart_svg,
        {run["name"]: run["precision_at_answered"] for run in summaries},
        title="ProofRAG Ablation Precision@Answered",
    )

    print(f"Wrote ablation summary to {summary_path}")
    print(f"Wrote ablation table to {table_path}")
    print(f"Wrote ablation chart to {args.chart_svg}")


def _summarize_run(spec: str) -> dict[str, Any]:
    if "=" not in spec:
        raise SystemExit("--run must use NAME=PATH")
    name, raw_path = spec.split("=", 1)
    if not name:
        raise SystemExit("--run name cannot be empty")
    rows = load_jsonl(raw_path)
    total = len(rows)
    answered = sum(1 for row in rows if _answered(row))
    correct = sum(1 for row in rows if _correct_when_answered(row))
    unsafe = sum(
        1
        for row in rows
        if bool(row.get("answer_allowed", row.get("actual_answer_allowed", False)))
        and row.get("expected_answer_allowed") is False
    )
    coverage_values = [
        float(row["coverage_score"])
        for row in rows
        if row.get("coverage_score") is not None
    ]
    return {
        "name": name,
        "path": raw_path,
        "total": total,
        "answered": answered,
        "abstained": total - answered,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "precision_at_answered": correct / answered if answered else 0.0,
        "unsafe_allow_count": unsafe,
        "unsafe_allow_rate": unsafe / total if total else 0.0,
        "mean_coverage": (
            sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
        ),
    }


def _answered(row: dict[str, Any]) -> bool:
    if "model_called" in row:
        return bool(row["model_called"])
    return bool(row.get("answer_allowed", row.get("actual_answer_allowed", False)))


def _correct_when_answered(row: dict[str, Any]) -> bool:
    if not _answered(row):
        return False
    if "correct_when_answered" in row:
        return bool(row["correct_when_answered"])
    if "proofrag_correct" in row:
        return bool(row["proofrag_correct"])
    if "answer_correct" in row:
        return bool(row["answer_correct"])
    return bool(row.get("expected_answer_allowed", False))


def _ablation_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| Run | Total | Answered | Abstained | Accuracy | Precision@Answered | Unsafe Allows | Mean Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in summaries:
        lines.append(
            "| {name} | {total} | {answered} | {abstained} | {accuracy:.1%} | "
            "{precision_at_answered:.1%} | {unsafe_allow_count} | "
            "{mean_coverage:.2f} |".format(**run)
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
