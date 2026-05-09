#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble publication-ready Markdown tables from report JSON."
    )
    parser.add_argument("--comparison-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--ablation-json")
    args = parser.parse_args()

    comparison = _load_json(args.comparison_json)
    sections = [
        "# ProofRAG Result Tables",
        "",
        "## MiniRAG vs MiniRAG+ProofRAG",
        "",
        _comparison_table(comparison),
    ]

    if args.ablation_json:
        sections.extend(
            [
                "",
                "## Ablations",
                "",
                _ablation_table(_load_json(args.ablation_json)),
            ]
        )

    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote publication tables to {output_path}")


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _comparison_table(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    proofrag = report["proofrag"]
    paired = report["paired_answer_accuracy"]
    lines = [
        "| Method | Total | Answered | Correct | Accuracy | Precision@Answered | Unsafe Allows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _method_row(baseline),
        _method_row(proofrag),
        "",
        "| Paired Test | Value |",
        "| --- | ---: |",
        f"| Treatment win delta | {_pct(paired['treatment_win_rate_delta'])} |",
        f"| Exact p-value | {paired['exact_p_value']:.4f} |",
    ]
    return "\n".join(lines)


def _method_row(method: dict[str, Any]) -> str:
    return (
        f"| {method['method']} | {method['total']} | {method['answered']} | "
        f"{method['correct']} | {_pct(method['accuracy'])} | "
        f"{_pct(method['precision_at_answered'])} | "
        f"{method['unsafe_allow_count']} |"
    )


def _ablation_table(report: dict[str, Any]) -> str:
    lines = [
        "| Run | Total | Answered | Accuracy | Precision@Answered | Unsafe Allow Rate | Mean Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report.get("runs", []):
        lines.append(
            f"| {run['name']} | {run['total']} | {run['answered']} | "
            f"{_pct(run['accuracy'])} | {_pct(run['precision_at_answered'])} | "
            f"{_pct(run['unsafe_allow_rate'])} | {run['mean_coverage']:.2f} |"
        )
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{value:.1%}"


if __name__ == "__main__":
    main()
