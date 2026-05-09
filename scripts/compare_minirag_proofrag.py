#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from proofrag.evaluation.comparison import (
    compare_minirag_vs_proofrag,
    load_jsonl,
)
from proofrag.evaluation.plots import write_bar_chart_svg
from proofrag.evaluation.tables import comparison_markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MiniRAG vs MiniRAG+ProofRAG comparison artifacts."
    )
    parser.add_argument("--input", required=True, help="Input result JSONL.")
    parser.add_argument("--summary-json", required=True, help="Output summary JSON.")
    parser.add_argument("--table-md", required=True, help="Output Markdown table.")
    parser.add_argument("--chart-svg", required=True, help="Output SVG chart.")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    report = compare_minirag_vs_proofrag(rows)

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    table_path = Path(args.table_md)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(comparison_markdown(report) + "\n", encoding="utf-8")

    write_bar_chart_svg(
        args.chart_svg,
        {
            "MiniRAG accuracy": report.baseline.accuracy,
            "ProofRAG accuracy": report.proofrag.accuracy,
            "ProofRAG precision@answered": report.proofrag.precision_at_answered,
            "ProofRAG unsafe allow rate": (
                report.proofrag.unsafe_allow_count / report.proofrag.total
                if report.proofrag.total
                else 0.0
            ),
        },
        title="MiniRAG vs MiniRAG+ProofRAG",
    )

    print(f"Wrote comparison summary to {summary_path}")
    print(f"Wrote comparison table to {table_path}")
    print(f"Wrote comparison chart to {args.chart_svg}")


if __name__ == "__main__":
    main()
