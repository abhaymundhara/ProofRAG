#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proofrag.evaluation.comparison import compare_minirag_vs_proofrag, load_jsonl
from proofrag.evaluation.lihua import load_lihua_qa_csv, resolve_lihua_sources
from proofrag.evaluation.minirag_experiment import MiniRAGExperimentRunner
from proofrag.evaluation.plots import write_bar_chart_svg
from proofrag.evaluation.tables import comparison_markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ProofRAG gating over a normalized MiniRAG LiHua export and "
            "write reproducible evaluation artifacts."
        )
    )
    parser.add_argument("--minirag-export", required=True, help="MiniRAG JSONL export.")
    parser.add_argument(
        "--output",
        required=True,
        help="ProofRAG-over-MiniRAG result JSONL to write.",
    )
    parser.add_argument("--summary-json", required=True, help="Comparison JSON output.")
    parser.add_argument("--table-md", required=True, help="Comparison Markdown output.")
    parser.add_argument("--chart-svg", required=True, help="Comparison SVG chart output.")
    parser.add_argument(
        "--qa-csv",
        help="Optional LiHua QA CSV used to check evidence source resolution.",
    )
    parser.add_argument(
        "--data-dir",
        help="Optional extracted LiHua data directory for source resolution checks.",
    )
    parser.add_argument(
        "--source-resolution-json",
        help="Optional JSON output for LiHua evidence source resolution coverage.",
    )
    args = parser.parse_args()

    results_path = Path(args.output)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    runner = MiniRAGExperimentRunner()
    results = runner.run_experiment(args.minirag_export, str(results_path))
    rows = load_jsonl(results_path)
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
        title="LiHua MiniRAG vs MiniRAG+ProofRAG",
    )

    if args.qa_csv or args.data_dir or args.source_resolution_json:
        if not (args.qa_csv and args.data_dir and args.source_resolution_json):
            raise SystemExit(
                "--qa-csv, --data-dir, and --source-resolution-json must be "
                "provided together."
            )
        _write_source_resolution_summary(
            qa_csv=args.qa_csv,
            data_dir=args.data_dir,
            output_path=args.source_resolution_json,
        )

    print(f"Processed {len(results)} LiHua/MiniRAG rows")
    print(f"Wrote ProofRAG results to {results_path}")
    print(f"Wrote comparison summary to {summary_path}")
    print(f"Wrote comparison table to {table_path}")
    print(f"Wrote comparison chart to {args.chart_svg}")


def _write_source_resolution_summary(
    *,
    qa_csv: str,
    data_dir: str,
    output_path: str,
) -> None:
    questions = load_lihua_qa_csv(qa_csv)
    evidence_ids = sorted({eid for question in questions for eid in question.evidence_ids})
    resolution = resolve_lihua_sources(evidence_ids=evidence_ids, data_dir=data_dir)
    summary = {
        "questions": len(questions),
        "unique_evidence_ids": len(evidence_ids),
        "resolved_sources": len(resolution.files),
        "missing_sources": len(resolution.missing_source_ids),
        "missing_source_ids": resolution.missing_source_ids,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
