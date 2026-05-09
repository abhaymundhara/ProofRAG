#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from proofrag.evaluation.report import (
    experiment_summary_markdown,
    load_experiment_log,
    summarize_experiment_log,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a ProofRAG ExperimentLogger JSONL file."
    )
    parser.add_argument("--input", required=True, help="Input experiment JSONL log.")
    parser.add_argument("--summary-json", required=True, help="Output summary JSON.")
    parser.add_argument("--table-md", required=True, help="Output Markdown summary.")
    args = parser.parse_args()

    summary = summarize_experiment_log(load_experiment_log(args.input))
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")

    table_path = Path(args.table_md)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(experiment_summary_markdown(summary) + "\n", encoding="utf-8")

    print(f"Wrote experiment summary JSON to {summary_path}")
    print(f"Wrote experiment summary Markdown to {table_path}")


if __name__ == "__main__":
    main()
