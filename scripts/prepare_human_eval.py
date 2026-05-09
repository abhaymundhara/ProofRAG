#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from proofrag.human_eval.schema import export_human_eval_jsonl, load_result_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ProofRAG result JSONL rows for human evaluation."
    )
    parser.add_argument("--input", required=True, help="Input experiment JSONL.")
    parser.add_argument("--output", required=True, help="Output human-eval JSONL.")
    args = parser.parse_args()

    rows = load_result_jsonl(Path(args.input))
    items = export_human_eval_jsonl(rows, Path(args.output))
    print(f"Wrote {len(items)} human-eval items to {args.output}")


if __name__ == "__main__":
    main()

