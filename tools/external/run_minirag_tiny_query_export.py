#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[2]))

from tools.external.minirag_exporter import run_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query an indexed MiniRAG tiny LiHua workspace and write ProofRAG's "
            "normalized MiniRAG JSONL export."
        )
    )
    parser.add_argument(
        "--qa-file",
        default="experiments/results/minirag_tiny_single_qa_subset.csv",
        help="MiniRAG/LiHua QA CSV or normalized JSONL question file.",
    )
    parser.add_argument(
        "--output",
        default="experiments/results/minirag_tiny_real_export.jsonl",
        help="Normalized MiniRAG JSONL export to write.",
    )
    parser.add_argument(
        "--minirag-dir",
        default="../external/MiniRAG",
        help="Path to the external MiniRAG repository.",
    )
    parser.add_argument(
        "--working-dir",
        default="experiments/minirag_tiny_sources/index",
        help="MiniRAG working directory containing the prebuilt index stores.",
    )
    parser.add_argument(
        "--mode",
        choices=["naive", "mini"],
        default="mini",
        help="MiniRAG query mode.",
    )
    parser.add_argument("--limit", type=int, help="Maximum number of questions to export.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write schema-valid synthetic rows without importing MiniRAG.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = run_export(
        minirag_root=args.minirag_dir,
        working_dir=args.working_dir,
        qa_file=args.qa_file,
        output_file=args.output,
        dry_run=args.dry_run,
        limit=args.limit,
        mode=args.mode,
    )
    print(f"Export complete. Written {count} items to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
