#!/usr/bin/env python3

from __future__ import annotations

import argparse

from proofrag.evaluation.lihua_augmentation import augment_export_with_lihua_sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append BM25-ranked full LiHua source rows to a normalized MiniRAG export."
    )
    parser.add_argument("--input", required=True, help="Input normalized JSONL export.")
    parser.add_argument("--output", required=True, help="Augmented JSONL export to write.")
    parser.add_argument("--data-dir", required=True, help="Extracted LiHua data directory.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of LiHua source files to append per row.")
    args = parser.parse_args()

    rows = augment_export_with_lihua_sources(
        input_path=args.input,
        output_path=args.output,
        data_dir=args.data_dir,
        top_k=args.top_k,
    )
    print(f"Augmented {rows} rows into {args.output}")


if __name__ == "__main__":
    main()
