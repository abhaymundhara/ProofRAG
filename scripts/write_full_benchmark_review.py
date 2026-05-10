#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


sys.path.append(str(Path(__file__).resolve().parents[1]))

from proofrag.evaluation.lihua import load_lihua_qa_csv, resolve_lihua_sources
from proofrag.evaluation.minirag_adapter import MiniRAGOutputAdapter
from scripts.validate_publication_claims import validate_claims


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _baseline_export_stats(path: str) -> dict[str, Any]:
    items = MiniRAGOutputAdapter().load_export(path)
    seen: set[str] = set()
    duplicates: list[str] = []
    empty_context: list[str] = []
    for item in items:
        if item.id in seen and item.id not in duplicates:
            duplicates.append(item.id)
        seen.add(item.id)
        if not any(str(context.get("text", "")).strip() for context in item.retrieved_context):
            empty_context.append(item.id)
    return {
        "rows": len(items),
        "duplicate_ids": duplicates,
        "empty_context_ids": empty_context,
    }


def _lihua_resolution(qa_csv: str | None, data_dir: str | None) -> str:
    if not qa_csv or not data_dir:
        return "not supplied"
    rows = load_lihua_qa_csv(qa_csv)
    evidence_ids = sorted(
        {
            evidence_id
            for row in rows
            for evidence_id in row.evidence_ids
            if evidence_id
        }
    )
    if not evidence_ids:
        return f"{len(rows)} QA rows; no evidence IDs found"
    resolution = resolve_lihua_sources(evidence_ids, Path(data_dir))
    ratio = len(resolution.files) / len(evidence_ids)
    return (
        f"{len(rows)} QA rows; {len(resolution.files)}/{len(evidence_ids)} "
        f"evidence IDs resolved ({_percent(ratio)})"
    )


def _claim_report(args: argparse.Namespace) -> dict[str, Any]:
    return validate_claims(
        argparse.Namespace(
            comparison_summary=args.comparison_summary,
            faithfulness_summary=args.faithfulness_summary,
            min_total=args.claim_min_total,
            max_accuracy_drop=args.claim_max_accuracy_drop,
            min_precision_at_answered=args.claim_min_precision_at_answered,
            max_unsafe_allow_rate=args.claim_max_unsafe_allow_rate,
            min_groundedness_delta=args.claim_min_groundedness_delta,
            max_unsupported_claim_ratio=args.claim_max_unsupported_claim_ratio,
            require_significance=args.require_claim_significance,
            alpha=args.claim_alpha,
        )
    )


def build_review(args: argparse.Namespace) -> str:
    comparison = _load_json(args.comparison_summary)
    faithfulness = _load_json(args.faithfulness_summary)
    export_stats = _baseline_export_stats(args.minirag_export)
    claim_report = _claim_report(args)

    baseline = comparison["baseline"]
    proofrag = comparison["proofrag"]
    paired = comparison["paired_answer_accuracy"]
    faith_summary = faithfulness["summary"]
    total = int(proofrag["total"])
    unsafe_rate = int(proofrag.get("unsafe_allow_count", 0)) / total if total else 0.0
    abstention_rate = int(proofrag.get("abstained", 0)) / total if total else 0.0
    baseline_unsupported = int(faith_summary["baseline_unsupported_claims"])
    proofrag_unsupported = int(faith_summary["proofrag_unsupported_claims"])
    unsupported_ratio = (
        proofrag_unsupported / baseline_unsupported if baseline_unsupported else 0.0
    )
    status = "approved" if claim_report["publication_claim_ready"] else "needs rerun"
    failed_checks = [
        check["name"] for check in claim_report["checks"] if not check["passed"]
    ]
    rationale = (
        "All configured publication-claim thresholds passed."
        if not failed_checks
        else "Publication-claim thresholds failed: " + ", ".join(failed_checks) + "."
    )
    duplicate_text = (
        ", ".join(export_stats["duplicate_ids"][:10])
        if export_stats["duplicate_ids"]
        else "none"
    )
    empty_context_text = (
        ", ".join(export_stats["empty_context_ids"][:10])
        if export_stats["empty_context_ids"]
        else "none"
    )

    return f"""# Full Benchmark Review

## Review Scope

- Reviewer: {args.reviewer}
- Review date: {args.review_date}
- Dataset: LiHua-World
- Benchmark split or query source: {args.benchmark_scope}
- LiHua QA CSV: {args.lihua_qa_csv or "not supplied"}
- LiHua source directory: {args.lihua_data_dir or "not supplied"}
- Normalized MiniRAG/LightRAG export: {args.minirag_export}
- ProofRAG result JSONL: {args.proofrag_results or "not supplied"}
- Comparison summary: {args.comparison_summary}
- Faithfulness summary: {args.faithfulness_summary}

## Benchmark Checks

- Total QA rows reviewed: {total}
- Rows excluded, with reasons: {args.rows_excluded}
- Evidence-ID source-resolution rate: {_lihua_resolution(args.lihua_qa_csv, args.lihua_data_dir)}
- Baseline export row count: {export_stats["rows"]}
- Duplicate row IDs found: {duplicate_text}
- Rows with empty retrieved context: {empty_context_text}

## Comparison Review

- MiniRAG accuracy: {_percent(float(baseline["accuracy"]))}
- MiniRAG+ProofRAG accuracy: {_percent(float(proofrag["accuracy"]))}
- Accuracy delta: {_percent(float(proofrag["accuracy"]) - float(baseline["accuracy"]))}
- Unsafe allow rate: {_percent(unsafe_rate)}
- Precision@Answered: {_percent(float(proofrag["precision_at_answered"]))}
- Abstention rate: {_percent(abstention_rate)}
- Latency or token-cost observations: {args.cost_notes}
- Statistical test and p-value: exact paired test p={float(paired["exact_p_value"]):.6f}

## Faithfulness Review

- MiniRAG groundedness: {_percent(float(faith_summary["baseline_mean_groundedness"]))}
- MiniRAG+ProofRAG groundedness: {_percent(float(faith_summary["proofrag_mean_groundedness"]))}
- Groundedness delta: {_percent(float(faith_summary["proofrag_mean_groundedness"]) - float(faith_summary["baseline_mean_groundedness"]))}
- MiniRAG unsupported-claim ratio: 1.000 baseline ({baseline_unsupported} unsupported claims)
- MiniRAG+ProofRAG unsupported-claim ratio: {unsupported_ratio:.3f} ({proofrag_unsupported} unsupported claims)
- LLM-judge configuration, if used: {args.judge_config}
- Manual spot-check notes: {args.spot_check_notes}

## Decision

- Claim status: {status}
- Rationale: {rationale}
- Known limitations: {args.known_limitations}
- Follow-up required before publication: {args.follow_up}

This review covers the benchmark scope plus the comparison and faithfulness
artifacts listed above.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a filled full-benchmark review note from real metric artifacts."
    )
    parser.add_argument("--comparison-summary", required=True)
    parser.add_argument("--faithfulness-summary", required=True)
    parser.add_argument("--minirag-export", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--review-date", required=True)
    parser.add_argument("--benchmark-scope", default="Full LiHua-World benchmark")
    parser.add_argument("--lihua-qa-csv")
    parser.add_argument("--lihua-data-dir")
    parser.add_argument("--proofrag-results")
    parser.add_argument("--rows-excluded", default="none")
    parser.add_argument("--cost-notes", default="not reviewed")
    parser.add_argument("--judge-config", default="deterministic claim-level scorer")
    parser.add_argument("--spot-check-notes", default="not recorded")
    parser.add_argument("--known-limitations", default="none recorded")
    parser.add_argument("--follow-up", default="none")
    parser.add_argument("--claim-min-total", type=int, default=100)
    parser.add_argument("--claim-max-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--claim-min-precision-at-answered", type=float, default=0.75)
    parser.add_argument("--claim-max-unsafe-allow-rate", type=float, default=0.0)
    parser.add_argument("--claim-min-groundedness-delta", type=float, default=0.10)
    parser.add_argument("--claim-max-unsupported-claim-ratio", type=float, default=0.75)
    parser.add_argument("--require-claim-significance", action="store_true")
    parser.add_argument("--claim-alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review = build_review(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(review, encoding="utf-8")
    print(f"Wrote full benchmark review to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
