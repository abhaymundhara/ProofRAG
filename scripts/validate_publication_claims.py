#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClaimCheck:
    name: str
    passed: bool
    observed: float | int | str
    threshold: float | int | str
    detail: str


def _load_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Required artifact not found: {candidate}")
    return json.loads(candidate.read_text(encoding="utf-8"))


def _blocked_artifact_report(error: Exception) -> dict[str, Any]:
    return {
        "publication_claim_ready": False,
        "status": "blocked",
        "checks": [
            asdict(
                ClaimCheck(
                    name="required_artifacts",
                    passed=False,
                    observed=str(error),
                    threshold="comparison and faithfulness JSON summaries must exist",
                    detail="Publication claims cannot be validated without both input artifacts.",
                )
            )
        ],
        "metrics": {},
    }


def _unsafe_allow_rate(report: dict[str, Any]) -> float:
    proofrag = report["proofrag"]
    total = int(proofrag.get("total", 0))
    unsafe = int(proofrag.get("unsafe_allow_count", 0))
    return unsafe / total if total else 0.0


def _unsupported_ratio(faithfulness: dict[str, Any]) -> float:
    summary = faithfulness["summary"]
    baseline = int(summary.get("baseline_unsupported_claims", 0))
    proofrag = int(summary.get("proofrag_unsupported_claims", 0))
    if baseline == 0:
        return 0.0 if proofrag == 0 else float("inf")
    return proofrag / baseline


def validate_claims(args: argparse.Namespace) -> dict[str, Any]:
    comparison = _load_json(args.comparison_summary)
    faithfulness = _load_json(args.faithfulness_summary)

    baseline = comparison["baseline"]
    proofrag = comparison["proofrag"]
    paired = comparison["paired_answer_accuracy"]
    faith_summary = faithfulness["summary"]

    total = int(proofrag.get("total", 0))
    baseline_total = int(baseline.get("total", 0))
    faith_total = int(faith_summary.get("total", 0))
    baseline_accuracy = float(baseline.get("accuracy", 0.0))
    proofrag_accuracy = float(proofrag.get("accuracy", 0.0))
    precision_at_answered = float(proofrag.get("precision_at_answered", 0.0))
    unsafe_rate = _unsafe_allow_rate(comparison)
    baseline_groundedness = float(faith_summary.get("baseline_mean_groundedness", 0.0))
    proofrag_groundedness = float(faith_summary.get("proofrag_mean_groundedness", 0.0))
    groundedness_delta = proofrag_groundedness - baseline_groundedness
    unsupported_ratio = _unsupported_ratio(faithfulness)
    exact_p_value = float(paired.get("exact_p_value", 1.0))

    checks = [
        ClaimCheck(
            name="sample_size",
            passed=total >= args.min_total
            and baseline_total == total
            and faith_total == total,
            observed=f"comparison={total}, baseline={baseline_total}, faithfulness={faith_total}",
            threshold=args.min_total,
            detail="Comparison and faithfulness artifacts must cover the same minimum sample.",
        ),
        ClaimCheck(
            name="accuracy_retention",
            passed=proofrag_accuracy >= baseline_accuracy - args.max_accuracy_drop,
            observed=round(proofrag_accuracy - baseline_accuracy, 6),
            threshold=f">= -{args.max_accuracy_drop}",
            detail="ProofRAG must remain accuracy-competitive with the baseline.",
        ),
        ClaimCheck(
            name="precision_at_answered",
            passed=precision_at_answered >= args.min_precision_at_answered,
            observed=round(precision_at_answered, 6),
            threshold=args.min_precision_at_answered,
            detail="Allowed answers must stay highly precise.",
        ),
        ClaimCheck(
            name="unsafe_allow_rate",
            passed=unsafe_rate <= args.max_unsafe_allow_rate,
            observed=round(unsafe_rate, 6),
            threshold=args.max_unsafe_allow_rate,
            detail="Safety claims require bounded unsafe allowed answers.",
        ),
        ClaimCheck(
            name="groundedness_delta",
            passed=groundedness_delta >= args.min_groundedness_delta,
            observed=round(groundedness_delta, 6),
            threshold=args.min_groundedness_delta,
            detail="Faithfulness claims require a minimum groundedness lift.",
        ),
        ClaimCheck(
            name="unsupported_claim_reduction",
            passed=unsupported_ratio <= args.max_unsupported_claim_ratio,
            observed=(
                "inf" if unsupported_ratio == float("inf") else round(unsupported_ratio, 6)
            ),
            threshold=args.max_unsupported_claim_ratio,
            detail="Unsupported claims must drop by the configured ratio.",
        ),
    ]

    if args.require_significance:
        checks.append(
            ClaimCheck(
                name="paired_significance",
                passed=exact_p_value <= args.alpha,
                observed=round(exact_p_value, 6),
                threshold=args.alpha,
                detail="Accuracy comparison must pass the configured paired exact test.",
            )
        )

    passed = all(check.passed for check in checks)
    return {
        "publication_claim_ready": passed,
        "status": "ready" if passed else "blocked",
        "checks": [asdict(check) for check in checks],
        "metrics": {
            "baseline_accuracy": baseline_accuracy,
            "proofrag_accuracy": proofrag_accuracy,
            "precision_at_answered": precision_at_answered,
            "unsafe_allow_rate": unsafe_rate,
            "baseline_groundedness": baseline_groundedness,
            "proofrag_groundedness": proofrag_groundedness,
            "groundedness_delta": groundedness_delta,
            "unsupported_claim_ratio": unsupported_ratio,
            "exact_p_value": exact_p_value,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether full benchmark artifacts justify ProofRAG "
            "publication/superiority claims."
        )
    )
    parser.add_argument("--comparison-summary", required=True)
    parser.add_argument("--faithfulness-summary", required=True)
    parser.add_argument("--output-json", help="Optional path to write the validation report.")
    parser.add_argument("--min-total", type=int, default=100)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--min-precision-at-answered", type=float, default=0.75)
    parser.add_argument("--max-unsafe-allow-rate", type=float, default=0.0)
    parser.add_argument("--min-groundedness-delta", type=float, default=0.10)
    parser.add_argument("--max-unsupported-claim-ratio", type=float, default=0.75)
    parser.add_argument("--require-significance", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate_claims(args)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        report = _blocked_artifact_report(exc)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(f"{payload}\n", encoding="utf-8")
    print(payload)
    return 0 if report["publication_claim_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
