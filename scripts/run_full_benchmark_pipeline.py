#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _extend(command: list[str], flag: str, value: str | int | float | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    output_dir = Path(args.output_dir)
    proofrag_results = output_dir / "full_proofrag_results.jsonl"
    comparison_json = output_dir / "full_comparison_summary.json"
    comparison_md = output_dir / "full_comparison_table.md"
    comparison_svg = output_dir / "full_comparison_chart.svg"
    source_resolution_json = output_dir / "full_source_resolution.json"
    faithfulness_json = output_dir / "full_faithfulness_summary.json"
    faithfulness_md = output_dir / "full_faithfulness_table.md"
    review_note = output_dir / "full_benchmark_review.md"
    manifest_json = output_dir / "external_evidence_manifest.json"
    manifest_md = output_dir / "external_evidence_manifest.md"
    gates_json = output_dir / "completion_gates.json"

    commands: list[list[str]] = [
        [
            sys.executable,
            "scripts/run_lihua_eval.py",
            "--minirag-export",
            args.minirag_export,
            "--output",
            str(proofrag_results),
            "--summary-json",
            str(comparison_json),
            "--table-md",
            str(comparison_md),
            "--chart-svg",
            str(comparison_svg),
        ],
        [
            sys.executable,
            "scripts/score_faithfulness.py",
            "--results",
            str(proofrag_results),
            "--minirag-export",
            args.minirag_export,
            "--summary-json",
            str(faithfulness_json),
            "--table-md",
            str(faithfulness_md),
            "--scorer",
            args.faithfulness_scorer,
        ],
        [
            sys.executable,
            "scripts/write_full_benchmark_review.py",
            "--comparison-summary",
            str(comparison_json),
            "--faithfulness-summary",
            str(faithfulness_json),
            "--minirag-export",
            args.minirag_export,
            "--output",
            str(review_note),
            "--reviewer",
            args.reviewer,
            "--review-date",
            args.review_date,
            "--benchmark-scope",
            args.benchmark_scope,
            "--proofrag-results",
            str(proofrag_results),
            "--cost-notes",
            args.cost_notes,
            "--judge-config",
            args.judge_config,
            "--spot-check-notes",
            args.spot_check_notes,
            "--known-limitations",
            args.known_limitations,
            "--follow-up",
            args.follow_up,
        ],
        [
            sys.executable,
            "scripts/write_external_evidence_manifest.py",
            "--min-baseline-export-rows",
            str(args.min_baseline_export_rows),
            "--minirag-export",
            args.minirag_export,
            "--comparison-summary",
            str(comparison_json),
            "--faithfulness-summary",
            str(faithfulness_json),
            "--review-note",
            str(review_note),
            "--output-json",
            str(manifest_json),
            "--output-md",
            str(manifest_md),
        ],
        [
            sys.executable,
            "scripts/check_completion_gates.py",
            "--min-baseline-export-rows",
            str(args.min_baseline_export_rows),
            "--minirag-export",
            args.minirag_export,
            "--comparison-summary",
            str(comparison_json),
            "--faithfulness-summary",
            str(faithfulness_json),
            "--review-note",
            str(review_note),
            "--output-json",
            str(gates_json),
        ],
    ]

    _append_run_lihua_args(commands[0], args, source_resolution_json)
    for command in commands[2:]:
        _append_lihua_evidence_args(command, args)
    for command in commands[2:]:
        _append_claim_args(command, args)
    _append_evidence_args(commands[3], args)
    _append_evidence_args(commands[4], args)
    if args.faithfulness_scorer == "llm-judge":
        _append_judge_args(commands[1], args)
    return commands


def _append_run_lihua_args(
    command: list[str],
    args: argparse.Namespace,
    source_resolution_json: Path,
) -> None:
    _extend(command, "--qa-csv", args.lihua_qa_csv)
    _extend(command, "--data-dir", args.lihua_data_dir)
    if args.lihua_qa_csv and args.lihua_data_dir:
        command.extend(["--source-resolution-json", str(source_resolution_json)])


def _append_lihua_evidence_args(command: list[str], args: argparse.Namespace) -> None:
    _extend(command, "--lihua-qa-csv", args.lihua_qa_csv)
    _extend(command, "--lihua-data-dir", args.lihua_data_dir)


def _append_claim_args(command: list[str], args: argparse.Namespace) -> None:
    for flag, value in [
        ("--claim-min-total", args.claim_min_total),
        ("--claim-max-accuracy-drop", args.claim_max_accuracy_drop),
        ("--claim-min-precision-at-answered", args.claim_min_precision_at_answered),
        ("--claim-max-unsafe-allow-rate", args.claim_max_unsafe_allow_rate),
        ("--claim-min-groundedness-delta", args.claim_min_groundedness_delta),
        ("--claim-max-unsupported-claim-ratio", args.claim_max_unsupported_claim_ratio),
        ("--claim-alpha", args.claim_alpha),
    ]:
        _extend(command, flag, value)
    if args.require_claim_significance:
        command.append("--require-claim-significance")


def _append_evidence_args(command: list[str], args: argparse.Namespace) -> None:
    for flag, value in [
        ("--docker-evidence", args.docker_evidence),
        ("--ci-evidence", args.ci_evidence),
        ("--ci-url", args.ci_url),
    ]:
        _extend(command, flag, value)


def _append_judge_args(command: list[str], args: argparse.Namespace) -> None:
    for flag, value in [
        ("--judge-backend", args.judge_backend),
        ("--judge-model", args.judge_model),
        ("--judge-base-url", args.judge_base_url),
        ("--judge-api-key", args.judge_api_key),
        ("--judge-timeout", args.judge_timeout),
        ("--judge-temperature", args.judge_temperature),
        ("--judge-max-tokens", args.judge_max_tokens),
        ("--judge-endpoint-mode", args.judge_endpoint_mode),
    ]:
        _extend(command, flag, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full external MiniRAG/LightRAG-to-ProofRAG benchmark "
            "artifact pipeline from a normalized export."
        )
    )
    parser.add_argument("--minirag-export", required=True)
    parser.add_argument("--output-dir", default="experiments/results/full_benchmark")
    parser.add_argument("--lihua-qa-csv")
    parser.add_argument("--lihua-data-dir")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--review-date", required=True)
    parser.add_argument("--benchmark-scope", default="Full LiHua-World benchmark")
    parser.add_argument("--cost-notes", default="not reviewed")
    parser.add_argument("--judge-config", default="deterministic claim-level scorer")
    parser.add_argument("--spot-check-notes", default="not recorded")
    parser.add_argument("--known-limitations", default="none recorded")
    parser.add_argument("--follow-up", default="none")
    parser.add_argument("--docker-evidence")
    parser.add_argument("--ci-evidence")
    parser.add_argument("--ci-url")
    parser.add_argument("--faithfulness-scorer", choices=("claim", "llm-judge"), default="claim")
    parser.add_argument("--judge-backend", choices=("ollama", "openai-compatible", "transformers"), default="ollama")
    parser.add_argument("--judge-model", default="qwen3.5:4b")
    parser.add_argument("--judge-base-url", default="http://localhost:11434")
    parser.add_argument("--judge-api-key")
    parser.add_argument("--judge-timeout", type=int, default=120)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument("--judge-endpoint-mode", choices=("chat", "generate"), default="chat")
    parser.add_argument("--claim-min-total", type=int, default=100)
    parser.add_argument("--min-baseline-export-rows", type=int, default=100)
    parser.add_argument("--claim-max-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--claim-min-precision-at-answered", type=float, default=0.75)
    parser.add_argument("--claim-max-unsafe-allow-rate", type=float, default=0.0)
    parser.add_argument("--claim-min-groundedness-delta", type=float, default=0.10)
    parser.add_argument("--claim-max-unsupported-claim-ratio", type=float, default=0.75)
    parser.add_argument("--claim-alpha", type=float, default=0.05)
    parser.add_argument("--require-claim-significance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for command in build_commands(args):
        _run(command, dry_run=args.dry_run)
    print(f"Full benchmark artifacts written under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
