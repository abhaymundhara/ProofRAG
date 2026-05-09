#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReleaseCommand:
    name: str
    command: list[str]
    required: bool = True
    expected_exit_codes: tuple[int, ...] = (0,)
    detail: str = ""


@dataclass
class ReleaseCheckResult:
    name: str
    command: list[str]
    status: str
    returncode: int | None = None
    duration_seconds: float = 0.0
    detail: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def build_release_commands(args: argparse.Namespace) -> list[ReleaseCommand]:
    output_dir = Path(args.output_dir)
    dist_dir = output_dir / "dist"
    reproduction_dir = output_dir / "reproducible"
    completion_report = output_dir / "completion_gates.json"
    readiness_report = output_dir / "completion_readiness_audit.json"
    cli_output = output_dir / "cli_smoke.jsonl"

    commands = [
        ReleaseCommand(
            name="ruff",
            command=[sys.executable, "-m", "ruff", "check", "proofrag", "scripts", "tests"],
            detail="Repository lint gate.",
        ),
        ReleaseCommand(
            name="mypy",
            command=[sys.executable, "-m", "mypy"],
            detail="Package type-check gate.",
        ),
        ReleaseCommand(
            name="pytest",
            command=[sys.executable, "-m", "pytest"],
            detail="Full test suite.",
        ),
        ReleaseCommand(
            name="toy_benchmark",
            command=[sys.executable, "scripts/run_toy_benchmark.py"],
            detail="Deterministic 30-example toy benchmark.",
        ),
        ReleaseCommand(
            name="cli_hybrid_iterative_smoke",
            command=[
                sys.executable,
                "-m",
                "proofrag.cli",
                "ask",
                "--question",
                "Who asked LiHua about the laptop warranty issue?",
                "--config",
                "configs/default.yaml",
                "--retriever",
                "hybrid",
                "--iterative",
                "--max-retrieval-rounds",
                "2",
                "--output",
                str(cli_output),
                "--json",
            ],
            detail="CLI smoke for hybrid retrieval plus iterative contract filling.",
        ),
        ReleaseCommand(
            name="reproduce_sample_artifacts",
            command=[
                "bash",
                "scripts/reproduce_paper_results.sh",
                "benchmarks/sample_minirag_export.jsonl",
                str(reproduction_dir),
            ],
            detail="Sample MiniRAG export comparison, ablation, chart, and tables.",
        ),
        ReleaseCommand(
            name="package_build",
            command=[
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--wheel",
                "--outdir",
                str(dist_dir),
                "--no-isolation",
            ],
            detail="Build source distribution and wheel using the current dev environment.",
        ),
        ReleaseCommand(
            name="distribution_contents",
            command=[
                sys.executable,
                "scripts/verify_distribution_contents.py",
                "--dist-dir",
                str(dist_dir),
                "--output-json",
                str(output_dir / "distribution_contents.json"),
            ],
            detail="Verify sdist and wheel include reproducibility assets.",
        ),
        ReleaseCommand(
            name="roadmap_artifact_matrix",
            command=[
                sys.executable,
                "scripts/check_roadmap_artifacts.py",
                "--output-json",
                str(output_dir / "roadmap_artifacts.json"),
                "--output-md",
                str(output_dir / "roadmap_artifact_matrix.md"),
            ],
            detail="Prompt-to-artifact roadmap checklist.",
        ),
        ReleaseCommand(
            name="completion_readiness_audit",
            command=_readiness_audit_args(
                args,
                readiness_report,
                output_dir / "completion_readiness_audit.md",
            ),
            required=bool(args.require_external_gates),
            expected_exit_codes=(0,) if args.require_external_gates else (0, 1),
            detail=(
                "Combined local roadmap and external superiority readiness audit. "
                "Exit 1 is expected unless external evidence is supplied."
            ),
        ),
        ReleaseCommand(
            name="external_completion_gates",
            command=_completion_gate_args(args, completion_report),
            required=bool(args.require_external_gates),
            expected_exit_codes=(0,) if args.require_external_gates else (0, 1),
            detail=(
                "Full external LiHua/MiniRAG, Docker, and remote CI gates. "
                "Exit 1 is expected unless external evidence is supplied."
            ),
        ),
    ]

    claim_args = _claim_validator_args(args, output_dir)
    if claim_args:
        commands.append(
            ReleaseCommand(
                name="publication_claim_validation",
                command=claim_args,
                required=bool(args.require_publication_claims),
                expected_exit_codes=(0,) if args.require_publication_claims else (0, 1),
                detail=(
                    "Metric-threshold validation for full benchmark superiority claims."
                ),
            )
        )
    elif args.require_publication_claims:
        commands.append(
            ReleaseCommand(
                name="publication_claim_validation",
                command=[],
                required=True,
                expected_exit_codes=(),
                detail=(
                    "Skipped because --claim-comparison-summary and "
                    "--claim-faithfulness-summary were not both provided."
                ),
            )
        )

    return commands


def _completion_gate_args(args: argparse.Namespace, output_json: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/check_completion_gates.py",
        "--output-json",
        str(output_json),
    ]
    optional_flags = [
        ("--lihua-qa-csv", args.lihua_qa_csv),
        ("--lihua-data-dir", args.lihua_data_dir),
        ("--min-lihua-qa-rows", args.min_lihua_qa_rows),
        ("--min-lihua-source-resolution", args.min_lihua_source_resolution),
        ("--minirag-export", args.minirag_export),
        ("--min-baseline-export-rows", args.min_baseline_export_rows),
        ("--comparison-summary", args.comparison_summary),
        ("--faithfulness-summary", args.faithfulness_summary),
        ("--review-note", args.review_note),
        ("--docker-evidence", args.docker_evidence),
        ("--ci-evidence", args.ci_evidence),
        ("--ci-url", args.ci_url),
        ("--docker-build-tag", args.docker_build_tag),
        ("--docker-build-context", args.docker_build_context),
        ("--claim-min-total", args.claim_min_total),
        ("--claim-max-accuracy-drop", args.claim_max_accuracy_drop),
        ("--claim-min-precision-at-answered", args.claim_min_precision_at_answered),
        ("--claim-max-unsafe-allow-rate", args.claim_max_unsafe_allow_rate),
        ("--claim-min-groundedness-delta", args.claim_min_groundedness_delta),
        ("--claim-max-unsupported-claim-ratio", args.claim_max_unsupported_claim_ratio),
        ("--claim-alpha", args.claim_alpha),
    ]
    for flag, value in optional_flags:
        if value is not None:
            command.extend([flag, str(value)])
    if args.check_docker_build:
        command.append("--check-docker-build")
    if args.require_significance:
        command.append("--require-claim-significance")
    return command


def _readiness_audit_args(args: argparse.Namespace, output_json: Path, output_md: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/audit_completion_readiness.py",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]
    optional_flags = [
        ("--lihua-qa-csv", args.lihua_qa_csv),
        ("--lihua-data-dir", args.lihua_data_dir),
        ("--min-lihua-qa-rows", args.min_lihua_qa_rows),
        ("--min-lihua-source-resolution", args.min_lihua_source_resolution),
        ("--minirag-export", args.minirag_export),
        ("--min-baseline-export-rows", args.min_baseline_export_rows),
        ("--comparison-summary", args.comparison_summary),
        ("--faithfulness-summary", args.faithfulness_summary),
        ("--review-note", args.review_note),
        ("--docker-evidence", args.docker_evidence),
        ("--ci-evidence", args.ci_evidence),
        ("--ci-url", args.ci_url),
        ("--docker-build-tag", args.docker_build_tag),
        ("--docker-build-context", args.docker_build_context),
        ("--claim-min-total", args.claim_min_total),
        ("--claim-max-accuracy-drop", args.claim_max_accuracy_drop),
        ("--claim-min-precision-at-answered", args.claim_min_precision_at_answered),
        ("--claim-max-unsafe-allow-rate", args.claim_max_unsafe_allow_rate),
        ("--claim-min-groundedness-delta", args.claim_min_groundedness_delta),
        ("--claim-max-unsupported-claim-ratio", args.claim_max_unsupported_claim_ratio),
        ("--claim-alpha", args.claim_alpha),
    ]
    for flag, value in optional_flags:
        if value is not None:
            command.extend([flag, str(value)])
    if args.check_docker_build:
        command.append("--check-docker-build")
    if args.require_significance:
        command.append("--require-claim-significance")
    return command


def _claim_validator_args(args: argparse.Namespace, output_dir: Path) -> list[str]:
    if not args.claim_comparison_summary and not args.claim_faithfulness_summary:
        return []
    command = [
        sys.executable,
        "scripts/validate_publication_claims.py",
        "--comparison-summary",
        str(args.claim_comparison_summary or ""),
        "--faithfulness-summary",
        str(args.claim_faithfulness_summary or ""),
        "--output-json",
        str(output_dir / "publication_claim_validation.json"),
    ]
    if args.require_significance:
        command.append("--require-significance")
    optional_flags = [
        ("--min-total", args.claim_min_total),
        ("--max-accuracy-drop", args.claim_max_accuracy_drop),
        ("--min-precision-at-answered", args.claim_min_precision_at_answered),
        ("--max-unsafe-allow-rate", args.claim_max_unsafe_allow_rate),
        ("--min-groundedness-delta", args.claim_min_groundedness_delta),
        ("--max-unsupported-claim-ratio", args.claim_max_unsupported_claim_ratio),
        ("--alpha", args.claim_alpha),
    ]
    for flag, value in optional_flags:
        if value is not None:
            command.extend([flag, str(value)])
    return command


def run_command(command: ReleaseCommand) -> ReleaseCheckResult:
    if not command.command:
        return ReleaseCheckResult(
            name=command.name,
            command=command.command,
            status="failed" if command.required else "skipped",
            detail=command.detail,
        )
    started = time.monotonic()
    completed = subprocess.run(
        command.command,
        check=False,
        text=True,
        capture_output=True,
    )
    duration = time.monotonic() - started
    passed = completed.returncode in command.expected_exit_codes
    status = "passed" if passed else "failed"
    if passed and completed.returncode != 0:
        status = "blocked_expected"
    return ReleaseCheckResult(
        name=command.name,
        command=command.command,
        status=status,
        returncode=completed.returncode,
        duration_seconds=round(duration, 3),
        detail=command.detail,
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
    )


def write_report(path: Path, results: list[ReleaseCheckResult]) -> dict[str, Any]:
    failed_required = [
        result
        for result in results
        if result.status == "failed"
    ]
    report = {
        "status": "passed" if not failed_required else "failed",
        "generated_at_unix": int(time.time()),
        "results": [asdict(result) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ProofRAG's local release verification gates."
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/proofrag_release_checks",
        help="Directory for generated reports and release artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command manifest without executing commands.",
    )
    parser.add_argument(
        "--require-external-gates",
        action="store_true",
        help="Require external completion gates to pass instead of accepting blocked status.",
    )
    parser.add_argument("--lihua-qa-csv")
    parser.add_argument("--lihua-data-dir")
    parser.add_argument("--min-lihua-qa-rows", type=int)
    parser.add_argument("--min-lihua-source-resolution", type=float)
    parser.add_argument("--minirag-export")
    parser.add_argument("--min-baseline-export-rows", type=int)
    parser.add_argument("--comparison-summary")
    parser.add_argument("--faithfulness-summary")
    parser.add_argument("--review-note")
    parser.add_argument("--docker-evidence")
    parser.add_argument("--ci-evidence")
    parser.add_argument("--ci-url")
    parser.add_argument("--check-docker-build", action="store_true")
    parser.add_argument("--docker-build-tag")
    parser.add_argument("--docker-build-context")
    parser.add_argument(
        "--require-publication-claims",
        action="store_true",
        help="Require publication claim validation to pass.",
    )
    parser.add_argument("--claim-comparison-summary")
    parser.add_argument("--claim-faithfulness-summary")
    parser.add_argument("--claim-min-total", type=int)
    parser.add_argument("--claim-max-accuracy-drop", type=float)
    parser.add_argument("--claim-min-precision-at-answered", type=float)
    parser.add_argument("--claim-max-unsafe-allow-rate", type=float)
    parser.add_argument("--claim-min-groundedness-delta", type=float)
    parser.add_argument("--claim-max-unsupported-claim-ratio", type=float)
    parser.add_argument("--claim-alpha", type=float)
    parser.add_argument(
        "--require-significance",
        action="store_true",
        help="Require paired significance when claim artifacts are supplied.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    commands = build_release_commands(args)

    if args.dry_run:
        print(json.dumps([asdict(command) for command in commands], indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_command(command) for command in commands]
    report = write_report(output_dir / "release_checks.json", results)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
