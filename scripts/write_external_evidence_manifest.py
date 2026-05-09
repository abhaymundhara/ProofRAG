#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvidenceRequirement:
    name: str
    purpose: str
    paths: tuple[str, ...]
    validation: str
    status: str


def _status(paths: tuple[str, ...]) -> str:
    return "provided" if all(paths) else "missing"


def _path_tuple(*paths: str | None) -> tuple[str, ...]:
    return tuple(path or "" for path in paths)


def _extend_optional(command: list[str], flag: str, value: str | int | float | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def build_gate_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "scripts/check_completion_gates.py"]
    for flag, value in [
        ("--lihua-qa-csv", args.lihua_qa_csv),
        ("--lihua-data-dir", args.lihua_data_dir),
        ("--min-lihua-qa-rows", args.min_lihua_qa_rows),
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
    ]:
        _extend_optional(command, flag, value)
    if args.check_docker_build:
        command.append("--check-docker-build")
    if args.require_claim_significance:
        command.append("--require-claim-significance")
    return command


def build_release_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_local_release_checks.py",
        "--require-external-gates",
        "--require-publication-claims",
        "--require-significance",
        "--output-dir",
        args.output_dir,
    ]
    for flag, value in [
        ("--lihua-qa-csv", args.lihua_qa_csv),
        ("--lihua-data-dir", args.lihua_data_dir),
        ("--min-lihua-qa-rows", args.min_lihua_qa_rows),
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
        ("--claim-comparison-summary", args.comparison_summary),
        ("--claim-faithfulness-summary", args.faithfulness_summary),
        ("--claim-min-total", args.claim_min_total),
        ("--claim-max-accuracy-drop", args.claim_max_accuracy_drop),
        ("--claim-min-precision-at-answered", args.claim_min_precision_at_answered),
        ("--claim-max-unsafe-allow-rate", args.claim_max_unsafe_allow_rate),
        ("--claim-min-groundedness-delta", args.claim_min_groundedness_delta),
        ("--claim-max-unsupported-claim-ratio", args.claim_max_unsupported_claim_ratio),
        ("--claim-alpha", args.claim_alpha),
    ]:
        _extend_optional(command, flag, value)
    if args.check_docker_build:
        command.append("--check-docker-build")
    return command


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    requirements = [
        EvidenceRequirement(
            name="full_lihua_world_data",
            purpose="Prove evaluation ran against a local full LiHua-World copy, not a toy fixture.",
            paths=_path_tuple(args.lihua_qa_csv, args.lihua_data_dir),
            validation=f"QA CSV must parse and contain at least {args.min_lihua_qa_rows} rows; data dir must contain source files.",
            status=_status(_path_tuple(args.lihua_qa_csv, args.lihua_data_dir)),
        ),
        EvidenceRequirement(
            name="normalized_baseline_export",
            purpose="Provide MiniRAG or LightRAG outputs in ProofRAG's normalized JSONL schema.",
            paths=_path_tuple(args.minirag_export),
            validation=f"Export must schema-validate and contain at least {args.min_baseline_export_rows} rows.",
            status=_status(_path_tuple(args.minirag_export)),
        ),
        EvidenceRequirement(
            name="reviewed_result_artifacts",
            purpose="Provide full-benchmark comparison, faithfulness, and review artifacts.",
            paths=_path_tuple(
                args.comparison_summary,
                args.faithfulness_summary,
                args.review_note,
            ),
            validation="Metric JSON files must have compatible totals; review note must mention review and benchmark scope.",
            status=_status(
                _path_tuple(
                    args.comparison_summary,
                    args.faithfulness_summary,
                    args.review_note,
                )
            ),
        ),
        EvidenceRequirement(
            name="docker_build_verified",
            purpose="Prove the Docker image builds outside the source tree.",
            paths=_path_tuple(args.docker_evidence),
            validation="Evidence must mention a successful docker build, or use --check-docker-build on a Docker-enabled machine.",
            status="provided" if args.docker_evidence or args.check_docker_build else "missing",
        ),
        EvidenceRequirement(
            name="remote_ci_verified",
            purpose="Prove the release checks ran in remote GitHub Actions.",
            paths=_path_tuple(args.ci_evidence or args.ci_url),
            validation="Evidence must indicate successful GitHub Actions/CI, or CI URL must be a GitHub Actions run URL.",
            status="provided" if args.ci_evidence or args.ci_url else "missing",
        ),
    ]
    gate_command = build_gate_command(args)
    release_command = build_release_command(args)
    return {
        "status": "ready_to_validate" if all(item.status == "provided" for item in requirements) else "missing_evidence",
        "requirements": [asdict(item) for item in requirements],
        "commands": {
            "completion_gate": gate_command,
            "completion_gate_shell": _quote_command(gate_command),
            "full_release": release_command,
            "full_release_shell": _quote_command(release_command),
        },
    }


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# ProofRAG External Evidence Manifest",
        "",
        f"Status: `{manifest['status']}`",
        "",
        "## Requirements",
        "",
    ]
    for item in manifest["requirements"]:
        paths = ", ".join(f"`{path}`" for path in item["paths"] if path) or "`<missing>`"
        lines.extend(
            [
                f"### {item['name']}",
                "",
                f"- Status: `{item['status']}`",
                f"- Purpose: {item['purpose']}",
                f"- Paths: {paths}",
                f"- Validation: {item['validation']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Commands",
            "",
            "Completion gate:",
            "",
            "```bash",
            manifest["commands"]["completion_gate_shell"],
            "```",
            "",
            "Full release validation:",
            "",
            "```bash",
            manifest["commands"]["full_release_shell"],
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a reviewer-facing manifest for ProofRAG external evidence artifacts."
    )
    parser.add_argument("--lihua-qa-csv")
    parser.add_argument("--lihua-data-dir")
    parser.add_argument("--min-lihua-qa-rows", type=int, default=100)
    parser.add_argument("--minirag-export")
    parser.add_argument("--min-baseline-export-rows", type=int, default=100)
    parser.add_argument("--comparison-summary")
    parser.add_argument("--faithfulness-summary")
    parser.add_argument("--review-note")
    parser.add_argument("--docker-evidence")
    parser.add_argument("--ci-evidence")
    parser.add_argument("--ci-url")
    parser.add_argument("--check-docker-build", action="store_true")
    parser.add_argument("--docker-build-tag", default="proofrag:completion-gate")
    parser.add_argument("--docker-build-context", default=".")
    parser.add_argument("--claim-min-total", type=int, default=100)
    parser.add_argument("--claim-max-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--claim-min-precision-at-answered", type=float, default=0.75)
    parser.add_argument("--claim-max-unsafe-allow-rate", type=float, default=0.0)
    parser.add_argument("--claim-min-groundedness-delta", type=float, default=0.10)
    parser.add_argument("--claim-max-unsupported-claim-ratio", type=float, default=0.75)
    parser.add_argument("--claim-alpha", type=float, default=0.05)
    parser.add_argument(
        "--require-claim-significance",
        action="store_true",
        help="Add paired exact-test significance to the generated completion-gate command.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/full_release_checks",
        help="Output directory used in the generated full release command.",
    )
    parser.add_argument("--output-json", help="Optional JSON manifest path.")
    parser.add_argument("--output-md", help="Optional Markdown manifest path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{payload}\n", encoding="utf-8")
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(path, manifest)
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
