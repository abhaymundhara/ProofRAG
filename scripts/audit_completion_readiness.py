#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_completion_gates import build_completion_report
from scripts.check_roadmap_artifacts import build_artifact_report


OBJECTIVE = (
    "Turn ProofRAG into a mature, production-grade, publishable research "
    "framework and only claim superiority over MiniRAG after external "
    "LiHua/MiniRAG, Docker, CI, and reviewed metric evidence pass hard gates."
)

GATE_BLOCKER_TEXT = {
    "full_lihua_world_data": {"Full LiHua-World data is external."},
    "normalized_baseline_export": {
        "Full external MiniRAG/LightRAG exports are not present.",
    },
    "reviewed_result_artifacts": {
        "Reviewed full-benchmark metrics are not present.",
    },
    "docker_build_verified": {"Docker image build evidence is not present."},
    "remote_ci_verified": {"Remote GitHub CI run evidence is not present."},
}


def _default_completion_args() -> argparse.Namespace:
    return argparse.Namespace(
        lihua_qa_csv=None,
        lihua_data_dir=None,
        min_lihua_qa_rows=100,
        min_lihua_source_resolution=0.90,
        minirag_export=None,
        min_baseline_export_rows=100,
        comparison_summary=None,
        faithfulness_summary=None,
        review_note=None,
        claim_min_total=100,
        claim_max_accuracy_drop=0.05,
        claim_min_precision_at_answered=0.75,
        claim_max_unsafe_allow_rate=0.0,
        claim_min_groundedness_delta=0.10,
        claim_max_unsupported_claim_ratio=0.75,
        require_claim_significance=False,
        claim_alpha=0.05,
        docker_evidence=None,
        docker_build_tag="proofrag:completion-gate",
        docker_build_context=".",
        ci_evidence=None,
        ci_url=None,
        check_docker_daemon=False,
        output_json=None,
    )


def build_audit_report(completion_args: argparse.Namespace | None = None) -> dict[str, Any]:
    roadmap = build_artifact_report()
    gates = build_completion_report(completion_args or _default_completion_args())
    failed_gate_names = {gate["name"] for gate in gates["gates"] if not gate["passed"]}
    open_items = [
        {
            "name": gate["name"],
            "detail": gate["detail"],
        }
        for gate in gates["gates"]
        if not gate["passed"]
    ]
    local_ready = roadmap["status"] == "passed_local_artifact_check"
    external_ready = bool(gates["ready_for_superiority_claim"])
    return {
        "objective": OBJECTIVE,
        "status": "ready" if local_ready and external_ready else "blocked",
        "local_artifacts_ready": local_ready,
        "external_superiority_ready": external_ready,
        "open_items": open_items,
        "roadmap_status": roadmap["status"],
        "externally_blocked_requirements": _active_roadmap_blockers(
            roadmap["externally_blocked"],
            failed_gate_names,
        ),
        "completion_gates": gates,
        "roadmap_artifacts": roadmap,
    }


def _active_roadmap_blockers(
    roadmap_blockers: list[dict[str, Any]],
    failed_gate_names: set[str],
) -> list[dict[str, Any]]:
    active_blocker_text = {
        blocker
        for gate_name in failed_gate_names
        for blocker in GATE_BLOCKER_TEXT.get(gate_name, set())
    }
    active_rows = []
    for row in roadmap_blockers:
        blockers = [
            blocker
            for blocker in row["external_blockers"]
            if blocker in active_blocker_text
        ]
        if not blockers:
            continue
        active_rows.append(
            {
                "phase": row["phase"],
                "requirement": row["requirement"],
                "external_blockers": blockers,
            }
        )
    return active_rows


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# ProofRAG Completion Readiness Audit",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Objective",
        "",
        report["objective"],
        "",
        "## Readiness",
        "",
        f"- Local artifacts ready: `{report['local_artifacts_ready']}`",
        f"- External superiority ready: `{report['external_superiority_ready']}`",
        f"- Roadmap status: `{report['roadmap_status']}`",
        "",
        "## Open Items",
        "",
    ]
    if report["open_items"]:
        for item in report["open_items"]:
            lines.append(f"- `{item['name']}`: {item['detail']}")
    else:
        lines.append("- None")
    lines.extend(["", "## Externally Blocked Roadmap Requirements", ""])
    for row in report["externally_blocked_requirements"]:
        blockers = "; ".join(row["external_blockers"]) or "none"
        lines.append(f"- {row['phase']} / {row['requirement']}: {blockers}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit current ProofRAG roadmap and external completion readiness."
    )
    parser.add_argument("--lihua-qa-csv")
    parser.add_argument("--lihua-data-dir")
    parser.add_argument("--min-lihua-qa-rows", type=int, default=100)
    parser.add_argument("--min-lihua-source-resolution", type=float, default=0.90)
    parser.add_argument("--minirag-export")
    parser.add_argument("--min-baseline-export-rows", type=int, default=100)
    parser.add_argument("--comparison-summary")
    parser.add_argument("--faithfulness-summary")
    parser.add_argument("--review-note")
    parser.add_argument("--claim-min-total", type=int, default=100)
    parser.add_argument("--claim-max-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--claim-min-precision-at-answered", type=float, default=0.75)
    parser.add_argument("--claim-max-unsafe-allow-rate", type=float, default=0.0)
    parser.add_argument("--claim-min-groundedness-delta", type=float, default=0.10)
    parser.add_argument("--claim-max-unsupported-claim-ratio", type=float, default=0.75)
    parser.add_argument("--require-claim-significance", action="store_true")
    parser.add_argument("--claim-alpha", type=float, default=0.05)
    parser.add_argument("--docker-evidence")
    parser.add_argument("--ci-evidence")
    parser.add_argument("--ci-url")
    parser.add_argument(
        "--check-docker-daemon",
        "--check-docker-build",
        dest="check_docker_daemon",
        action="store_true",
    )
    parser.add_argument("--docker-build-tag", default="proofrag:completion-gate")
    parser.add_argument("--docker-build-context", default=".")
    parser.add_argument("--output-json", help="Optional JSON output path.")
    parser.add_argument("--output-md", help="Optional Markdown output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_audit_report(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(f"{payload}\n", encoding="utf-8")
    if args.output_md:
        write_markdown(Path(args.output_md), report)
    print(payload)
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
