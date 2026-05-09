from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from proofrag.evaluation.lihua import load_lihua_qa_csv
from proofrag.evaluation.minirag_adapter import MiniRAGOutputAdapter


@dataclass(frozen=True)
class CompletionGate:
    name: str
    passed: bool
    detail: str
    evidence: str | None = None


def _existing_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    return candidate if candidate.exists() else None


def _nonempty_file(path: str | None) -> Path | None:
    candidate = _existing_path(path)
    if candidate is None or not candidate.is_file() or candidate.stat().st_size == 0:
        return None
    return candidate


def _count_export_rows(path: Path) -> int:
    adapter = MiniRAGOutputAdapter()
    return len(adapter.load_export(str(path)))


def _check_lihua_data(
    qa_csv: str | None,
    data_dir: str | None,
    min_rows: int,
) -> CompletionGate:
    qa_path = _nonempty_file(qa_csv)
    data_path = _existing_path(data_dir)
    if qa_path and data_path and data_path.is_dir():
        try:
            rows = load_lihua_qa_csv(qa_path)
        except Exception as exc:
            return CompletionGate(
                name="full_lihua_world_data",
                passed=False,
                detail=f"LiHua QA CSV failed validation: {exc}",
                evidence=f"{qa_path}; {data_path}",
            )
        if len(rows) < min_rows:
            return CompletionGate(
                name="full_lihua_world_data",
                passed=False,
                detail=f"LiHua QA CSV has {len(rows)} rows; expected at least {min_rows}.",
                evidence=f"{qa_path}; {data_path}",
            )
        source_files = [
            path
            for path in data_path.rglob("*")
            if path.is_file() and not path.name.startswith(".")
        ]
        if not source_files:
            return CompletionGate(
                name="full_lihua_world_data",
                passed=False,
                detail="LiHua data directory contains no source files.",
                evidence=f"{qa_path}; {data_path}",
            )
        return CompletionGate(
            name="full_lihua_world_data",
            passed=True,
            detail=(
                f"LiHua QA CSV has {len(rows)} rows and extracted data directory "
                f"has {len(source_files)} files."
            ),
            evidence=f"{qa_path}; {data_path}",
        )
    return CompletionGate(
        name="full_lihua_world_data",
        passed=False,
        detail=(
            "Full LiHua-World data was not proven. Provide --lihua-qa-csv and "
            "--lihua-data-dir pointing to local external artifacts."
        ),
    )


def _check_baseline_export(path: str | None, min_rows: int) -> CompletionGate:
    export_path = _nonempty_file(path)
    if export_path is None:
        return CompletionGate(
            name="normalized_baseline_export",
            passed=False,
            detail=(
                "No non-empty normalized MiniRAG/LightRAG export was provided. "
                "Use --minirag-export with a JSONL export."
            ),
        )
    try:
        rows = _count_export_rows(export_path)
    except Exception as exc:
        return CompletionGate(
            name="normalized_baseline_export",
            passed=False,
            detail=f"Export exists but failed schema validation: {exc}",
            evidence=str(export_path),
        )
    return CompletionGate(
        name="normalized_baseline_export",
        passed=rows >= min_rows,
        detail=f"Validated {rows} normalized export rows; expected at least {min_rows}.",
        evidence=str(export_path),
    )


def _check_result_artifacts(
    comparison_summary: str | None,
    faithfulness_summary: str | None,
    review_note: str | None,
) -> CompletionGate:
    comparison_path = _nonempty_file(comparison_summary)
    faithfulness_path = _nonempty_file(faithfulness_summary)
    review_path = _nonempty_file(review_note)
    if comparison_path and faithfulness_path and review_path:
        try:
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            faithfulness = json.loads(faithfulness_path.read_text(encoding="utf-8"))
            _validate_result_artifact_shapes(comparison, faithfulness)
            _validate_review_note(review_path)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return CompletionGate(
                name="reviewed_result_artifacts",
                passed=False,
                detail=f"Reviewed metric artifacts failed validation: {exc}",
                evidence=f"{comparison_path}; {faithfulness_path}; {review_path}",
            )
        return CompletionGate(
            name="reviewed_result_artifacts",
            passed=True,
            detail=(
                "Comparison, faithfulness, and review-note artifacts are present and schema-valid."
            ),
            evidence=f"{comparison_path}; {faithfulness_path}; {review_path}",
        )
    return CompletionGate(
        name="reviewed_result_artifacts",
        passed=False,
        detail=(
            "Reviewed full-benchmark metrics were not proven. Provide "
            "--comparison-summary, --faithfulness-summary, and --review-note."
        ),
    )


def _validate_result_artifact_shapes(
    comparison: dict[str, Any],
    faithfulness: dict[str, Any],
) -> None:
    baseline = comparison["baseline"]
    proofrag = comparison["proofrag"]
    paired = comparison["paired_answer_accuracy"]
    faith_summary = faithfulness["summary"]

    total = int(proofrag["total"])
    baseline_total = int(baseline["total"])
    faith_total = int(faith_summary["total"])
    if total <= 0:
        raise ValueError("comparison proofrag.total must be positive")
    if baseline_total != total or faith_total != total:
        raise ValueError(
            "comparison baseline/proofrag totals must match faithfulness summary total"
        )

    float(proofrag["accuracy"])
    float(baseline["accuracy"])
    float(proofrag["precision_at_answered"])
    int(proofrag.get("unsafe_allow_count", 0))
    float(paired["exact_p_value"])
    float(faith_summary["baseline_mean_groundedness"])
    float(faith_summary["proofrag_mean_groundedness"])
    int(faith_summary["baseline_unsupported_claims"])
    int(faith_summary["proofrag_unsupported_claims"])


def _validate_review_note(path: Path) -> None:
    text = path.read_text(encoding="utf-8").strip().lower()
    if "review" not in text or "benchmark" not in text:
        raise ValueError("review note must mention review and benchmark scope")


def _check_docker_build(
    check_docker_daemon: bool,
    docker_evidence: str | None,
    docker_build_tag: str,
    docker_build_context: str,
) -> CompletionGate:
    evidence_path = _nonempty_file(docker_evidence)
    if evidence_path:
        try:
            _validate_docker_evidence(evidence_path)
        except ValueError as exc:
            return CompletionGate(
                name="docker_build_verified",
                passed=False,
                detail=f"Docker build evidence failed validation: {exc}",
                evidence=str(evidence_path),
            )
        return CompletionGate(
            name="docker_build_verified",
            passed=True,
            detail="Docker build evidence file is present and indicates success.",
            evidence=str(evidence_path),
        )
    if not check_docker_daemon:
        return CompletionGate(
            name="docker_build_verified",
            passed=False,
            detail=(
                "Docker build was not checked. Provide --docker-evidence or run "
                "with --check-docker-build on a machine with Docker available."
            ),
        )
    context_path = _existing_path(docker_build_context)
    if context_path is None:
        return CompletionGate(
            name="docker_build_verified",
            passed=False,
            detail=f"Docker build context does not exist: {docker_build_context}",
        )
    try:
        result = subprocess.run(
            ["docker", "build", "-t", docker_build_tag, str(context_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return CompletionGate(
            name="docker_build_verified",
            passed=False,
            detail=f"Docker build failed: {exc}",
        )
    return CompletionGate(
        name="docker_build_verified",
        passed=True,
        detail="Docker image built successfully.",
        evidence=f"docker build -t {docker_build_tag} {context_path}; {result.stdout[-500:]}",
    )


def _validate_docker_evidence(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    if "docker build" not in text:
        raise ValueError("evidence must mention docker build")
    if not any(marker in text for marker in ("success", "succeeded", "successfully built")):
        raise ValueError("evidence must indicate successful build completion")


def _check_remote_ci(ci_evidence: str | None, ci_url: str | None) -> CompletionGate:
    evidence_path = _nonempty_file(ci_evidence)
    if evidence_path:
        try:
            _validate_ci_evidence(evidence_path)
        except ValueError as exc:
            return CompletionGate(
                name="remote_ci_verified",
                passed=False,
                detail=f"Remote CI evidence failed validation: {exc}",
                evidence=str(evidence_path),
            )
    if ci_url and not _is_github_actions_run_url(ci_url):
        return CompletionGate(
            name="remote_ci_verified",
            passed=False,
            detail="Remote CI URL must be a GitHub Actions run URL.",
            evidence=ci_url,
        )
    if evidence_path or ci_url:
        return CompletionGate(
            name="remote_ci_verified",
            passed=True,
            detail="Remote CI evidence was supplied and indicates success.",
            evidence=str(evidence_path) if evidence_path else ci_url,
        )
    return CompletionGate(
        name="remote_ci_verified",
        passed=False,
        detail="Remote GitHub CI run evidence was not supplied.",
    )


def _validate_ci_evidence(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    if not any(marker in text for marker in ("github", "actions", "ci")):
        raise ValueError("evidence must mention GitHub Actions or CI")
    if not any(marker in text for marker in ("success", "succeeded", "conclusion: success", '"conclusion": "success"')):
        raise ValueError("evidence must indicate a successful CI conclusion")


def _is_github_actions_run_url(url: str) -> bool:
    return re.match(r"^https://github\.com/[^/]+/[^/]+/actions/runs/\d+(?:\b|[/?#])", url) is not None


def build_completion_report(args: argparse.Namespace) -> dict[str, Any]:
    gates = [
        _check_lihua_data(
            args.lihua_qa_csv,
            args.lihua_data_dir,
            args.min_lihua_qa_rows,
        ),
        _check_baseline_export(args.minirag_export, args.min_baseline_export_rows),
        _check_result_artifacts(
            args.comparison_summary,
            args.faithfulness_summary,
            args.review_note,
        ),
        _check_docker_build(
            args.check_docker_daemon,
            args.docker_evidence,
            args.docker_build_tag,
            args.docker_build_context,
        ),
        _check_remote_ci(args.ci_evidence, args.ci_url),
    ]
    passed = all(gate.passed for gate in gates)
    return {
        "ready_for_superiority_claim": passed,
        "status": "ready" if passed else "blocked",
        "gates": [asdict(gate) for gate in gates],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether ProofRAG's external publication gates have concrete "
            "evidence artifacts."
        )
    )
    parser.add_argument("--lihua-qa-csv", help="External LiHua-World QA CSV.")
    parser.add_argument("--lihua-data-dir", help="Extracted external LiHua-World data dir.")
    parser.add_argument("--min-lihua-qa-rows", type=int, default=100)
    parser.add_argument("--minirag-export", help="Normalized MiniRAG/LightRAG JSONL export.")
    parser.add_argument("--min-baseline-export-rows", type=int, default=100)
    parser.add_argument("--comparison-summary", help="Full benchmark comparison JSON summary.")
    parser.add_argument("--faithfulness-summary", help="Full benchmark faithfulness JSON summary.")
    parser.add_argument("--review-note", help="Human review note for the full benchmark artifacts.")
    parser.add_argument("--docker-evidence", help="Text/JSON evidence from a successful Docker build.")
    parser.add_argument("--ci-evidence", help="Text/JSON evidence from a successful remote CI run.")
    parser.add_argument("--ci-url", help="URL to a successful remote CI run.")
    parser.add_argument(
        "--check-docker-daemon",
        "--check-docker-build",
        dest="check_docker_daemon",
        action="store_true",
        help="Run a local Docker build as Docker verification.",
    )
    parser.add_argument("--docker-build-tag", default="proofrag:completion-gate")
    parser.add_argument("--docker-build-context", default=".")
    parser.add_argument("--output-json", help="Optional path to write the gate report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_completion_report(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(f"{payload}\n", encoding="utf-8")
    print(payload)
    return 0 if report["ready_for_superiority_claim"] else 1


if __name__ == "__main__":
    sys.exit(main())
