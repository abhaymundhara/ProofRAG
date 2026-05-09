#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RoadmapRequirement:
    phase: str
    requirement: str
    artifacts: tuple[str, ...]
    verification: tuple[str, ...]
    external_blockers: tuple[str, ...] = ()


REQUIREMENTS: tuple[RoadmapRequirement, ...] = (
    RoadmapRequirement(
        phase="Phase 0",
        requirement="FACE execution plan with files to create or modify",
        artifacts=("docs/FACE_EXECUTION_PLAN.md",),
        verification=("docs/completion_audit.md",),
    ),
    RoadmapRequirement(
        phase="Phase 0",
        requirement="YAML/CLI/Pydantic configuration and structured logging",
        artifacts=(
            "proofrag/config.py",
            "configs/default.yaml",
            "proofrag/cli.py",
            "proofrag/logger.py",
        ),
        verification=("tests/test_config.py", "tests/test_experiment_logger.py"),
    ),
    RoadmapRequirement(
        phase="Phase 0",
        requirement="Expanded toy benchmark with at least 30 examples",
        artifacts=("benchmarks/toy_lihua.jsonl",),
        verification=("tests/test_benchmark_harness.py", "scripts/run_toy_benchmark.py"),
    ),
    RoadmapRequirement(
        phase="Phase 1",
        requirement="BM25, hybrid, reranking, vector, and iterative retrieval",
        artifacts=(
            "proofrag/retrieval/base.py",
            "proofrag/retrieval/bm25.py",
            "proofrag/retrieval/hybrid.py",
            "proofrag/retrieval/rerank.py",
            "proofrag/retrieval/vector.py",
            "proofrag/retrieval/iterative.py",
        ),
        verification=(
            "tests/test_bm25_retriever.py",
            "tests/test_hybrid_retriever.py",
            "tests/test_vector_retriever_scaffold.py",
            "tests/test_iterative_retrieval.py",
        ),
    ),
    RoadmapRequirement(
        phase="Phase 1",
        requirement="Ollama, transformers, and OpenAI-compatible generation",
        artifacts=(
            "proofrag/generation/ollama.py",
            "proofrag/generation/transformers.py",
            "proofrag/generation/openai_compatible.py",
        ),
        verification=(
            "tests/test_ollama_generator.py",
            "tests/test_transformers_generator.py",
            "tests/test_openai_compatible_generator.py",
        ),
    ),
    RoadmapRequirement(
        phase="Phase 1",
        requirement="Rule, adaptive, and LLM-assisted contract inference",
        artifacts=(
            "proofrag/contracts/infer.py",
            "proofrag/contracts/adaptive.py",
            "proofrag/contracts/llm.py",
        ),
        verification=(
            "tests/test_v06_contract_inference.py",
            "tests/test_adaptive_contracts.py",
            "tests/test_llm_contract_inference.py",
        ),
    ),
    RoadmapRequirement(
        phase="Phase 1",
        requirement="Evidence extraction and contradiction detection",
        artifacts=(
            "proofrag/evidence/extraction.py",
            "proofrag/evidence/contradiction.py",
            "proofrag/evaluation/minirag_adapter.py",
        ),
        verification=("tests/test_evidence_extraction.py", "tests/test_minirag_adapter.py"),
    ),
    RoadmapRequirement(
        phase="Phase 2",
        requirement="LiHua-World helpers and normalized MiniRAG/LightRAG export support",
        artifacts=(
            "proofrag/evaluation/lihua.py",
            "scripts/run_lihua_eval.py",
            "tools/external/minirag_exporter.py",
            "proofrag/evaluation/minirag_adapter.py",
        ),
        verification=(
            "tests/test_lihua_loader.py",
            "tests/test_publication_scripts.py",
            "tests/test_minirag_export_schema.py",
        ),
        external_blockers=("Full LiHua-World data is external.",),
    ),
    RoadmapRequirement(
        phase="Phase 2",
        requirement="Metrics, faithfulness, statistics, comparison, and error analysis",
        artifacts=(
            "proofrag/evaluation/metrics.py",
            "proofrag/evaluation/faithfulness.py",
            "proofrag/evaluation/statistics.py",
            "proofrag/evaluation/comparison.py",
            "proofrag/evaluation/error_analysis.py",
        ),
        verification=(
            "tests/test_faithfulness_metrics.py",
            "tests/test_statistics.py",
            "tests/test_comparison.py",
            "tests/test_error_analysis.py",
        ),
    ),
    RoadmapRequirement(
        phase="Phase 2",
        requirement="Tables, charts, ablations, and publication artifacts",
        artifacts=(
            "proofrag/evaluation/tables.py",
            "proofrag/evaluation/plots.py",
            "scripts/run_ablation.py",
            "scripts/make_publication_tables.py",
            "scripts/reproduce_paper_results.sh",
        ),
        verification=("tests/test_tables_plots.py", "tests/test_publication_scripts.py"),
    ),
    RoadmapRequirement(
        phase="Phase 2",
        requirement="Quantified MiniRAG superiority claims",
        artifacts=(
            "scripts/check_completion_gates.py",
            "scripts/validate_publication_claims.py",
            "scripts/write_external_evidence_manifest.py",
            "docs/results_snapshot.md",
        ),
        verification=(
            "tests/test_completion_gates.py",
            "tests/test_publication_claims.py",
            "tests/test_external_evidence_manifest.py",
        ),
        external_blockers=(
            "Full external MiniRAG/LightRAG exports are not present.",
            "Reviewed full-benchmark metrics are not present.",
        ),
    ),
    RoadmapRequirement(
        phase="Phase 3",
        requirement="Adaptive contracts, API, Docker, package metadata, and human evaluation",
        artifacts=(
            "proofrag/contracts/adaptive.py",
            "proofrag/api/main.py",
            "proofrag/api/schemas.py",
            "Dockerfile",
            "docker-compose.yml",
            "pyproject.toml",
            "proofrag/py.typed",
            "proofrag/human_eval/schema.py",
            "scripts/prepare_human_eval.py",
        ),
        verification=(
            "tests/test_adaptive_contracts.py",
            "tests/test_api.py",
            "tests/test_packaging_metadata.py",
            "tests/test_human_eval.py",
        ),
        external_blockers=("Docker image build evidence is not present.",),
    ),
    RoadmapRequirement(
        phase="Phase 4",
        requirement="Publication documentation, architecture, reproducibility, and abstract",
        artifacts=(
            "README.md",
            "docs/architecture.md",
            "docs/reproducibility.md",
            "docs/paper_abstract.md",
            "docs/figures/architecture.svg",
            "docs/completion_audit.md",
        ),
        verification=("tests/test_packaging_metadata.py",),
    ),
    RoadmapRequirement(
        phase="Cross-cutting",
        requirement="Local release evidence and CI workflow",
        artifacts=(
            "scripts/run_local_release_checks.py",
            ".github/workflows/ci.yml",
            "scripts/write_external_evidence_manifest.py",
        ),
        verification=("tests/test_release_checks.py", "tests/test_external_evidence_manifest.py"),
        external_blockers=("Remote GitHub CI run evidence is not present.",),
    ),
)


def _missing(paths: tuple[str, ...]) -> list[str]:
    return [path for path in paths if not Path(path).exists()]


def build_artifact_report() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for requirement in REQUIREMENTS:
        missing_artifacts = _missing(requirement.artifacts)
        missing_verification = _missing(requirement.verification)
        local_complete = not missing_artifacts and not missing_verification
        if not local_complete:
            status = "missing_local_artifacts"
        elif requirement.external_blockers:
            status = "externally_blocked"
        else:
            status = "locally_complete"
        rows.append(
            {
                **asdict(requirement),
                "missing_artifacts": missing_artifacts,
                "missing_verification": missing_verification,
                "local_complete": local_complete,
                "status": status,
            }
        )

    return {
        "status": (
            "passed_local_artifact_check"
            if all(row["local_complete"] for row in rows)
            else "missing_local_artifacts"
        ),
        "externally_blocked": [
            row for row in rows if row["status"] == "externally_blocked"
        ],
        "requirements": rows,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# ProofRAG Roadmap Artifact Matrix",
        "",
        "This matrix maps the development roadmap to concrete source artifacts,",
        "verification files, and known external blockers. It is generated from",
        "`scripts/check_roadmap_artifacts.py`.",
        "",
        "| Phase | Requirement | Status | Artifacts | Verification | External Blockers |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["requirements"]:
        lines.append(
            "| {phase} | {requirement} | {status} | {artifacts} | {verification} | {blockers} |".format(
                phase=row["phase"],
                requirement=row["requirement"],
                status=row["status"],
                artifacts="<br>".join(row["artifacts"]),
                verification="<br>".join(row["verification"]),
                blockers="<br>".join(row["external_blockers"]) or "(none)",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check roadmap requirements against local artifacts."
    )
    parser.add_argument("--output-json", help="Optional JSON report path.")
    parser.add_argument("--output-md", help="Optional Markdown matrix path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_artifact_report()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(f"{payload}\n", encoding="utf-8")
    if args.output_md:
        write_markdown(Path(args.output_md), report)
    print(payload)
    return 0 if report["status"] == "passed_local_artifact_check" else 1


if __name__ == "__main__":
    sys.exit(main())
