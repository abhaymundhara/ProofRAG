#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REQUIRED_SDIST_PATHS = (
    "README.md",
    "pyproject.toml",
    "MANIFEST.in",
    "Dockerfile",
    "docker-compose.yml",
    "configs/default.yaml",
    "benchmarks/toy_lihua.jsonl",
    "benchmarks/sample_minirag_export.jsonl",
    "examples/context.json",
    "docs/architecture.md",
    "docs/completion_audit.md",
    "docs/external_evidence_checklist.md",
    "docs/full_benchmark_review_template.md",
    "docs/paper_abstract.md",
    "docs/reproducibility.md",
    "docs/release_evidence_templates.md",
    "docs/results_snapshot.md",
    "docs/roadmap_artifact_matrix.md",
    "docs/figures/architecture.svg",
    "scripts/check_completion_gates.py",
    "scripts/compare_minirag_proofrag.py",
    "scripts/make_publication_tables.py",
    "scripts/run_local_release_checks.py",
    "scripts/run_lihua_eval.py",
    "scripts/run_ablation.py",
    "scripts/score_faithfulness.py",
    "scripts/summarize_experiment_log.py",
    "scripts/audit_completion_readiness.py",
    "scripts/init_external_evidence_bundle.py",
    "scripts/write_external_evidence_manifest.py",
    "scripts/write_full_benchmark_review.py",
    "scripts/reproduce_paper_results.sh",
    "scripts/check_roadmap_artifacts.py",
    "scripts/validate_publication_claims.py",
    "tools/external/minirag_exporter.py",
    "tools/external/run_minirag_tiny_query_export.py",
    "tests/test_release_checks.py",
    "tests/test_completion_readiness_audit.py",
    "tests/test_external_evidence_bundle.py",
    "proofrag/py.typed",
)

REQUIRED_WHEEL_PATHS = (
    "proofrag/cli.py",
    "proofrag/config.py",
    "proofrag/py.typed",
    "proofrag/retrieval/hybrid.py",
    "proofrag/evaluation/minirag_adapter.py",
)


@dataclass(frozen=True)
class DistributionCheck:
    artifact: str
    passed: bool
    missing: list[str]
    checked_paths: tuple[str, ...]


def _sdist_names(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
    if not names:
        return set()
    root = names[0].split("/", 1)[0]
    return {
        name[len(root) + 1 :]
        for name in names
        if name != root and name.startswith(f"{root}/")
    }


def _wheel_names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


def _find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {pattern} in {directory}, found {len(matches)}"
        )
    return matches[0]


def build_distribution_report(directory: str | Path) -> dict[str, Any]:
    dist_dir = Path(directory)
    sdist = _find_one(dist_dir, "*.tar.gz")
    wheel = _find_one(dist_dir, "*.whl")

    sdist_names = _sdist_names(sdist)
    wheel_names = _wheel_names(wheel)
    checks = [
        DistributionCheck(
            artifact=str(sdist),
            passed=all(path in sdist_names for path in REQUIRED_SDIST_PATHS),
            missing=[path for path in REQUIRED_SDIST_PATHS if path not in sdist_names],
            checked_paths=REQUIRED_SDIST_PATHS,
        ),
        DistributionCheck(
            artifact=str(wheel),
            passed=all(path in wheel_names for path in REQUIRED_WHEEL_PATHS),
            missing=[path for path in REQUIRED_WHEEL_PATHS if path not in wheel_names],
            checked_paths=REQUIRED_WHEEL_PATHS,
        ),
    ]
    return {
        "status": "passed" if all(check.passed for check in checks) else "failed",
        "checks": [asdict(check) for check in checks],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify reproducibility assets in ProofRAG distribution artifacts."
    )
    parser.add_argument("--dist-dir", required=True, help="Directory containing one sdist and one wheel.")
    parser.add_argument("--output-json", help="Optional JSON report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_distribution_report(args.dist_dir)
    except (FileNotFoundError, tarfile.TarError, zipfile.BadZipFile) as exc:
        report = {
            "status": "failed",
            "checks": [
                {
                    "artifact": str(args.dist_dir),
                    "passed": False,
                    "missing": [str(exc)],
                    "checked_paths": [],
                }
            ],
        }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(f"{payload}\n", encoding="utf-8")
    print(payload)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
