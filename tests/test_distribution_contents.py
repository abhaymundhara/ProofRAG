from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

from scripts.verify_distribution_contents import build_distribution_report


def test_distribution_content_verifier_passes_for_required_assets(tmp_path: Path):
    sdist = tmp_path / "proofrag-0.1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name in [
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
            "scripts/run_full_benchmark_pipeline.py",
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
            "tools/external/run_minirag_chunk_index.py",
            "tools/external/run_minirag_tiny_query_export.py",
            "tests/test_release_checks.py",
            "tests/test_completion_readiness_audit.py",
            "tests/test_external_evidence_bundle.py",
            "proofrag/py.typed",
        ]:
            data = b"x"
            info = tarfile.TarInfo(f"proofrag-0.1.0/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    wheel = tmp_path / "proofrag-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in [
            "proofrag/cli.py",
            "proofrag/config.py",
            "proofrag/py.typed",
            "proofrag/retrieval/hybrid.py",
            "proofrag/evaluation/minirag_adapter.py",
        ]:
            archive.writestr(name, "x")

    report = build_distribution_report(tmp_path)

    assert report["status"] == "passed"
    assert all(check["passed"] for check in report["checks"])


def test_distribution_content_verifier_reports_missing_assets(tmp_path: Path):
    sdist = tmp_path / "proofrag-0.1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        data = b"x"
        info = tarfile.TarInfo("proofrag-0.1.0/README.md")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    wheel = tmp_path / "proofrag-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("proofrag/cli.py", "x")

    report = build_distribution_report(tmp_path)

    assert report["status"] == "failed"
    missing = {
        item
        for check in report["checks"]
        for item in check["missing"]
    }
    assert "docs/reproducibility.md" in missing
    assert "proofrag/config.py" in missing


def test_manifest_includes_reproducibility_assets():
    manifest = Path("MANIFEST.in").read_text(encoding="utf-8")

    assert "recursive-include docs *.md *.svg" in manifest
    assert "recursive-include scripts *.py *.sh" in manifest
    assert "recursive-include benchmarks *.jsonl" in manifest
    assert "recursive-include tools *.py *.md" in manifest


def test_distribution_verifier_cli_writes_report(tmp_path: Path):
    result_json = tmp_path / "report.json"
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "empty.tar.gz").write_bytes(b"not a tar")

    from subprocess import run
    import sys

    result = run(
        [
            sys.executable,
            "scripts/verify_distribution_contents.py",
            "--dist-dir",
            str(dist_dir),
            "--output-json",
            str(result_json),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    report = json.loads(result_json.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
