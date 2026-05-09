from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.check_roadmap_artifacts import build_artifact_report, write_markdown


def test_roadmap_artifact_report_maps_all_phases():
    report = build_artifact_report()

    assert report["status"] == "passed_local_artifact_check"
    phases = {row["phase"] for row in report["requirements"]}
    assert {"Phase 0", "Phase 1", "Phase 2", "Phase 3", "Phase 4"} <= phases
    assert all(row["local_complete"] for row in report["requirements"])
    blocked = {row["requirement"] for row in report["externally_blocked"]}
    assert "Quantified MiniRAG superiority claims" in blocked
    assert "Local release evidence and CI workflow" in blocked
    release = next(
        row
        for row in report["requirements"]
        if row["requirement"] == "Local release evidence and CI workflow"
    )
    assert "docs/release_evidence_templates.md" in release["artifacts"]
    quantified = next(
        row
        for row in report["requirements"]
        if row["requirement"] == "Quantified MiniRAG superiority claims"
    )
    assert "scripts/audit_completion_readiness.py" in quantified["artifacts"]
    assert "scripts/init_external_evidence_bundle.py" in quantified["artifacts"]
    assert "scripts/write_external_evidence_manifest.py" in quantified["artifacts"]
    assert "docs/external_evidence_checklist.md" in quantified["artifacts"]
    assert "docs/full_benchmark_review_template.md" in quantified["artifacts"]
    assert "tests/test_completion_readiness_audit.py" in quantified["verification"]
    assert "tests/test_external_evidence_manifest.py" in quantified["verification"]
    assert "tests/test_external_evidence_bundle.py" in quantified["verification"]


def test_roadmap_artifact_cli_writes_json_and_markdown(tmp_path: Path):
    output_json = tmp_path / "roadmap.json"
    output_md = tmp_path / "roadmap.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_roadmap_artifacts.py",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "passed_local_artifact_check" in result.stdout
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["requirements"]
    markdown = output_md.read_text(encoding="utf-8")
    assert "ProofRAG Roadmap Artifact Matrix" in markdown
    assert "Quantified MiniRAG superiority claims" in markdown
    assert "scripts/write_external_evidence_manifest.py" in markdown
    assert "scripts/audit_completion_readiness.py" in markdown
    assert "scripts/init_external_evidence_bundle.py" in markdown
    assert "docs/external_evidence_checklist.md" in markdown
    assert "docs/full_benchmark_review_template.md" in markdown
    assert "docs/release_evidence_templates.md" in markdown


def test_roadmap_markdown_writer_includes_external_blockers(tmp_path: Path):
    report = build_artifact_report()
    output_md = tmp_path / "matrix.md"

    write_markdown(output_md, report)

    text = output_md.read_text(encoding="utf-8")
    assert "Full external MiniRAG/LightRAG exports are not present." in text
    assert "Docker image build evidence is not present." in text
