from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.audit_completion_readiness import build_audit_report


def test_completion_readiness_audit_blocks_without_external_evidence():
    report = build_audit_report()

    assert report["status"] == "blocked"
    assert report["local_artifacts_ready"] is True
    assert report["external_superiority_ready"] is False
    open_names = {item["name"] for item in report["open_items"]}
    assert "full_lihua_world_data" in open_names
    assert "normalized_baseline_export" in open_names
    assert "remote_ci_verified" in open_names


def test_completion_readiness_audit_cli_writes_outputs(tmp_path: Path):
    output_json = tmp_path / "audit.json"
    output_md = tmp_path / "audit.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_completion_readiness.py",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    markdown = output_md.read_text(encoding="utf-8")
    assert "ProofRAG Completion Readiness Audit" in markdown
    assert "External superiority ready: `False`" in markdown
