from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.check_completion_gates import build_completion_report


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "lihua_qa_csv": None,
        "lihua_data_dir": None,
        "min_lihua_qa_rows": 100,
        "minirag_export": None,
        "min_baseline_export_rows": 100,
        "comparison_summary": None,
        "faithfulness_summary": None,
        "review_note": None,
        "claim_min_total": 100,
        "claim_max_accuracy_drop": 0.05,
        "claim_min_precision_at_answered": 0.75,
        "claim_max_unsafe_allow_rate": 0.0,
        "claim_min_groundedness_delta": 0.10,
        "claim_max_unsupported_claim_ratio": 0.75,
        "require_claim_significance": False,
        "claim_alpha": 0.05,
        "docker_evidence": None,
        "docker_build_tag": "proofrag:test",
        "docker_build_context": ".",
        "ci_evidence": None,
        "ci_url": None,
        "check_docker_daemon": False,
        "output_json": str(tmp_path / "report.json"),
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_completion_gates_fail_closed_without_external_evidence(tmp_path: Path):
    report = build_completion_report(_args(tmp_path))

    assert report["ready_for_superiority_claim"] is False
    assert report["status"] == "blocked"
    assert {gate["name"] for gate in report["gates"]} == {
        "full_lihua_world_data",
        "normalized_baseline_export",
        "reviewed_result_artifacts",
        "docker_build_verified",
        "remote_ci_verified",
    }
    assert all(gate["passed"] is False for gate in report["gates"])


def test_completion_gates_pass_with_concrete_artifacts(tmp_path: Path):
    qa_csv = tmp_path / "query_set.csv"
    qa_csv.write_text("Question,Answer,Evidence\nWho asked LiHua?,Tom,doc-1\n", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "doc-1.txt").write_text("Tom asked LiHua.\n", encoding="utf-8")
    export = tmp_path / "export.jsonl"
    export.write_text(Path("benchmarks/sample_minirag_export.jsonl").read_text(), encoding="utf-8")
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "baseline": {"total": 3, "accuracy": 0.67},
                "proofrag": {
                    "total": 3,
                    "accuracy": 0.67,
                    "precision_at_answered": 1.0,
                    "unsafe_allow_count": 0,
                },
                "paired_answer_accuracy": {"exact_p_value": 1.0},
            }
        ),
        encoding="utf-8",
    )
    faithfulness = tmp_path / "faithfulness.json"
    faithfulness.write_text(
        json.dumps(
            {
                "summary": {
                    "total": 3,
                    "baseline_mean_groundedness": 0.25,
                    "proofrag_mean_groundedness": 0.5,
                    "baseline_unsupported_claims": 4,
                    "proofrag_unsupported_claims": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    review = tmp_path / "review.md"
    review.write_text("Reviewed full external benchmark artifacts.\n", encoding="utf-8")
    docker_evidence = tmp_path / "docker.txt"
    docker_evidence.write_text("docker build succeeded\n", encoding="utf-8")
    ci_evidence = tmp_path / "ci.txt"
    ci_evidence.write_text("GitHub Actions conclusion: success\n", encoding="utf-8")

    report = build_completion_report(
        _args(
            tmp_path,
            lihua_qa_csv=str(qa_csv),
            lihua_data_dir=str(data_dir),
            min_lihua_qa_rows=1,
            minirag_export=str(export),
            min_baseline_export_rows=3,
            comparison_summary=str(comparison),
            faithfulness_summary=str(faithfulness),
            review_note=str(review),
            claim_min_total=1,
            docker_evidence=str(docker_evidence),
            ci_evidence=str(ci_evidence),
            ci_url="https://github.com/example/proofrag/actions/runs/1",
        )
    )

    assert report["ready_for_superiority_claim"] is True
    assert report["status"] == "ready"
    assert all(gate["passed"] is True for gate in report["gates"])


def test_completion_gates_reject_vague_docker_evidence(tmp_path: Path):
    evidence = tmp_path / "docker.txt"
    evidence.write_text("image exists\n", encoding="utf-8")

    report = build_completion_report(_args(tmp_path, docker_evidence=str(evidence)))

    gate = next(
        item for item in report["gates"] if item["name"] == "docker_build_verified"
    )
    assert gate["passed"] is False
    assert "docker build" in gate["detail"]


def test_completion_gates_reject_non_actions_ci_url(tmp_path: Path):
    report = build_completion_report(
        _args(tmp_path, ci_url="https://github.com/example/proofrag")
    )

    gate = next(
        item for item in report["gates"] if item["name"] == "remote_ci_verified"
    )
    assert gate["passed"] is False
    assert "GitHub Actions run URL" in gate["detail"]


def test_completion_gates_accept_successful_ci_evidence_file(tmp_path: Path):
    evidence = tmp_path / "ci.txt"
    evidence.write_text("GitHub Actions conclusion: success\n", encoding="utf-8")

    report = build_completion_report(_args(tmp_path, ci_evidence=str(evidence)))

    gate = next(
        item for item in report["gates"] if item["name"] == "remote_ci_verified"
    )
    assert gate["passed"] is True


def test_completion_gates_reject_actions_url_without_success_evidence(tmp_path: Path):
    report = build_completion_report(
        _args(tmp_path, ci_url="https://github.com/example/proofrag/actions/runs/1")
    )

    gate = next(
        item for item in report["gates"] if item["name"] == "remote_ci_verified"
    )
    assert gate["passed"] is False
    assert "URL alone does not prove" in gate["detail"]


def test_completion_gates_reject_tiny_lihua_fixture_as_full_data(tmp_path: Path):
    qa_csv = tmp_path / "query_set.csv"
    qa_csv.write_text("Question,Answer,Evidence\nWho asked LiHua?,Tom,doc-1\n", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "doc-1.txt").write_text("Tom asked LiHua.\n", encoding="utf-8")

    report = build_completion_report(
        _args(
            tmp_path,
            lihua_qa_csv=str(qa_csv),
            lihua_data_dir=str(data_dir),
            min_lihua_qa_rows=2,
        )
    )

    gate = next(
        item for item in report["gates"] if item["name"] == "full_lihua_world_data"
    )
    assert gate["passed"] is False
    assert "expected at least 2" in gate["detail"]


def test_completion_gates_reject_tiny_baseline_export(tmp_path: Path):
    export = tmp_path / "export.jsonl"
    export.write_text(Path("benchmarks/sample_minirag_export.jsonl").read_text(), encoding="utf-8")

    report = build_completion_report(
        _args(tmp_path, minirag_export=str(export), min_baseline_export_rows=4)
    )

    gate = next(
        item for item in report["gates"] if item["name"] == "normalized_baseline_export"
    )
    assert gate["passed"] is False
    assert "expected at least 4" in gate["detail"]


def test_completion_gates_reject_malformed_result_artifacts(tmp_path: Path):
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({"rows": 3}), encoding="utf-8")
    faithfulness = tmp_path / "faithfulness.json"
    faithfulness.write_text(json.dumps({"rows": 3}), encoding="utf-8")
    review = tmp_path / "review.md"
    review.write_text("Reviewed full external benchmark artifacts.\n", encoding="utf-8")

    report = build_completion_report(
        _args(
            tmp_path,
            comparison_summary=str(comparison),
            faithfulness_summary=str(faithfulness),
            review_note=str(review),
        )
    )

    gate = next(
        item for item in report["gates"] if item["name"] == "reviewed_result_artifacts"
    )
    assert gate["passed"] is False
    assert "failed validation" in gate["detail"]


def test_completion_gates_reject_unreviewed_result_note(tmp_path: Path):
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "baseline": {"total": 1, "accuracy": 0.0},
                "proofrag": {
                    "total": 1,
                    "accuracy": 1.0,
                    "precision_at_answered": 1.0,
                    "unsafe_allow_count": 0,
                },
                "paired_answer_accuracy": {"exact_p_value": 1.0},
            }
        ),
        encoding="utf-8",
    )
    faithfulness = tmp_path / "faithfulness.json"
    faithfulness.write_text(
        json.dumps(
            {
                "summary": {
                    "total": 1,
                    "baseline_mean_groundedness": 0.0,
                    "proofrag_mean_groundedness": 1.0,
                    "baseline_unsupported_claims": 1,
                    "proofrag_unsupported_claims": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    note = tmp_path / "note.md"
    note.write_text("draft metrics\n", encoding="utf-8")

    report = build_completion_report(
        _args(
            tmp_path,
            comparison_summary=str(comparison),
            faithfulness_summary=str(faithfulness),
            review_note=str(note),
        )
    )

    gate = next(
        item for item in report["gates"] if item["name"] == "reviewed_result_artifacts"
    )
    assert gate["passed"] is False
    assert "review note" in gate["detail"]


def test_completion_gates_reject_schema_valid_but_weak_claims(tmp_path: Path):
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "baseline": {"total": 10, "accuracy": 0.9},
                "proofrag": {
                    "total": 10,
                    "accuracy": 0.7,
                    "precision_at_answered": 0.6,
                    "unsafe_allow_count": 2,
                },
                "paired_answer_accuracy": {"exact_p_value": 0.5},
            }
        ),
        encoding="utf-8",
    )
    faithfulness = tmp_path / "faithfulness.json"
    faithfulness.write_text(
        json.dumps(
            {
                "summary": {
                    "total": 10,
                    "baseline_mean_groundedness": 0.5,
                    "proofrag_mean_groundedness": 0.52,
                    "baseline_unsupported_claims": 4,
                    "proofrag_unsupported_claims": 4,
                }
            }
        ),
        encoding="utf-8",
    )
    review = tmp_path / "review.md"
    review.write_text("Reviewed full external benchmark artifacts.\n", encoding="utf-8")

    report = build_completion_report(
        _args(
            tmp_path,
            comparison_summary=str(comparison),
            faithfulness_summary=str(faithfulness),
            review_note=str(review),
            claim_min_total=10,
        )
    )

    gate = next(
        item for item in report["gates"] if item["name"] == "reviewed_result_artifacts"
    )
    assert gate["passed"] is False
    assert "publication claim thresholds failed" in gate["detail"]


def test_completion_gate_cli_exits_nonzero_when_blocked(tmp_path: Path):
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_completion_gates.py",
            "--output-json",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert output.exists()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"


def test_completion_gate_runs_real_docker_build_when_requested(
    tmp_path: Path, monkeypatch
):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="Successfully built\n", stderr="")

    monkeypatch.setattr("scripts.check_completion_gates.subprocess.run", fake_run)

    report = build_completion_report(
        _args(tmp_path, check_docker_daemon=True, docker_build_context=str(tmp_path))
    )

    docker_gate = next(
        gate for gate in report["gates"] if gate["name"] == "docker_build_verified"
    )
    assert docker_gate["passed"] is True
    assert calls == [["docker", "build", "-t", "proofrag:test", str(tmp_path)]]


def test_completion_gate_reports_missing_docker_context(tmp_path: Path):
    report = build_completion_report(
        _args(
            tmp_path,
            check_docker_daemon=True,
            docker_build_context=str(tmp_path / "missing"),
        )
    )

    docker_gate = next(
        gate for gate in report["gates"] if gate["name"] == "docker_build_verified"
    )
    assert docker_gate["passed"] is False
    assert "context does not exist" in docker_gate["detail"]
