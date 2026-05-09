from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from proofrag.generation.base import BaseGenerator
from scripts.score_faithfulness import _score_row


class QueueJudge(BaseGenerator):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def generate(self, prompt: str) -> str:
        assert "Judge whether each claim" in prompt
        return self.responses.pop(0)


def test_run_lihua_eval_writes_reproducible_artifacts(tmp_path: Path):
    output = tmp_path / "proofrag_results.jsonl"
    summary = tmp_path / "summary.json"
    table = tmp_path / "table.md"
    chart = tmp_path / "chart.svg"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_lihua_eval.py",
            "--minirag-export",
            "benchmarks/sample_minirag_export.jsonl",
            "--output",
            str(output),
            "--summary-json",
            str(summary),
            "--table-md",
            str(table),
            "--chart-svg",
            str(chart),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Processed 3 LiHua/MiniRAG rows" in result.stdout
    assert len(output.read_text(encoding="utf-8").strip().splitlines()) == 3
    report = json.loads(summary.read_text(encoding="utf-8"))
    assert report["proofrag"]["method"] == "minirag+proofrag"
    assert "Precision@Answered" in table.read_text(encoding="utf-8")
    assert chart.read_text(encoding="utf-8").startswith("<svg")


def test_run_lihua_eval_can_write_source_resolution_summary(tmp_path: Path):
    qa_csv = tmp_path / "qa.csv"
    data_dir = tmp_path / "data"
    source_summary = tmp_path / "sources.json"
    qa_csv.write_text(
        "ID,Question,Gold Answer,Evidence,Type\n"
        "q1,Who asked?,Tom,doc-001<and>doc-002,single\n",
        encoding="utf-8",
    )
    data_dir.mkdir()
    (data_dir / "doc-001.txt").write_text("Tom asked LiHua.", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/run_lihua_eval.py",
            "--minirag-export",
            "benchmarks/sample_minirag_export.jsonl",
            "--output",
            str(tmp_path / "proofrag.jsonl"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--table-md",
            str(tmp_path / "table.md"),
            "--chart-svg",
            str(tmp_path / "chart.svg"),
            "--qa-csv",
            str(qa_csv),
            "--data-dir",
            str(data_dir),
            "--source-resolution-json",
            str(source_summary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    assert summary["questions"] == 1
    assert summary["resolved_sources"] == 1
    assert summary["missing_source_ids"] == ["doc-002"]


def test_ablation_and_publication_table_scripts(tmp_path: Path):
    run_jsonl = tmp_path / "run.jsonl"
    run_jsonl.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "id": "q1",
                    "baseline_method": "minirag",
                    "baseline_correct": False,
                    "answer_allowed": True,
                    "model_called": True,
                    "correct_when_answered": True,
                    "expected_answer_allowed": True,
                    "coverage_score": 1.0,
                },
                {
                    "id": "q2",
                    "baseline_method": "minirag",
                    "baseline_correct": True,
                    "answer_allowed": False,
                    "model_called": False,
                    "correct_when_answered": False,
                    "expected_answer_allowed": False,
                    "coverage_score": 0.5,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ablation_json = tmp_path / "ablation.json"
    ablation_md = tmp_path / "ablation.md"
    ablation_svg = tmp_path / "ablation.svg"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_ablation.py",
            "--run",
            f"sample={run_jsonl}",
            "--summary-json",
            str(ablation_json),
            "--table-md",
            str(ablation_md),
            "--chart-svg",
            str(ablation_svg),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    ablation = json.loads(ablation_json.read_text(encoding="utf-8"))
    assert ablation["runs"][0]["name"] == "sample"
    assert "Precision@Answered" in ablation_md.read_text(encoding="utf-8")
    assert ablation_svg.read_text(encoding="utf-8").startswith("<svg")

    comparison_json = tmp_path / "comparison.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_minirag_proofrag.py",
            "--input",
            str(run_jsonl),
            "--summary-json",
            str(comparison_json),
            "--table-md",
            str(tmp_path / "comparison.md"),
            "--chart-svg",
            str(tmp_path / "comparison.svg"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    publication_md = tmp_path / "publication.md"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_publication_tables.py",
            "--comparison-json",
            str(comparison_json),
            "--ablation-json",
            str(ablation_json),
            "--output-md",
            str(publication_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    publication = publication_md.read_text(encoding="utf-8")
    assert "MiniRAG vs MiniRAG+ProofRAG" in publication
    assert "Ablations" in publication


def test_reproduce_paper_results_script_runs_core_pipeline(tmp_path: Path):
    output_dir = tmp_path / "paper"

    result = subprocess.run(
        [
            "bash",
            "scripts/reproduce_paper_results.sh",
            "benchmarks/sample_minirag_export.jsonl",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Reproduced core ProofRAG artifacts" in result.stdout
    publication = output_dir / "publication_tables.md"
    assert publication.exists()
    assert "MiniRAG vs MiniRAG+ProofRAG" in publication.read_text(
        encoding="utf-8"
    )
    assert (output_dir / "comparison_summary.json").exists()
    assert (output_dir / "ablation_summary.json").exists()


def test_score_faithfulness_script_joins_export_evidence(tmp_path: Path):
    results = tmp_path / "results.jsonl"
    export = tmp_path / "export.jsonl"
    summary = tmp_path / "faithfulness.json"
    table = tmp_path / "faithfulness.md"

    results.write_text(
        json.dumps(
            {
                "id": "q1",
                "baseline_answer": "Tom asked LiHua. Sarah approved it.",
                "proofrag_generated_answer": "Tom asked LiHua.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    export.write_text(
        json.dumps(
            {
                "id": "q1",
                "dataset": "LiHua-World",
                "question": "Who asked LiHua?",
                "query_type": "Single",
                "gold_answer": "Tom",
                "gold_supporting_sources": ["doc-001"],
                "retrieved_context": [
                    {
                        "source_id": "doc-001",
                        "text": (
                            "-----Sources-----\n```csv\n"
                            "id,content\n"
                            "0,\"Tom asked LiHua about the warranty.\"\n"
                            "```"
                        ),
                        "metadata": {},
                    }
                ],
                "baseline_answer": "Tom asked LiHua.",
                "baseline_method": "minirag",
                "baseline_metrics": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/score_faithfulness.py",
            "--results",
            str(results),
            "--minirag-export",
            str(export),
            "--summary-json",
            str(summary),
            "--table-md",
            str(table),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["summary"]["baseline_unsupported_claims"] == 1
    assert data["summary"]["proofrag_unsupported_claims"] == 0
    assert data["scorer"] == "claim"
    assert "Mean Groundedness" in table.read_text(encoding="utf-8")


def test_score_faithfulness_row_supports_llm_judge():
    row = {
        "id": "q1",
        "baseline_answer": "Tom asked LiHua. Sarah approved it.",
        "proofrag_generated_answer": "Tom asked LiHua.",
    }
    judge = QueueJudge(
        [
            json.dumps(
                {
                    "claims": [
                        {
                            "claim": "Tom asked LiHua",
                            "supported": True,
                            "supporting_evidence_indices": [0],
                        },
                        {
                            "claim": "Sarah approved it",
                            "supported": False,
                            "supporting_evidence_indices": [],
                        },
                    ]
                }
            ),
            json.dumps(
                {
                    "claims": [
                        {
                            "claim": "Tom asked LiHua",
                            "supported": True,
                            "supporting_evidence_indices": [0],
                        }
                    ]
                }
            ),
        ]
    )

    scored = _score_row(
        row,
        ["Tom asked LiHua about the warranty."],
        scorer="llm-judge",
        judge_generator=judge,
    )

    assert scored["scorer"] == "llm-judge"
    assert scored["baseline_unsupported_claims"] == 1
    assert scored["proofrag_unsupported_claims"] == 0
