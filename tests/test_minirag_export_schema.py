import json
import pytest
from pathlib import Path
from tools.external.minirag_exporter import validate_minirag_export_row, run_export
from proofrag.evaluation.minirag_adapter import MiniRAGOutputAdapter

def test_validate_good_row():
    row = {
        "id": "q1",
        "dataset": "ds",
        "question": "Q?",
        "query_type": "type",
        "gold_answer": "A",
        "gold_supporting_sources": ["s1"],
        "retrieved_context": [{"source_id": "c1", "text": "context"}],
        "baseline_answer": "B",
        "baseline_method": "minirag",
        "baseline_metrics": {},
    }
    # Should not raise
    validate_minirag_export_row(row)

def test_validate_bad_row():
    row = {"id": "q1"}
    with pytest.raises(KeyError):
        validate_minirag_export_row(row)
    
    row = {
        "id": "q1",
        "dataset": "ds",
        "question": "Q?",
        "query_type": "type",
        "gold_answer": "A",
        "gold_supporting_sources": "not-a-list",
        "retrieved_context": [],
        "baseline_answer": "B",
        "baseline_method": "minirag",
        "baseline_metrics": {},
    }
    with pytest.raises(TypeError):
        validate_minirag_export_row(row)

def test_dry_run_export_compatibility(tmp_path):
    # 1. Create a tiny temp QA file
    qa_file = tmp_path / "tiny_qa.jsonl"
    qa_data = {
        "id": "t1",
        "question": "Who asked about the laptop warranty?",
        "gold_answer": "Tom",
        "query_type": "factoid"
    }
    qa_file.write_text(json.dumps(qa_data) + "\n")
    
    output_file = tmp_path / "dry_run_output.jsonl"
    
    # 2. Run dry-run export
    run_export(
        minirag_root="none",
        working_dir="none",
        qa_file=str(qa_file),
        output_file=str(output_file),
        dry_run=True
    )
    
    # 3. Confirm output exists and is valid
    assert output_file.exists()
    with open(output_file, "r") as f:
        exported_data = json.loads(f.read())
        validate_minirag_export_row(exported_data)
        assert "Dry-run context" in exported_data["retrieved_context"][0]["text"]

    # 4. Confirm MiniRAGOutputAdapter can load it
    adapter = MiniRAGOutputAdapter()
    items = adapter.load_export(str(output_file))
    assert len(items) == 1
    
    # 5. Confirm process_item works (ProofRAG pipeline)
    result = adapter.process_item(items[0])
    assert result["id"] == "t1"
    assert "sufficiency_report" in result
