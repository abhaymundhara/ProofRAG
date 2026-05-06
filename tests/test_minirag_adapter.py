import pytest
import json
from pathlib import Path
from proofrag.evaluation.minirag_adapter import MiniRAGOutputAdapter, MiniRAGExportItem

@pytest.fixture
def sample_export_file(tmp_path):
    file_path = tmp_path / "test_export.jsonl"
    data = [
        {
            "id": "test-001",
            "dataset": "test",
            "question": "Who asked about the warranty?",
            "query_type": "factoid",
            "gold_answer": "Tom",
            "gold_supporting_sources": ["d1"],
            "retrieved_context": [
                {"source_id": "d1", "text": "Tom asked about the warranty.", "metadata": {}},
                {"source_id": "d2", "text": "The laptop has a warranty.", "metadata": {}}
            ],
            "baseline_answer": "Tom asked.",
            "baseline_method": "minirag",
            "baseline_metrics": {}
        },
        {
            "id": "test-002",
            "dataset": "test",
            "question": "Who asked about the warranty?",
            "query_type": "factoid",
            "gold_answer": "Tom",
            "gold_supporting_sources": ["d1"],
            "retrieved_context": [
                {"source_id": "d2", "text": "The laptop has a warranty.", "metadata": {}}
            ],
            "baseline_answer": "Unknown.",
            "baseline_method": "minirag",
            "baseline_metrics": {}
        }
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    return str(file_path)

def test_adapter_loading(sample_export_file):
    adapter = MiniRAGOutputAdapter()
    items = adapter.load_export(sample_export_file)
    assert len(items) == 2
    assert items[0].id == "test-001"
    assert items[1].id == "test-002"

def test_direct_evidence_allows_answer(sample_export_file):
    adapter = MiniRAGOutputAdapter()
    items = adapter.load_export(sample_export_file)
    
    result = adapter.process_item(items[0])
    assert result["sufficiency_report"]["answer_allowed"] is True
    assert "who_asked" not in result["sufficiency_report"]["missing_required_slots"]

def test_missing_evidence_blocks_answer(sample_export_file):
    adapter = MiniRAGOutputAdapter()
    items = adapter.load_export(sample_export_file)
    
    result = adapter.process_item(items[1])
    assert result["sufficiency_report"]["answer_allowed"] is False
    assert "who_asked" in result["sufficiency_report"]["missing_required_slots"]

def test_contradiction_blocks_answer(tmp_path):
    file_path = tmp_path / "test_contra.jsonl"
    item = {
        "id": "test-003",
        "dataset": "test",
        "question": "Who asked about the warranty?",
        "query_type": "factoid",
        "gold_answer": "Tom",
        "gold_supporting_sources": ["d1"],
        "retrieved_context": [
            {"source_id": "d1", "text": "Tom asked about the warranty.", "metadata": {}},
            {"source_id": "d3", "text": "No one asked about the warranty.", "metadata": {"contradiction": True}}
        ],
        "baseline_answer": "Conflict.",
        "baseline_method": "minirag",
        "baseline_metrics": {}
    }
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(item) + "\n")
    
    adapter = MiniRAGOutputAdapter()
    items = adapter.load_export(str(file_path))
    result = adapter.process_item(items[0])
    
    assert result["sufficiency_report"]["answer_allowed"] is False
    assert result["sufficiency_report"]["contradiction_count"] > 0
