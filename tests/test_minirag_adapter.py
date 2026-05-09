import json
import pytest

from proofrag.evaluation.minirag_adapter import (
    LightRAGOutputAdapter,
    MiniRAGExportItem,
    MiniRAGOutputAdapter,
)

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

def test_extract_minirag_source_rows():
    adapter = MiniRAGOutputAdapter()
    text = '''
-----Entities-----
```csv
entity,score,description
"E1",1.0,"desc"
```
-----Sources-----
```csv
id,content
0,"Time: 20260106_15:00
AdamSmith: Hey Li Hua"
1,"Time: 20260106_13:00
LiHua: the water tab in the apartment is broken"
```
'''
    rows = adapter.extract_source_rows(text)
    assert len(rows) == 2
    assert rows[0]["id"] == "0"
    assert "Time: 20260106_15:00" in rows[0]["content"]
    assert rows[1]["id"] == "1"
    assert "water tab" in rows[1]["content"]

def test_lihua_single_0008_regression():
    adapter = MiniRAGOutputAdapter()
    text = '''
-----Sources-----
```csv
id,content
0,"Time: 20260106_15:00
AdamSmith: plumber arrives tomorrow at 10 AM"
3,"Time: 20260106_13:00
LiHua: just wanted to let you know that the water tab in the apartment is broken."
```
'''
    item = MiniRAGExportItem(
        id="lihua-single-0008",
        dataset="test",
        question="What does Li Hua report to Adam on January 6th?",
        query_type="factoid",
        gold_answer="the water tab in the apartment is broken",
        gold_supporting_sources=["d1"],
        retrieved_context=[{"source_id": "minirag-retrieval", "text": text, "metadata": {}}],
        baseline_answer="",
        baseline_method="minirag",
        baseline_metrics={}
    )
    result = adapter.process_item(item)
    
    report = result["sufficiency_report"]
    records = result["evidence_records"]
    
    assert len(records) == 2
    assert report["answer_allowed"] is True
    
    # Check that record for source 3 exists and supports slots
    water_tab_record = next(r for r in records if "water tab" in r["text"])
    assert "answer" in water_tab_record["supports_slots"]
    assert "topic_context" in water_tab_record["supports_slots"]
    assert water_tab_record["evidence_strength"] == "direct"
    
    # Check that record for source 0 exists but doesn't necessarily support the required slots directly
    plumber_record = next(r for r in records if "plumber" in r["text"])
    assert "answer" not in plumber_record["supports_slots"]


def test_lightrag_adapter_consumes_same_normalized_export(tmp_path):
    file_path = tmp_path / "lightrag.jsonl"
    row = {
        "id": "light-001",
        "dataset": "test",
        "question": "Who asked about the warranty?",
        "query_type": "factoid",
        "gold_answer": "Tom",
        "gold_supporting_sources": ["d1"],
        "retrieved_context": [
            {
                "source_id": "d1",
                "text": "Tom asked about the warranty.",
                "metadata": {},
            }
        ],
        "baseline_answer": "Tom asked.",
        "baseline_method": "lightrag",
        "baseline_metrics": {},
    }
    file_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    adapter = LightRAGOutputAdapter()
    item = adapter.load_export(str(file_path))[0]
    result = adapter.process_item(item)

    assert item.baseline_method == "lightrag"
    assert result["baseline_method"] == "lightrag"
    assert result["sufficiency_report"]["answer_allowed"] is True
