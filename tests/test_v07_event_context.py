import pytest
from proofrag.evaluation.minirag_adapter import MiniRAGOutputAdapter, MiniRAGExportItem

def test_event_context_meal_mismatch_allowed():
    """
    Test that 'dinner' question matches 'lunch' evidence if other keywords match.
    Query: 'When was the first time Li Hua had dinner with Wolfgang this year?'
    Evidence: 'WolfgangSchulz: Hey! Just wanted to remind you about our lunch at the cozy café downtown at 12:00...'
    """
    adapter = MiniRAGOutputAdapter()
    question = "When was the first time Li Hua had dinner with Wolfgang this year?"
    text = "WolfgangSchulz: Hey! Just wanted to remind you about our lunch at the cozy café downtown at 12:00. Time: 20260108_11:00"
    source_id = "20260108_11:00"
    
    item = MiniRAGExportItem(
        id="q2",
        dataset="lihua-world",
        question=question,
        query_type="temporal_query",
        gold_answer="20260108",
        gold_supporting_sources=[source_id],
        retrieved_context=[{"source_id": source_id, "text": text, "metadata": {}}],
        baseline_answer="",
        baseline_method="minirag",
        baseline_metrics={}
    )
    
    result = adapter.process_item(item)
    report = result["sufficiency_report"]
    
    # Current behavior (expected to fail before fix): event_context is missing
    # after fix: answer_allowed should be True
    assert "date_or_time_answer" in report["covered_slots"]
    assert "event_context" in report["covered_slots"]
    assert report["answer_allowed"] is True

def test_event_context_negative_timestamp_only():
    """
    Negative test: evidence only contains a timestamp but no relevant event interaction.
    Expected: event_context is not covered.
    """
    adapter = MiniRAGOutputAdapter()
    question = "When was the first time Li Hua had dinner with Wolfgang this year?"
    text = "Time: 20260108_11:00. No other content here."
    source_id = "20260108_11:00"
    
    item = MiniRAGExportItem(
        id="q2-neg1",
        dataset="lihua-world",
        question=question,
        query_type="temporal_query",
        gold_answer="20260108",
        gold_supporting_sources=[source_id],
        retrieved_context=[{"source_id": source_id, "text": text, "metadata": {}}],
        baseline_answer="",
        baseline_method="minirag",
        baseline_metrics={}
    )
    
    result = adapter.process_item(item)
    report = result["sufficiency_report"]
    
    assert "date_or_time_answer" in report["covered_slots"]
    assert "event_context" not in report["covered_slots"]
    assert report["answer_allowed"] is False

def test_event_context_negative_placeholder():
    """
    Negative test: evidence contains placeholder or dry-run wording.
    Expected: event_context is not covered.
    """
    adapter = MiniRAGOutputAdapter()
    question = "When was the first time Li Hua had dinner with Wolfgang this year?"
    text = "This is a dry-run placeholder context for Wolfgang and Li Hua."
    source_id = "20260108_11:00"
    
    item = MiniRAGExportItem(
        id="q2-neg2",
        dataset="lihua-world",
        question=question,
        query_type="temporal_query",
        gold_answer="20260108",
        gold_supporting_sources=[source_id],
        retrieved_context=[{"source_id": source_id, "text": text, "metadata": {}}],
        baseline_answer="",
        baseline_method="minirag",
        baseline_metrics={}
    )
    
    result = adapter.process_item(item)
    report = result["sufficiency_report"]
    
    assert "event_context" not in report["covered_slots"]
    assert report["answer_allowed"] is False
