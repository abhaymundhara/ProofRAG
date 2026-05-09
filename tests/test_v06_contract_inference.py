from proofrag.contracts.infer import infer_contract_from_question
from proofrag.evaluation.minirag_adapter import MiniRAGOutputAdapter

def test_contract_inference_rules():
    # What time
    c1 = infer_contract_from_question("What time does Li Hua check in?")
    assert "time_answer" in [s.slot_id for s in c1.slots]
    assert "event_context" in [s.slot_id for s in c1.slots]
    
    # When
    c2 = infer_contract_from_question("When was the first time Li Hua had dinner?")
    assert "date_or_time_answer" in [s.slot_id for s in c2.slots]
    assert "event_context" in [s.slot_id for s in c2.slots]
    
    # Who
    c3 = infer_contract_from_question("Who asked LiHua about the move?")
    assert "who_asked" in [s.slot_id for s in c3.slots]
    assert "topic_context" in [s.slot_id for s in c3.slots]
    
    # Other
    c4 = infer_contract_from_question("Is the place nice?")
    assert "answer" in [s.slot_id for s in c4.slots]
    assert "topic_context" in [s.slot_id for s in c4.slots]

def test_minirag_adapter_evidence_mapping():
    adapter = MiniRAGOutputAdapter()
    
    # 1. Time query
    q1 = "What time does Li Hua check in?"
    text1 = "LiHua: I'll be there around 5:30 PM then."
    inf1 = adapter._infer_evidence(text1, q1, {})
    assert "time_answer" in inf1["supports_slots"]
    assert inf1["evidence_strength"] == "direct"
    
    # 2. Date query
    q2 = "When was the dinner?"
    text2 = "On 20260108, we had dinner."
    inf2 = adapter._infer_evidence(text2, q2, {}, source_id="20260108_1100")
    assert "date_or_time_answer" in inf2["supports_slots"]
    assert "event_context" in inf2["supports_slots"]
    
    # 3. Strict mode: lunch for dinner
    q3 = "When was the dinner?"
    text3 = "We had lunch at 12:00."
    inf3 = adapter._infer_evidence(text3, q3, {})
    # Should NOT have event_context because dinner vs lunch
    assert "event_context" not in inf3["supports_slots"]

def test_source_id_date_detection():
    adapter = MiniRAGOutputAdapter()
    q = "When was it?"
    text = "Nothing here."
    inf = adapter._infer_evidence(text, q, {}, source_id="20260105_1400")
    assert "date_or_time_answer" in inf["supports_slots"]

def test_is_answer_correct_logic():
    from proofrag.evaluation.answer_metrics import is_answer_correct
    
    # 1. Correct
    assert is_answer_correct("The dinner was on 20260108.", "20260108") is True
    
    # 2. Block phrases
    assert is_answer_correct("The answer is incomplete. Source says 20260108.", "20260108") is False
    assert is_answer_correct("Insufficient evidence to confirm 20260108.", "20260108") is False
    
    # 3. Source ID leakage
    # Gold is part of source ID, but not in text as an answer
    assert is_answer_correct("See source 20260108_1100.", "20260108", source_ids=["20260108_1100"]) is False
    # Gold is standalone in text, even if source ID is present
    assert is_answer_correct("The date is 20260108. (Source: 20260108_1100)", "20260108", source_ids=["20260108_1100"]) is True

def test_placeholder_context_inference():
    adapter = MiniRAGOutputAdapter()
    q = "When was it?"
    text = "Dry-run placeholder context. Real MiniRAG retrieval was not executed."
    inf = adapter._infer_evidence(text, q, {})
    assert inf["supports_slots"] == []
    assert inf["evidence_strength"] == "background"
