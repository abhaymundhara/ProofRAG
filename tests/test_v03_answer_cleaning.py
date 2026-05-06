from proofrag.evaluation.answer_metrics import clean_model_answer, contains_gold_answer

def test_clean_model_answer_gemma_style():
    # Thought and channel tags
    raw = "thought\nThe user wants to know who asked...\nAnswer formulation...<channel|>Tom and Sarah asked."
    assert clean_model_answer(raw) == "Tom and Sarah asked."
    
    # Just thought prefix
    raw = "thought\nTom asked."
    assert clean_model_answer(raw) == "Tom asked."
    
    # Normal text
    raw = "Tom asked."
    assert clean_model_answer(raw) == "Tom asked."
    
    # Multiple channel tags (take last)
    raw = "thought...<channel|>Intermediate...<channel|>Final Answer."
    assert clean_model_answer(raw) == "Final Answer."

def test_contains_gold_answer_after_cleaning():
    # Gemma style output with thought/channel and citations
    generated = "thought\nAnalyze...<channel|>Tom [record_id=rec-0] and Sarah [record_id=rec-1] asked."
    gold = "Tom and Sarah"
    assert contains_gold_answer(generated, gold) is True
    
def test_clean_model_answer_whitespace():
    raw = "  \n Tom asked. \t "
    assert clean_model_answer(raw) == "Tom asked."
