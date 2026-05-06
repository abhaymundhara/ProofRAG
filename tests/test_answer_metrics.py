from proofrag.evaluation.answer_metrics import normalize_answer, contains_gold_answer

def test_normalize_answer():
    assert normalize_answer("Hello, World!") == "hello world"
    assert normalize_answer("  Multiple   Spaces  ") == "multiple spaces"
    assert normalize_answer("Punctuation... Check???") == "punctuation check"
    assert normalize_answer("") == ""
    assert normalize_answer(None) == ""

def test_contains_gold_answer():
    assert contains_gold_answer("The answer is Tom.", "Tom") == True
    assert contains_gold_answer("Tom asked LiHua.", "tom") == True
    assert contains_gold_answer("The meeting was on Monday.", "Tuesday") == False
    assert contains_gold_answer("I don't know.", "Tom") == False
    assert contains_gold_answer("Tom", "Tom") == True
    assert contains_gold_answer("The person was Tom Smith.", "Tom") == True
    assert contains_gold_answer("Tom Smith", "Tom Smith") == True
