from proofrag.evaluation.answer_metrics import (
    contains_gold_answer,
    is_answer_correct,
    normalize_answer,
)

def test_normalize_answer():
    assert normalize_answer("Hello, World!") == "hello world"
    assert normalize_answer("  Multiple   Spaces  ") == "multiple spaces"
    assert normalize_answer("Punctuation... Check???") == "punctuation check"
    assert normalize_answer("") == ""
    assert normalize_answer(None) == ""

def test_contains_gold_answer():
    assert contains_gold_answer("The answer is Tom.", "Tom")
    assert contains_gold_answer("Tom asked LiHua.", "tom")
    assert not contains_gold_answer("The meeting was on Monday.", "Tuesday")
    assert not contains_gold_answer("I don't know.", "Tom")
    assert contains_gold_answer("Tom", "Tom")
    assert contains_gold_answer("The person was Tom Smith.", "Tom")
    assert contains_gold_answer("Tom Smith", "Tom Smith")
    assert contains_gold_answer("Li Hua asked at 20260206 at 16:00.", "20260206_16:00")

def test_water_tap_typo_normalization():
    # 'water tap in the apartment is broken' should match gold 'water tab in the apartment is broken'
    assert is_answer_correct(
        "the water tap in the apartment is broken",
        "the water tab in the apartment is broken",
    )

def test_date_variants_normalization():
    # 'January 8, 2026' should match gold '20260108'
    assert is_answer_correct("January 8, 2026", "20260108")
    assert is_answer_correct("Jan 8, 2026", "20260108")
    assert is_answer_correct("8 January 2026", "20260108")
    assert is_answer_correct("The event is on 20260108.", "20260108")
    assert is_answer_correct("The event is on 20260108_11:00.", "20260108")
    assert is_answer_correct("The event is on 20260108_1100.", "20260108")
    assert is_answer_correct("The event is on 20260108 at 11:00.", "20260108_11:00")
    assert is_answer_correct("Li Hua plans to celebrate on Jan 18th.", "20260118")


def test_answer_prefix_and_yes_no_normalization():
    assert is_answer_correct("Yes. Evidence [1] supports it.", "Answer: Yes")
    assert is_answer_correct("No, the ordering is reversed.", "No")
    assert not is_answer_correct("No, the ordering is reversed.", "Answer: Yes")


def test_ampersand_and_negated_damage_paraphrases():
    assert is_answer_correct(
        "Jennifer suggests hydration, active recovery, stretching, foam rolling, and rest.",
        "hydration&active recovery&stretching&foam rolling&rest",
    )
    assert is_answer_correct(
        "It is allowed as long as it is reversible and doesn't damage anything.",
        "must be reversible and not damage anything",
    )

def test_source_id_isolation():
    # An answer that only contains 'record_id=minirag-lihua-single-0002-0-src0' should not count as correct for date gold '20260108'
    generated = "[record_id=minirag-lihua-single-0002-0-src0]"
    assert not is_answer_correct(generated, "20260108", source_ids=["20260108"])

    generated2 = "Source 20260108_11:00 says hello."
    assert not is_answer_correct(
        generated2,
        "20260108",
        source_ids=["20260108_11:00"],
    )

    generated2b = "The answer is 20260108_11:00."
    assert is_answer_correct(
        generated2b,
        "20260108",
        source_ids=["20260108_11:00"],
    )

    # But if the natural date is in the answer, it should be correct
    generated3 = "Source 20260108_11:00 says the date is 20260108."
    assert is_answer_correct(generated3, "20260108", source_ids=["20260108_11:00"])
