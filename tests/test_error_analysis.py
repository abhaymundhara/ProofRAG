from __future__ import annotations

from proofrag.evaluation.error_analysis import analyze_errors, classify_result


def test_classify_result_for_expected_allow_mismatches():
    assert (
        classify_result(
            {
                "answer_allowed": True,
                "expected_answer_allowed": False,
            }
        )
        == "unsafe_false_allow"
    )
    assert (
        classify_result(
            {
                "answer_allowed": False,
                "expected_answer_allowed": True,
            }
        )
        == "false_abstain"
    )


def test_classify_result_for_block_reasons_and_answer_quality():
    assert classify_result({"answer_allowed": False, "contradiction_count": 1}) == (
        "blocked_contradiction"
    )
    assert classify_result({"answer_allowed": False, "missing_required_slots": ["answer"]}) == (
        "blocked_missing_evidence"
    )
    assert classify_result({"answer_allowed": True, "model_called": False}) == (
        "allowed_but_model_not_called"
    )
    assert classify_result({"answer_allowed": True, "model_called": True, "answer_correct": True}) == (
        "correct"
    )
    assert classify_result(
        {
            "answer_allowed": True,
            "model_called": True,
            "contains_gold_answer_raw": True,
        }
    ) == "gold_present_but_semantically_wrong"


def test_analyze_errors_groups_ids_deterministically():
    report = analyze_errors(
        [
            {"id": "a", "answer_allowed": False, "missing_required_slots": ["x"]},
            {"id": "b", "answer_allowed": False, "contradiction_count": 1},
            {"id": "c", "answer_allowed": False, "missing_required_slots": ["y"]},
        ]
    )

    buckets = {bucket.label: bucket for bucket in report.buckets}
    assert report.total == 3
    assert buckets["blocked_missing_evidence"].count == 2
    assert buckets["blocked_missing_evidence"].ids == ["a", "c"]
    assert buckets["blocked_contradiction"].ids == ["b"]

