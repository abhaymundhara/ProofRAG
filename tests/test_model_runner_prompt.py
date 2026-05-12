from scripts.run_proofrag_over_minirag_with_model import _append_concise_answer_policy


def test_append_concise_answer_policy_requests_short_answer():
    prompt = _append_concise_answer_policy("## QUESTION\nWhat happened?")

    assert "## CONCISE ANSWER FORMAT" in prompt
    assert "one short sentence" in prompt
    assert "Insufficient evidence" in prompt
