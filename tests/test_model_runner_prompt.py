import json

from scripts.run_proofrag_over_minirag_with_model import (
    _append_concise_answer_policy,
    _load_completed_ids,
)


def test_append_concise_answer_policy_requests_short_answer():
    prompt = _append_concise_answer_policy("## QUESTION\nWhat happened?")

    assert "## CONCISE ANSWER FORMAT" in prompt
    assert "one short sentence" in prompt
    assert "Insufficient evidence" in prompt


def test_load_completed_ids_ignores_bad_lines(tmp_path):
    output = tmp_path / "results.jsonl"
    output.write_text(
        "\n".join(
            [
                json.dumps({"id": "q1"}),
                "{bad json",
                json.dumps({"id": "q2"}),
                json.dumps({"id": ""}),
            ]
        ),
        encoding="utf-8",
    )

    assert _load_completed_ids(output) == {"q1", "q2"}
