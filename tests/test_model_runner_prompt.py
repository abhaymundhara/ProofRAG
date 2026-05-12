import json

from scripts.run_proofrag_over_minirag_with_model import (
    _append_concise_answer_policy,
    _drop_error_rows,
    _load_completed_ids,
)


def test_append_concise_answer_policy_requests_short_answer():
    prompt = _append_concise_answer_policy(
        "## QUESTION\nWhat happened?",
        question="What happened?",
    )

    assert "## CONCISE ANSWER FORMAT" in prompt
    assert "do not answer Yes or No" in prompt
    assert "one short sentence" in prompt
    assert "Insufficient evidence" in prompt


def test_append_concise_answer_policy_shapes_yes_no_answers():
    prompt = _append_concise_answer_policy(
        "## QUESTION\nDid Li Hua ask Adam?",
        question="Did Li Hua ask Adam?",
    )

    assert 'start with exactly "Yes," or "No,"' in prompt
    assert "consistent with that label" in prompt


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


def test_drop_error_rows_removes_retryable_failures(tmp_path):
    output = tmp_path / "results.jsonl"
    output.write_text(
        "\n".join(
            [
                json.dumps({"id": "q1", "proofrag_generated_answer": "Yes"}),
                json.dumps({"id": "q2", "proofrag_generated_answer": "ERROR: timed out"}),
                "{bad json",
            ]
        ),
        encoding="utf-8",
    )

    _drop_error_rows(output)

    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == [{"id": "q1", "proofrag_generated_answer": "Yes"}]
