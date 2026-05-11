from proofrag.evaluation.source_coverage import (
    expected_answer_allowed_from_sources,
    retrieved_source_ids,
    source_ids_match,
)


def test_retrieved_source_ids_include_nested_minirag_rows():
    retrieved = [{"source_id": "minirag-retrieval"}]
    records = [{"source_id": "minirag-retrieval#src20260108_11:00"}]

    assert retrieved_source_ids(retrieved, records) == {
        "minirag-retrieval",
        "minirag-retrieval#src20260108_11:00",
        "20260108_11:00",
    }


def test_source_ids_match_timestamp_delimiter_variants():
    assert source_ids_match("20260108_11:00", "202601081100")
    assert source_ids_match("20260108_11:00", "20260108-11:00")


def test_expected_answer_allowed_requires_gold_coverage_without_contradiction():
    retrieved = [{"source_id": "ctx"}]
    records = [
        {
            "source_id": "ctx#src20260108_11:00",
            "contradicts": [],
        }
    ]

    assert expected_answer_allowed_from_sources(
        gold_supporting_sources=["20260108_11:00"],
        retrieved_context=retrieved,
        evidence_records=records,
    )
    assert not expected_answer_allowed_from_sources(
        gold_supporting_sources=["20260109_11:00"],
        retrieved_context=retrieved,
        evidence_records=records,
    )
    assert not expected_answer_allowed_from_sources(
        gold_supporting_sources=["20260108_11:00"],
        retrieved_context=retrieved,
        evidence_records=[{**records[0], "contradicts": ["answer"]}],
    )
