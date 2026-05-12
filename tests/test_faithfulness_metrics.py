from __future__ import annotations

import json

from proofrag.evaluation.faithfulness import (
    claim_level_faithfulness,
    extract_claims,
    judge_faithfulness_with_llm,
)
from proofrag.generation.base import BaseGenerator


class StaticGenerator(BaseGenerator):
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        assert "Judge whether each claim" in prompt
        return self.response


def test_extract_claims_sentence_level():
    assert extract_claims("Tom asked LiHua. Sarah replied!") == [
        "Tom asked LiHua",
        "Sarah replied",
    ]


def test_claim_level_faithfulness_scores_supported_claims():
    report = claim_level_faithfulness(
        answer="Tom asked LiHua about the warranty. Sarah approved the refund.",
        evidence_texts=[
            "Tom asked LiHua about the laptop warranty during the meeting.",
        ],
    )

    assert len(report.claims) == 2
    assert report.claims[0].supported is True
    assert report.claims[1].supported is False
    assert report.groundedness == 0.5
    assert report.unsupported_claim_count == 1


def test_claim_level_faithfulness_ignores_citation_lines():
    assert extract_claims(
        "Tom asked LiHua.\nCitations: [record_id=minirag-q1-0-srcdoc-001]"
    ) == ["Tom asked LiHua"]


def test_claim_level_faithfulness_treats_abstention_as_claim_free():
    report = claim_level_faithfulness(
        answer="ABSTAINED: insufficient evidence",
        evidence_texts=["Tom asked LiHua about the warranty."],
    )

    assert report.claims == []
    assert report.groundedness == 1.0
    assert report.unsupported_claim_count == 0


def test_claim_level_faithfulness_supports_multi_evidence_claims():
    report = claim_level_faithfulness(
        answer=(
            "Yes, Adam Smith sent Li Hua a maintenance schedule message before "
            "the administrators announced a temporary construction schedule "
            "change due to weather conditions."
        ),
        evidence_texts=[
            (
                "Time: 20260121_10:00\nAdamSmith: Hi Li Hua! Just wanted to "
                "let you know that there will be some maintenance work in the "
                "building soon."
            ),
            (
                "Time: 20260701_10:00\nAdministrators announced a temporary "
                "change in the construction schedule due to weather conditions."
            ),
        ],
    )

    assert report.groundedness == 1.0
    assert report.claims[0].supporting_evidence_indices == [0, 1]


def test_claim_level_faithfulness_keeps_negated_claims_strict():
    report = claim_level_faithfulness(
        answer="The record does not state that Tom asked LiHua.",
        evidence_texts=["Tom asked LiHua about the warranty."],
    )

    assert report.groundedness == 0.0
    assert report.unsupported_claim_count == 1


def test_claim_level_faithfulness_matches_compound_names_and_inflections():
    report = claim_level_faithfulness(
        answer=(
            "Turalyon announced construction updates after Illidan Stormrage "
            "complained about construction noise and requested resident feedback."
        ),
        evidence_texts=[
            (
                "IllidanStormrage complained about the construction noise, and "
                "Turalyon asked residents to share feedback on construction updates."
            )
        ],
    )

    assert report.groundedness == 1.0


def test_llm_judge_faithfulness_parser():
    response = json.dumps(
        {
            "claims": [
                {
                    "claim": "Tom asked LiHua",
                    "supported": True,
                    "supporting_evidence_indices": [0],
                },
                {
                    "claim": "Sarah approved a refund",
                    "supported": False,
                    "supporting_evidence_indices": [],
                },
            ]
        }
    )

    report = judge_faithfulness_with_llm(
        answer="Tom asked LiHua. Sarah approved a refund.",
        evidence_texts=["Tom asked LiHua."],
        generator=StaticGenerator(response),
    )

    assert report.groundedness == 0.5
    assert report.claims[0].supporting_evidence_indices == [0]
