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

