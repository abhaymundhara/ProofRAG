from __future__ import annotations

import json

import pytest

from proofrag.contracts.llm import infer_contract_with_llm, parse_contract_json
from proofrag.generation.base import BaseGenerator


class StaticGenerator(BaseGenerator):
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_parse_contract_json_from_fenced_model_output():
    raw = """```json
{"query_type":"factoid","slots":[{"slot_id":"answer","description":"Answer evidence","evidence_type":"factual","required":true,"min_sources":1}],"must_check_contradictions":true,"strict_mode":true}
```"""

    contract = parse_contract_json(raw, question="What happened?")

    assert contract.question == "What happened?"
    assert contract.query_type == "factoid"
    assert contract.required_slots[0].slot_id == "answer"


def test_llm_contract_inference_uses_generator_json():
    response = json.dumps(
        {
            "query_type": "actor_query",
            "slots": [
                {
                    "slot_id": "who_asked",
                    "description": "The person who asked",
                    "evidence_type": "actor",
                    "required": True,
                    "min_sources": 1,
                }
            ],
            "must_check_contradictions": True,
            "strict_mode": True,
        }
    )
    generator = StaticGenerator(response)

    contract = infer_contract_with_llm(
        question="Who asked LiHua?",
        generator=generator,
    )

    assert contract.slot_ids == ["who_asked"]
    assert "Infer a strict ProofRAG EvidenceContract" in generator.prompts[0]


def test_llm_contract_inference_falls_back_to_rule_based():
    contract = infer_contract_with_llm(
        question="Who asked LiHua?",
        generator=StaticGenerator("not json"),
    )

    assert "who_asked" in contract.slot_ids


def test_llm_contract_inference_can_raise_on_invalid_output():
    with pytest.raises(ValueError):
        infer_contract_with_llm(
            question="Who asked LiHua?",
            generator=StaticGenerator("not json"),
            fallback_to_rule_based=False,
        )

