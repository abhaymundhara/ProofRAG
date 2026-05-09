"""
llm.py — LLM-assisted EvidenceContract inference.

The parser is intentionally strict and falls back to rule-based inference by
default so invalid model output cannot silently weaken the evidence gate.
"""

from __future__ import annotations

import json
import re

from proofrag.contracts.infer import infer_contract_from_question
from proofrag.contracts.schema import EvidenceContract
from proofrag.generation.base import BaseGenerator


def infer_contract_with_llm(
    *,
    question: str,
    generator: BaseGenerator,
    fallback_to_rule_based: bool = True,
) -> EvidenceContract:
    """Infer an EvidenceContract from a model-produced JSON object."""

    prompt = _contract_prompt(question)
    raw = generator.generate(prompt)
    try:
        return parse_contract_json(raw, question=question)
    except (ValueError, json.JSONDecodeError):
        if fallback_to_rule_based:
            return infer_contract_from_question(question)
        raise


def parse_contract_json(raw: str, *, question: str) -> EvidenceContract:
    """Parse and validate an EvidenceContract JSON object from model output."""

    payload = _extract_json_object(raw)
    data = json.loads(payload)
    data["question"] = question
    contract = EvidenceContract(**data)
    if not contract.required_slots:
        raise ValueError("LLM-inferred contract must include at least one required slot")
    return contract


def _extract_json_object(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM contract output")
    return match.group(0)


def _contract_prompt(question: str) -> str:
    return (
        "Infer a strict ProofRAG EvidenceContract for this question. "
        "Return only JSON with keys: query_type, slots, "
        "must_check_contradictions, strict_mode. Each slot must have slot_id, "
        "description, evidence_type, required, min_sources.\n\n"
        f"Question: {question}"
    )

