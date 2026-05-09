"""
faithfulness.py — Lightweight groundedness and faithfulness metrics.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from proofrag.evaluation.answer_metrics import normalize_answer
from proofrag.generation.base import BaseGenerator


class ClaimSupport(BaseModel):
    """Support decision for one generated-answer claim."""

    claim: str
    supported: bool
    supporting_evidence_indices: list[int] = Field(default_factory=list)


class FaithfulnessReport(BaseModel):
    """Claim-level groundedness report."""

    claims: list[ClaimSupport]
    groundedness: float
    unsupported_claim_count: int


def extract_claims(answer: str) -> list[str]:
    """Extract simple sentence-level claims from an answer."""

    cleaned = answer.strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    return [part.strip(" .!?") for part in parts if part.strip(" .!?")]


def claim_level_faithfulness(
    *,
    answer: str,
    evidence_texts: list[str],
) -> FaithfulnessReport:
    """Score whether answer claims are supported by evidence snippets."""

    claims = extract_claims(answer)
    supports: list[ClaimSupport] = []
    for claim in claims:
        supporting_indices = [
            index
            for index, evidence in enumerate(evidence_texts)
            if _claim_supported_by_text(claim, evidence)
        ]
        supports.append(
            ClaimSupport(
                claim=claim,
                supported=bool(supporting_indices),
                supporting_evidence_indices=supporting_indices,
            )
        )

    supported_count = sum(1 for support in supports if support.supported)
    total = len(supports)
    return FaithfulnessReport(
        claims=supports,
        groundedness=supported_count / total if total else 0.0,
        unsupported_claim_count=total - supported_count,
    )


def judge_faithfulness_with_llm(
    *,
    answer: str,
    evidence_texts: list[str],
    generator: BaseGenerator,
) -> FaithfulnessReport:
    """Use a generator as an optional judge and parse its JSON response."""

    prompt = (
        "Judge whether each claim in the answer is supported by the evidence. "
        "Return JSON with key `claims`, each item having `claim`, `supported`, "
        "and optional `supporting_evidence_indices`.\n\n"
        f"Evidence:\n{json.dumps(evidence_texts, indent=2)}\n\n"
        f"Answer:\n{answer}"
    )
    raw = generator.generate(prompt)
    data = json.loads(_extract_json_object(raw))
    claims = [ClaimSupport(**item) for item in data.get("claims", [])]
    supported_count = sum(1 for claim in claims if claim.supported)
    total = len(claims)
    return FaithfulnessReport(
        claims=claims,
        groundedness=supported_count / total if total else 0.0,
        unsupported_claim_count=total - supported_count,
    )


def _claim_supported_by_text(claim: str, evidence: str) -> bool:
    normalized_claim = normalize_answer(claim)
    normalized_evidence = normalize_answer(evidence)
    if not normalized_claim or not normalized_evidence:
        return False
    if normalized_claim in normalized_evidence:
        return True
    claim_terms = {
        term for term in normalized_claim.split() if len(term) > 3
    }
    if not claim_terms:
        return False
    overlap = sum(1 for term in claim_terms if term in normalized_evidence)
    return overlap / len(claim_terms) >= 0.8


def _extract_json_object(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in judge output")
    return match.group(0)

