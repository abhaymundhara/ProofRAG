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


_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "before",
    "but",
    "can",
    "cannot",
    "could",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "here",
    "hua",
    "into",
    "its",
    "let",
    "li",
    "not",
    "that",
    "the",
    "there",
    "this",
    "through",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "yes",
}


def extract_claims(answer: str) -> list[str]:
    """Extract simple sentence-level claims from an answer."""

    cleaned = answer.strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    claims = [part.strip(" .!?") for part in parts if part.strip(" .!?")]
    return [
        claim
        for claim in claims
        if not _is_citation_only_claim(claim) and not _is_abstention_claim(claim)
    ]


def claim_level_faithfulness(
    *,
    answer: str,
    evidence_texts: list[str],
) -> FaithfulnessReport:
    """Score whether answer claims are supported by evidence snippets."""

    claims = extract_claims(answer)
    if not claims and _is_abstention_claim(answer):
        return FaithfulnessReport(
            claims=[],
            groundedness=1.0,
            unsupported_claim_count=0,
        )

    supports: list[ClaimSupport] = []
    for claim in claims:
        supporting_indices = [
            index
            for index, evidence in enumerate(evidence_texts)
            if _claim_supported_by_text(claim, evidence)
        ]
        if not supporting_indices and _claim_supported_by_evidence_set(
            claim, evidence_texts
        ):
            supporting_indices = _overlapping_evidence_indices(claim, evidence_texts)
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
    if _is_negated(normalized_claim) and not _is_negated(normalized_evidence):
        return False
    claim_terms = _meaningful_terms(normalized_claim)
    if not claim_terms:
        return False
    evidence_terms = set(normalized_evidence.split())
    overlap = sum(1 for term in claim_terms if _term_supported(term, evidence_terms))
    return overlap / len(claim_terms) >= 0.8


def _claim_supported_by_evidence_set(claim: str, evidence_texts: list[str]) -> bool:
    normalized_claim = normalize_answer(claim)
    normalized_evidence = normalize_answer(" ".join(evidence_texts))
    if not normalized_claim or not normalized_evidence:
        return False
    if _is_negated(normalized_claim) and not _is_negated(normalized_evidence):
        return False
    claim_terms = _meaningful_terms(normalized_claim)
    if not claim_terms:
        return False
    evidence_terms = set(normalized_evidence.split())
    overlap = sum(1 for term in claim_terms if _term_supported(term, evidence_terms))
    return overlap / len(claim_terms) >= 0.65


def _overlapping_evidence_indices(claim: str, evidence_texts: list[str]) -> list[int]:
    claim_terms = _meaningful_terms(normalize_answer(claim))
    if not claim_terms:
        return []
    indices: list[int] = []
    for index, evidence in enumerate(evidence_texts):
        evidence_terms = set(normalize_answer(evidence).split())
        if any(_term_supported(term, evidence_terms) for term in claim_terms):
            indices.append(index)
    return indices


def _meaningful_terms(normalized_text: str) -> set[str]:
    return {
        _stem_term(term)
        for term in normalized_text.split()
        if len(term) > 3 and term not in _STOPWORDS and not term.startswith("record_id")
    }


def _term_supported(term: str, evidence_terms: set[str]) -> bool:
    if term in evidence_terms:
        return True
    variants = _term_variants(term)
    if variants & evidence_terms:
        return True
    return any(len(term) > 5 and term in evidence_term for evidence_term in evidence_terms)


def _term_variants(term: str) -> set[str]:
    return {
        term,
        f"{term}s",
        f"{term}ed",
        f"{term}ing",
    }


def _stem_term(term: str) -> str:
    for suffix in ("ing", "ed", "s"):
        if len(term) > len(suffix) + 4 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def _is_citation_only_claim(claim: str) -> bool:
    normalized = normalize_answer(claim)
    if not normalized:
        return True
    if normalized.startswith(("citation", "citations", "sources", "source")):
        return True
    terms = normalized.split()
    return bool(terms) and all(
        term.startswith("record_id") or re.fullmatch(r"\d+", term) for term in terms
    )


def _is_abstention_claim(claim: str) -> bool:
    normalized = normalize_answer(claim)
    return any(
        pattern in normalized
        for pattern in (
            "abstained insufficient evidence",
            "insufficient evidence",
            "not enough information",
            "cannot be determined",
            "cannot determine",
            "cannot be confirmed",
            "no evidence",
            "no information",
            "no record",
        )
    )


def _is_negated(normalized_text: str) -> bool:
    return bool(
        re.search(
            r"\b(no|not|cannot|cant|couldnt|didnt|doesnt|dont|isnt|wasnt|without)\b",
            normalized_text,
        )
    )


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
