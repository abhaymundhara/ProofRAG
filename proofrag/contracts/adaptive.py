"""
adaptive.py — Deterministic adaptive EvidenceContract strengthening.

Adaptive contracts keep the default rule-based inference lightweight while
raising evidence requirements for questions where unsafe hallucination would be
more costly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from proofrag.contracts.infer import infer_contract_from_question
from proofrag.contracts.schema import EvidenceContract, EvidenceSlot


class ContractRiskAssessment(BaseModel):
    """Risk features detected from a user question."""

    risk_level: str
    reasons: list[str] = Field(default_factory=list)


class AdaptiveContractBuilder:
    """Build an EvidenceContract and strengthen it based on question risk."""

    HIGH_RISK_TERMS = {
        "medical",
        "medicine",
        "doctor",
        "diagnosis",
        "legal",
        "lawyer",
        "lawsuit",
        "financial",
        "investment",
        "safety",
        "emergency",
    }
    MULTI_HOP_TERMS = {"why", "how", "compare", "difference", "cause", "because"}

    def build(self, question: str) -> EvidenceContract:
        """Return an adaptive contract for a question."""

        base = infer_contract_from_question(question)
        assessment = self.assess(question)
        slots = [
            slot.model_copy(
                update={
                    "min_sources": max(
                        slot.min_sources,
                        2 if assessment.risk_level == "high" and slot.required else slot.min_sources,
                    )
                }
            )
            for slot in base.slots
        ]

        if "multi_hop" in assessment.reasons and "reasoning_context" not in {
            slot.slot_id for slot in slots
        }:
            slots.append(
                EvidenceSlot(
                    slot_id="reasoning_context",
                    description="Evidence connecting the answer to the reasoning or causal chain",
                    evidence_type="reasoning",
                    required=True,
                    min_sources=1 if assessment.risk_level != "high" else 2,
                )
            )

        return EvidenceContract(
            question=base.question,
            query_type=f"adaptive_{base.query_type}",
            slots=slots,
            must_check_contradictions=True,
            strict_mode=True,
        )

    def assess(self, question: str) -> ContractRiskAssessment:
        """Return deterministic risk labels for a question."""

        q_lower = question.lower()
        reasons: list[str] = []
        if any(term in q_lower for term in self.HIGH_RISK_TERMS):
            reasons.append("high_stakes_domain")
        if any(term in q_lower for term in self.MULTI_HOP_TERMS):
            reasons.append("multi_hop")

        risk_level = "high" if "high_stakes_domain" in reasons else "standard"
        return ContractRiskAssessment(risk_level=risk_level, reasons=reasons)


def infer_adaptive_contract(question: str) -> EvidenceContract:
    """Convenience wrapper for adaptive contract inference."""

    return AdaptiveContractBuilder().build(question)

