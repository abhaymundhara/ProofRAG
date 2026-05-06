"""
sufficiency.py — Rule-based sufficiency scoring.

RuleBasedSufficiencyScorer evaluates an EvidenceLedger against an
EvidenceContract and returns a SufficiencyReport that gates generation.

Rules (applied in order):
  1. All required slots must be covered by at least one record.
  2. Each required slot must be supported by at least ``slot.min_sources``
     distinct records.
  3. If ``contract.strict_mode`` is True and any contradiction exists in the
     ledger, ``answer_allowed`` is forced to False.
  4. ``coverage_score`` = covered required slots / total required slots.
"""

from __future__ import annotations

from pydantic import BaseModel

from proofrag.contracts.schema import EvidenceContract
from proofrag.evidence.ledger import EvidenceLedger


class SufficiencyReport(BaseModel):
    """The result of a sufficiency check.

    Attributes:
        answer_allowed:         True only when all required evidence is present
                                and no blocking contradictions exist.
        coverage_score:         Fraction of required slots that are satisfied
                                (0.0 – 1.0).
        covered_slots:          Slot IDs that are satisfied.
        missing_required_slots: Required slots with insufficient evidence.
        contradiction_count:    Number of records that contain contradictions.
        contract_slot_ids:      All slot IDs defined in the contract.
        reason:                 Human-readable explanation of the decision.
    """

    answer_allowed: bool
    coverage_score: float
    covered_slots: list[str]
    missing_required_slots: list[str]
    contradiction_count: int
    contract_slot_ids: list[str]
    reason: str


class RuleBasedSufficiencyScorer:
    """Evaluates evidence sufficiency using deterministic, rule-based logic.

    This scorer is intentionally simple and fully auditable — no model calls,
    no embeddings, no probabilities beyond confidence thresholds.
    """

    def score(
        self,
        contract: EvidenceContract,
        ledger: EvidenceLedger,
    ) -> SufficiencyReport:
        """Run the sufficiency check and return a SufficiencyReport.

        Args:
            contract: The evidence contract for the current query.
            ledger:   The collected evidence records.

        Returns:
            A :class:`SufficiencyReport` describing what is satisfied,
            what is missing, and whether generation is permitted.
        """
        required_slots = contract.required_slots

        # ------------------------------------------------------------------ #
        # 1 & 2 — coverage and min_sources
        # ------------------------------------------------------------------ #
        covered: list[str] = []
        missing: list[str] = []

        for slot in required_slots:
            supporting = ledger.records_for_slot(slot.slot_id)
            # Only direct evidence can satisfy required slots by default
            direct_supporting = [
                r for r in supporting if r.evidence_strength == "direct"
            ]
            if len(direct_supporting) >= slot.min_sources:
                covered.append(slot.slot_id)
            else:
                missing.append(slot.slot_id)

        total_required = len(required_slots)
        coverage_score: float = (
            len(covered) / total_required if total_required > 0 else 1.0
        )

        # ------------------------------------------------------------------ #
        # 3 — contradiction check
        # ------------------------------------------------------------------ #
        contradicting_records = ledger.contradictions()
        contradiction_count = len(contradicting_records)

        # ------------------------------------------------------------------ #
        # Decision
        # ------------------------------------------------------------------ #
        answer_allowed = len(missing) == 0

        # Track whether contradictions are what tips the decision to False,
        # so the reason string is causally accurate (not just descriptive).
        _blocked_by_contradiction = (
            answer_allowed and contract.strict_mode and contradiction_count > 0
        )
        if _blocked_by_contradiction:
            answer_allowed = False

        # Build human-readable reason
        reason_parts: list[str] = []

        if missing:
            reason_parts.append(
                f"Missing required slots: {', '.join(missing)}."
            )
        if _blocked_by_contradiction:
            reason_parts.append(
                f"Strict mode blocked: {contradiction_count} contradicting "
                f"record(s) found."
            )
        if answer_allowed:
            reason_parts.append(
                "All required evidence present; no blocking contradictions."
            )

        reason = " ".join(reason_parts) if reason_parts else "Insufficient evidence."

        return SufficiencyReport(
            answer_allowed=answer_allowed,
            coverage_score=round(coverage_score, 4),
            covered_slots=covered,
            missing_required_slots=missing,
            contradiction_count=contradiction_count,
            contract_slot_ids=contract.slot_ids,
            reason=reason,
        )
