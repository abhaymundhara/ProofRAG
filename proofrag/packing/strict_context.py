"""
strict_context.py — StrictContextPacker.

Assembles a structured, evidence-gated prompt that instructs the downstream
language model to answer *only* from the verified evidence above, and to
explicitly admit incompleteness rather than hallucinate.
"""

from __future__ import annotations

from proofrag.contracts.schema import EvidenceContract
from proofrag.evidence.ledger import EvidenceLedger
from proofrag.evidence.sufficiency import SufficiencyReport


_SECTION_SEP = "\n" + "─" * 72 + "\n"


class StrictContextPacker:
    """Packs a verified evidence context into a structured generation prompt.

    The packed prompt is deliberately verbose so that even a small language
    model can follow its structure.  Every section is labelled and separated
    by a visible rule so the model can easily locate the relevant content.
    """

    def pack(
        self,
        question: str,
        contract: EvidenceContract,
        ledger: EvidenceLedger,
        report: SufficiencyReport,
    ) -> str:
        """Build and return the final prompt string.

        Args:
            question: The original natural-language question.
            contract: The evidence contract for this query.
            ledger:   All retrieved evidence records.
            report:   The sufficiency report produced by the scorer.

        Returns:
            A multi-section prompt string ready to be passed to a generator.
        """
        lines: list[str] = []

        # ------------------------------------------------------------------ #
        # Section 1: Question
        # ------------------------------------------------------------------ #
        lines.append("## QUESTION")
        lines.append(question)

        # ------------------------------------------------------------------ #
        # Section 2: Required Evidence
        # ------------------------------------------------------------------ #
        lines.append(_SECTION_SEP + "## REQUIRED EVIDENCE")
        if contract.required_slots:
            for slot in contract.required_slots:
                status = (
                    "✓ COVERED"
                    if slot.slot_id in report.covered_slots
                    else "✗ MISSING"
                )
                lines.append(
                    f"  [{status}] {slot.slot_id}: {slot.description} "
                    f"(min_sources={slot.min_sources})"
                )
        else:
            lines.append("  (no required slots declared)")

        # ------------------------------------------------------------------ #
        # Section 3: Evidence Found
        # ------------------------------------------------------------------ #
        lines.append(_SECTION_SEP + "## EVIDENCE FOUND")
        required_slot_ids = {s.slot_id for s in contract.required_slots}
        supporting_records = [
            rec for rec in ledger.records 
            if any(s in required_slot_ids for s in rec.supports_slots)
        ]
        if supporting_records:
            for rec in supporting_records:
                slots_label = ", ".join(rec.supports_slots)
                lines.append(
                    f"  [record_id={rec.record_id}] "
                    f"[source={rec.source_id}] "
                    f"[strength={rec.evidence_strength}] "
                    f"[confidence={rec.confidence:.2f}] "
                    f"[slots={slots_label}]"
                )
                lines.append(f"  \"{rec.text}\"")
                lines.append("")
        else:
            lines.append("  (no supporting evidence found)")

        # ------------------------------------------------------------------ #
        # Section 4: Missing Evidence
        # ------------------------------------------------------------------ #
        lines.append(_SECTION_SEP + "## MISSING EVIDENCE")
        if report.missing_required_slots:
            for slot_id in report.missing_required_slots:
                slot_obj = next(
                    (s for s in contract.slots if s.slot_id == slot_id), None
                )
                desc = slot_obj.description if slot_obj else "(unknown slot)"
                lines.append(f"  ✗ {slot_id}: {desc}")
        else:
            lines.append("  (no missing evidence — all required slots covered)")

        # ------------------------------------------------------------------ #
        # Section 5: Contradictions
        # ------------------------------------------------------------------ #
        lines.append(_SECTION_SEP + "## CONTRADICTIONS")
        contradicting = ledger.contradictions()
        if contradicting:
            for rec in contradicting:
                contra_label = ", ".join(rec.contradicts)
                lines.append(
                    f"  ⚠ [record_id={rec.record_id}] contradicts slots: "
                    f"{contra_label}"
                )
                lines.append(f"  \"{rec.text}\"")
                lines.append("")
        else:
            lines.append("  (no contradictions detected)")

        # ------------------------------------------------------------------ #
        # Section 6: Answer Policy
        # ------------------------------------------------------------------ #
        lines.append(_SECTION_SEP + "## ANSWER POLICY")
        if report.answer_allowed:
            lines.append(
                "  PERMITTED — sufficient evidence is available.\n"
                f"  Coverage score: {report.coverage_score:.0%} of required slots satisfied."
            )
        else:
            lines.append(
                "  BLOCKED — the answer is not permitted because:\n"
                f"  {report.reason}"
            )

        # ------------------------------------------------------------------ #
        # Section 7: Instruction
        # ------------------------------------------------------------------ #
        lines.append(_SECTION_SEP + "## INSTRUCTION")
        lines.append(
            "Answer only using the evidence above. "
            "If required evidence is missing, say the answer is incomplete "
            "rather than guessing."
        )
        lines.append(
            "\nDo NOT introduce facts that are not present in the evidence "
            "sections above.  Cite [record_id] when referencing a specific "
            "piece of evidence."
        )
        lines.append(
            "\nIMPORTANT: If the question uses a specific label for an event "
            "(e.g., 'dinner') but the evidence uses a different label "
            "(e.g., 'lunch', 'café'), you MUST mention the mismatch explicitly. "
            "Do NOT restate unsupported labels from the question as facts."
        )

        return "\n".join(lines)
