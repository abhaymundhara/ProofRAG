import json
import re
from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel

from proofrag.contracts.schema import EvidenceContract, EvidenceSlot
from proofrag.evidence.ledger import EvidenceRecord, EvidenceLedger
from proofrag.evidence.sufficiency import RuleBasedSufficiencyScorer
from proofrag.packing.strict_context import StrictContextPacker


class MiniRAGExportItem(BaseModel):
    id: str
    dataset: str
    question: str
    query_type: str
    gold_answer: str
    gold_supporting_sources: List[str]
    retrieved_context: List[Dict[str, Any]]
    baseline_answer: str
    baseline_method: str
    baseline_metrics: Dict[str, Any]


class MiniRAGOutputAdapter:
    """Adapts MiniRAG's exported results into ProofRAG's evidence framework."""

    def __init__(self):
        self.scorer = RuleBasedSufficiencyScorer()
        self.packer = StrictContextPacker()

    def load_export(self, file_path: str) -> List[MiniRAGExportItem]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Export file not found: {file_path}")

        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(MiniRAGExportItem(**json.loads(line)))
        return items

    def _infer_evidence(self, text: str, question: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Simple rules to infer supports_slots and evidence_strength."""
        supports_slots = []
        contradicts = []
        evidence_strength = "background"

        # Topic context: check for keywords from question
        # Simplified: if any word > 4 chars from question is in text
        q_keywords = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 4]
        if any(kw in text.lower() for kw in q_keywords):
            supports_slots.append("topic_context")
            evidence_strength = "indirect"

        # who_asked: check for actor-action phrases
        # Common names or pronouns followed by 'asked'
        if re.search(r"\b(Tom|Sarah|LiHua|someone|he|she|they)\b\s+asked\b", text, re.I):
            supports_slots.append("who_asked")
            evidence_strength = "direct"
        
        # Contradiction markers
        if metadata.get("contradiction") is True or "no one asked" in text.lower():
            if re.search(r"who\s+asked", question, re.I):
                contradicts.append("who_asked")

        return {
            "supports_slots": supports_slots,
            "contradicts": contradicts,
            "evidence_strength": evidence_strength
        }

    def process_item(self, item: MiniRAGExportItem) -> Dict[str, Any]:
        """Converts a MiniRAG item into a ProofRAG sufficiency report and prompt."""
        
        # 1. Build EvidenceContract
        # For 'who asked' questions, we require who_asked and topic_context
        slots = [
            EvidenceSlot(
                slot_id="who_asked",
                description="The person who initiated the request or question",
                evidence_type="actor",
                required=True,
                min_sources=1
            ),
            EvidenceSlot(
                slot_id="topic_context",
                description="Context about the topic being discussed",
                evidence_type="context",
                required=True,
                min_sources=1
            )
        ]
        contract = EvidenceContract(
            question=item.question,
            query_type=item.query_type,
            slots=slots,
            must_check_contradictions=True,
            strict_mode=True
        )

        # 2. Build EvidenceRecords
        records = []
        for i, ctx in enumerate(item.retrieved_context):
            inference = self._infer_evidence(ctx["text"], item.question, ctx.get("metadata", {}))
            records.append(EvidenceRecord(
                record_id=f"minirag-{item.id}-{i}",
                source_id=ctx["source_id"],
                text=ctx["text"],
                supports_slots=inference["supports_slots"],
                contradicts=inference["contradicts"],
                evidence_strength=inference["evidence_strength"],
                confidence=1.0  # Default for now
            ))

        # 3. Pipeline
        ledger = EvidenceLedger(records=records)
        report = self.scorer.score(contract, ledger)
        packed_prompt = self.packer.pack(
            question=item.question,
            contract=contract,
            ledger=ledger,
            report=report
        )

        return {
            "id": item.id,
            "dataset": item.dataset,
            "baseline_method": item.baseline_method,
            "question": item.question,
            "baseline_answer": item.baseline_answer,
            "gold_answer": item.gold_answer,
            "sufficiency_report": report.model_dump(),
            "evidence_records": [r.model_dump() for r in records],
            "packed_prompt_preview": packed_prompt[:500] + "..."
        }
