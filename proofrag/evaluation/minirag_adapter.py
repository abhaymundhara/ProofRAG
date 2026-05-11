import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel

from proofrag.contracts.infer import infer_contract_from_question
from proofrag.evidence.extraction import infer_evidence_from_text
from proofrag.evidence.ledger import EvidenceRecord, EvidenceLedger
from proofrag.evidence.sufficiency import RuleBasedSufficiencyScorer
from proofrag.packing.strict_context import StrictContextPacker

class NormalizedRAGExportItem(BaseModel):
    """Normalized external RAG export consumed by ProofRAG gating.

    MiniRAG and LightRAG-style exports share this lightweight JSONL schema. The
    `baseline_method` field records which upstream system produced the answer.
    """

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


MiniRAGExportItem = NormalizedRAGExportItem


class MiniRAGOutputAdapter:
    """Adapts normalized MiniRAG exports into ProofRAG's evidence framework."""

    def __init__(self):
        self.scorer = RuleBasedSufficiencyScorer()
        self.packer = StrictContextPacker()

    def load_export(self, file_path: str) -> List[NormalizedRAGExportItem]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Export file not found: {file_path}")

        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(NormalizedRAGExportItem(**json.loads(line)))
        return items

    def _infer_evidence(self, text: str, question: str, metadata: Dict[str, Any], source_id: str = "") -> Dict[str, Any]:
        """Compatibility wrapper around shared evidence extraction rules."""
        return infer_evidence_from_text(
            text=text,
            question=question,
            metadata=metadata,
            source_id=source_id,
        ).model_dump()

    def _extract_minirag_source_rows(self, text: str) -> List[Dict[str, str]]:
        return self.extract_source_rows(text)

    def extract_source_rows(self, text: str) -> List[Dict[str, str]]:
        """Extract source rows from MiniRAG/LightRAG-style context text."""

        text = text or ""
        sources_marker = "-----Sources-----"
        marker_index = text.find(sources_marker)
        if marker_index == -1:
            return self._extract_time_delimited_rows(text)
        sources_text = text[marker_index + len(sources_marker):]
        fence_match = re.search(r"```csv\s*(.*?)\s*```", sources_text, re.DOTALL | re.IGNORECASE)
        if not fence_match:
            return self._extract_time_delimited_rows(text)
        csv_text = fence_match.group(1).strip()
        if not csv_text:
            return self._extract_time_delimited_rows(text)
        try:
            reader = csv.DictReader(io.StringIO(csv_text))
            rows: list[Dict[str, str]] = []
            for row in reader:
                row_id = str(row.get("id", "")).strip()
                content = row.get("content", "")
                if content:
                    rows.append({
                        "id": row_id or str(len(rows)),
                        "content": content.strip()
                    })
            return rows or [{"id": "raw", "content": text}]
        except Exception:
            return self._extract_time_delimited_rows(text)

    def _extract_time_delimited_rows(self, text: str) -> List[Dict[str, str]]:
        chunks = [chunk.strip() for chunk in text.split("--New Chunk--") if chunk.strip()]
        rows: list[Dict[str, str]] = []
        for index, chunk in enumerate(chunks):
            match = re.search(r"^\s*Time:\s*([0-9]{8}[_:][0-9]{2}:?[0-9]{2})", chunk, re.MULTILINE)
            source_id = match.group(1) if match else f"raw-{index}"
            rows.append({"id": source_id, "content": chunk})
        return rows or [{"id": "raw", "content": text}]

    def process_item(self, item: NormalizedRAGExportItem) -> Dict[str, Any]:
        """Convert a normalized external RAG item into ProofRAG artifacts."""
        
        # 1. Build EvidenceContract using inference
        contract = infer_contract_from_question(item.question)
        contract.query_type = item.query_type # Preserve external type if provided
        contract.strict_mode = True

        # 2. Build EvidenceRecords
        records = []
        for i, ctx in enumerate(item.retrieved_context):
            source_rows = self.extract_source_rows(ctx.get("text", ""))
            
            for j, row in enumerate(source_rows):
                source_row_id = row.get("id") or str(j)
                source_text = row.get("content", "")
                inference = self._infer_evidence(
                    source_text, 
                    item.question, 
                    ctx.get("metadata", {}),
                    source_id=source_row_id
                )
                records.append(EvidenceRecord(
                    record_id=f"minirag-{item.id}-{i}-src{source_row_id}",
                    source_id=f'{ctx["source_id"]}#src{source_row_id}',
                    text=source_text,
                    supports_slots=inference["supports_slots"],
                    contradicts=inference["contradicts"],
                    evidence_strength=inference["evidence_strength"],
                    confidence=1.0
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
            "packed_prompt": packed_prompt,
            "packed_prompt_preview": packed_prompt[:500] + "..."
        }


class LightRAGOutputAdapter(MiniRAGOutputAdapter):
    """Adapts normalized LightRAG-style exports into ProofRAG artifacts.

    LightRAG exports should use the same JSONL shape as MiniRAG exports and set
    `baseline_method` to `lightrag` or another explicit upstream identifier.
    """
