"""
dummy.py — DummyRetriever (keyword-based retrieval over examples/context.json).

This retriever is intentionally simple: it loads a static JSON file and
matches documents to evidence slots by checking whether any keyword from the
question (or the slot description) appears in the document text.

It is a development stub — replace with a real retriever in v0.2+.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from proofrag.contracts.schema import EvidenceContract
from proofrag.evidence.ledger import EvidenceRecord, EvidenceLedger


# Resolve the examples/context.json path relative to the package root.
# Works both in an editable install and in a built distribution.
_CONTEXT_FILE = Path(__file__).resolve().parents[2] / "examples" / "context.json"


class DummyRetriever:
    """Keyword-based retriever backed by ``examples/context.json``.

    Retrieval logic:
      - Tokenise the question and each slot description into lowercase words.
      - For each document, check whether any token appears in the document's
        ``keywords`` list.
      - Assign a fixed confidence of 0.75 to all matches.
      - A document is linked to every slot whose description shares a keyword
        with the document.

    Args:
        context_path: Override the default ``examples/context.json`` path.
                      Useful for testing.
    """

    def __init__(self, context_path: Path | None = None) -> None:
        self._context_path = context_path or _CONTEXT_FILE
        self.total_docs = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        contract: EvidenceContract,
    ) -> EvidenceLedger:
        """Load context documents and return a populated EvidenceLedger.

        Args:
            question: The natural-language question being answered.
            contract: The evidence contract describing required slots.

        Returns:
            An :class:`EvidenceLedger` containing matched evidence records.
        """
        docs = self._load_docs()
        self.total_docs = len(docs)
        question_tokens = self._tokenise(question)

        records: list[EvidenceRecord] = []

        for doc in docs:
            # 1. Prefer explicit supports_slots if present in context.json
            if "supports_slots" in doc:
                supported_slots = [
                    s for s in doc["supports_slots"] if s in contract.slot_ids
                ]
            else:
                # 2. Fallback to keyword matching
                doc_keywords: set[str] = set(kw.lower() for kw in doc.get("keywords", []))
                supported_slots = []
                for slot in contract.slots:
                    slot_tokens = self._tokenise(slot.description)
                    if doc_keywords & slot_tokens:
                        supported_slots.append(slot.slot_id)

            if not supported_slots:
                continue  # Document is irrelevant to this query

            record = EvidenceRecord(
                record_id=f"rec-{uuid.uuid4().hex[:8]}",
                source_id=doc["source_id"],
                text=doc["text"],
                supports_slots=supported_slots,
                confidence=0.75,
                contradicts=[],
            )
            records.append(record)

        return EvidenceLedger(records=records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_docs(self) -> list[dict]:
        with open(self._context_path, encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _tokenise(text: str) -> set[str]:
        """Lower-case, split on whitespace/punctuation, return unique tokens."""
        return set(re.findall(r"[a-z0-9]+", text.lower()))
