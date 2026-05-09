"""
base.py — Common retrieval types for ProofRAG retrievers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

from proofrag.contracts.schema import EvidenceContract
from proofrag.evidence.ledger import EvidenceLedger


class RetrievalDocument(BaseModel):
    """A source document or chunk available to a retriever."""

    source_id: str
    text: str
    keywords: list[str] = Field(default_factory=list)
    supports_slots: list[str] | None = None
    contradicts: list[str] = Field(default_factory=list)
    evidence_strength: str = "direct"


class BaseRetriever(ABC):
    """Abstract base class for retriever backends."""

    total_docs: int

    @abstractmethod
    def retrieve(
        self,
        question: str,
        contract: EvidenceContract,
    ) -> EvidenceLedger:
        """Return evidence records for a question and contract."""


def load_json_documents(context_path: Path) -> list[RetrievalDocument]:
    """Load retrieval documents from the repository JSON corpus format."""

    import json

    with open(context_path, encoding="utf-8") as fh:
        raw_docs = json.load(fh)
    return [RetrievalDocument(**doc) for doc in raw_docs]
