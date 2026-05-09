"""
bm25.py — Dependency-light BM25 retriever.

This backend provides a real lexical ranking step without adding heavy
dependencies. It is intended as the first production-shaped retriever before
optional vector stores and hybrid retrieval are added.
"""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from pathlib import Path

from proofrag.contracts.schema import EvidenceContract
from proofrag.evidence.ledger import EvidenceLedger, EvidenceRecord
from proofrag.retrieval.base import BaseRetriever, RetrievalDocument, load_json_documents


_CONTEXT_FILE = Path(__file__).resolve().parents[2] / "examples" / "context.json"


class BM25Retriever(BaseRetriever):
    """Rank JSON corpus documents with BM25 and return contract-linked evidence."""

    def __init__(
        self,
        context_path: Path | None = None,
        *,
        top_k: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._context_path = context_path or _CONTEXT_FILE
        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self.total_docs = 0

    def retrieve(
        self,
        question: str,
        contract: EvidenceContract,
    ) -> EvidenceLedger:
        """Return a ledger populated from top-ranked BM25 documents."""

        docs = load_json_documents(self._context_path)
        self.total_docs = len(docs)
        if not docs:
            return EvidenceLedger(records=[])

        query_tokens = self._query_tokens(question, contract)
        ranked = self._rank(query_tokens, docs)
        records: list[EvidenceRecord] = []

        for score, doc in ranked[: self.top_k]:
            if score <= 0:
                continue
            supported_slots = self._supported_slots(doc, contract)
            contradicts = [s for s in doc.contradicts if s in contract.slot_ids]
            if not supported_slots and not contradicts:
                continue
            records.append(
                EvidenceRecord(
                    record_id=f"bm25-{uuid.uuid4().hex[:8]}",
                    source_id=doc.source_id,
                    text=doc.text,
                    supports_slots=supported_slots,
                    contradicts=contradicts,
                    evidence_strength=doc.evidence_strength,
                    confidence=round(score / (score + 1.0), 4),
                )
            )

        return EvidenceLedger(records=records)

    def _rank(
        self,
        query_tokens: list[str],
        docs: list[RetrievalDocument],
    ) -> list[tuple[float, RetrievalDocument]]:
        doc_tokens = [self._tokenise(self._doc_text(doc)) for doc in docs]
        avgdl = sum(len(tokens) for tokens in doc_tokens) / len(doc_tokens)
        document_frequency = self._document_frequency(doc_tokens)

        ranked: list[tuple[float, RetrievalDocument]] = []
        for doc, tokens in zip(docs, doc_tokens):
            term_counts = Counter(tokens)
            score = 0.0
            for term in query_tokens:
                if term not in term_counts:
                    continue
                score += self._term_score(
                    term=term,
                    term_frequency=term_counts[term],
                    doc_len=len(tokens),
                    avgdl=avgdl,
                    doc_count=len(docs),
                    document_frequency=document_frequency.get(term, 0),
                )
            ranked.append((score, doc))

        return sorted(ranked, key=lambda item: item[0], reverse=True)

    def _term_score(
        self,
        *,
        term: str,
        term_frequency: int,
        doc_len: int,
        avgdl: float,
        doc_count: int,
        document_frequency: int,
    ) -> float:
        _ = term
        idf = math.log(1 + (doc_count - document_frequency + 0.5) / (document_frequency + 0.5))
        numerator = term_frequency * (self.k1 + 1)
        denominator = term_frequency + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
        return idf * numerator / denominator

    @staticmethod
    def _document_frequency(doc_tokens: list[list[str]]) -> dict[str, int]:
        frequency: dict[str, int] = {}
        for tokens in doc_tokens:
            for token in set(tokens):
                frequency[token] = frequency.get(token, 0) + 1
        return frequency

    @staticmethod
    def _query_tokens(question: str, contract: EvidenceContract) -> list[str]:
        query_text = " ".join(
            [question, *(slot.description for slot in contract.slots)]
        )
        return BM25Retriever._tokenise(query_text)

    @staticmethod
    def _doc_text(doc: RetrievalDocument) -> str:
        return " ".join([doc.text, *doc.keywords])

    @staticmethod
    def _supported_slots(
        doc: RetrievalDocument,
        contract: EvidenceContract,
    ) -> list[str]:
        if doc.supports_slots is not None:
            return [slot for slot in doc.supports_slots if slot in contract.slot_ids]

        doc_terms = set(BM25Retriever._tokenise(BM25Retriever._doc_text(doc)))
        supported: list[str] = []
        for slot in contract.slots:
            slot_terms = set(BM25Retriever._tokenise(slot.description))
            if doc_terms & slot_terms:
                supported.append(slot.slot_id)
        return supported

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())
