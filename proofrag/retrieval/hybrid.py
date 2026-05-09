"""
hybrid.py — Lightweight hybrid retrieval.

The current hybrid retriever combines BM25 candidate generation with
contract-aware reranking. Optional dense vector stores can be added later behind
extras without changing the public retriever interface.
"""

from __future__ import annotations

from pathlib import Path

from proofrag.contracts.schema import EvidenceContract
from proofrag.evidence.ledger import EvidenceLedger
from proofrag.retrieval.base import BaseRetriever
from proofrag.retrieval.bm25 import BM25Retriever
from proofrag.retrieval.rerank import ContractAwareReranker


class HybridRetriever(BaseRetriever):
    """BM25 candidates plus deterministic contract-aware reranking."""

    def __init__(
        self,
        context_path: Path | None = None,
        *,
        top_k: int = 5,
        candidate_k: int | None = None,
        reranker: ContractAwareReranker | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.top_k = top_k
        self.candidate_k = candidate_k or max(top_k * 3, top_k)
        self._bm25 = BM25Retriever(
            context_path=context_path,
            top_k=self.candidate_k,
        )
        self._reranker = reranker or ContractAwareReranker()
        self.total_docs = 0

    def retrieve(
        self,
        question: str,
        contract: EvidenceContract,
    ) -> EvidenceLedger:
        """Return reranked BM25 evidence candidates."""

        candidate_ledger = self._bm25.retrieve(
            question=question,
            contract=contract,
        )
        self.total_docs = self._bm25.total_docs
        reranked = self._reranker.rerank(
            records=candidate_ledger.records,
            contract=contract,
            top_k=self.top_k,
        )
        return EvidenceLedger(records=reranked)

