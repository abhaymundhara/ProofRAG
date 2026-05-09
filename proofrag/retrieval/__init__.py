"""
retrieval — Pluggable retrieval backends.
"""

from proofrag.retrieval.bm25 import BM25Retriever
from proofrag.retrieval.dummy import DummyRetriever
from proofrag.retrieval.hybrid import HybridRetriever
from proofrag.retrieval.iterative import ContractGapRetriever, IterativeRetrievalResult
from proofrag.retrieval.rerank import ContractAwareReranker
from proofrag.retrieval.vector import (
    ChromaRetriever,
    FAISSRetriever,
    LanceDBRetriever,
    OptionalVectorDependencyError,
)

__all__ = [
    "BM25Retriever",
    "ContractGapRetriever",
    "ContractAwareReranker",
    "ChromaRetriever",
    "DummyRetriever",
    "FAISSRetriever",
    "HybridRetriever",
    "IterativeRetrievalResult",
    "LanceDBRetriever",
    "OptionalVectorDependencyError",
]
