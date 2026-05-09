"""
vector.py — Optional vector-store retriever adapters.

The default ProofRAG install stays lightweight. FAISS, Chroma, and LanceDB are
imported only when their retrievers are instantiated. All three adapters use a
deterministic hashing embedder by default so they do not require a separate
embedding model.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
import uuid
from pathlib import Path
from typing import Any, Protocol, cast

from proofrag.contracts.schema import EvidenceContract
from proofrag.evidence.ledger import EvidenceLedger, EvidenceRecord
from proofrag.retrieval.base import (
    BaseRetriever,
    RetrievalDocument,
    load_json_documents,
)
from proofrag.retrieval.bm25 import _CONTEXT_FILE


class OptionalVectorDependencyError(ImportError):
    """Raised when an optional vector backend is requested but unavailable."""


class Embedder(Protocol):
    """Embedding callable used by vector retrievers."""

    def embed(self, text: str) -> list[float]:
        """Return a vector embedding for text."""


class HashingEmbedder:
    """Deterministic dependency-free hashing embedder."""

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be >= 8")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _tokenise(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class _VectorRetrieverBase(BaseRetriever):
    backend_name = "vector"
    extra_name = "vector"

    def __init__(
        self,
        context_path: Path | None = None,
        *,
        top_k: int = 5,
        embedder: Embedder | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._context_path = context_path or _CONTEXT_FILE
        self.top_k = top_k
        self.embedder = embedder or HashingEmbedder()
        self.total_docs = 0
        self._check_dependencies()

    def retrieve(self, question: str, contract: EvidenceContract) -> EvidenceLedger:
        docs = load_json_documents(self._context_path)
        self.total_docs = len(docs)
        if not docs:
            return EvidenceLedger(records=[])
        query_text = " ".join([question, *(slot.description for slot in contract.slots)])
        ranked_docs = self._query_documents(
            docs=docs,
            query_vector=self.embedder.embed(query_text),
        )
        records = [
            _document_to_record(
                doc=doc,
                contract=contract,
                confidence=score,
                backend_name=self.backend_name,
            )
            for doc, score in ranked_docs[: self.top_k]
        ]
        return EvidenceLedger(
            records=[record for record in records if record is not None]
        )

    def _query_documents(
        self,
        *,
        docs: list[RetrievalDocument],
        query_vector: list[float],
    ) -> list[tuple[RetrievalDocument, float]]:
        raise NotImplementedError

    def _check_dependencies(self) -> None:
        """Validate optional backend dependencies at construction time."""

    def _dependency_error(self, package: str) -> OptionalVectorDependencyError:
        return OptionalVectorDependencyError(
            f"{self.backend_name} retrieval requires optional dependency "
            f"`{package}`. Install ProofRAG with the `{self.extra_name}` extra, "
            "or use `bm25`/`hybrid`."
        )


class FAISSRetriever(_VectorRetrieverBase):
    """FAISS-backed vector retriever."""

    backend_name = "faiss"
    extra_name = "faiss"

    def _check_dependencies(self) -> None:
        try:
            import faiss  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as exc:
            raise self._dependency_error("faiss-cpu") from exc

    def _query_documents(
        self,
        *,
        docs: list[RetrievalDocument],
        query_vector: list[float],
    ) -> list[tuple[RetrievalDocument, float]]:
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise self._dependency_error("faiss-cpu") from exc

        vectors = np.array(
            [self.embedder.embed(_document_text(doc)) for doc in docs],
            dtype="float32",
        )
        query = np.array([query_vector], dtype="float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        scores, indices = index.search(query, min(self.top_k, len(docs)))
        return [
            (docs[int(index_)], _normalise_similarity(float(score)))
            for score, index_ in zip(scores[0], indices[0])
            if int(index_) >= 0
        ]


class ChromaRetriever(_VectorRetrieverBase):
    """Chroma-backed vector retriever."""

    backend_name = "chroma"
    extra_name = "chroma"

    def _check_dependencies(self) -> None:
        try:
            import chromadb  # noqa: F401
        except ImportError as exc:
            raise self._dependency_error("chromadb") from exc

    def _query_documents(
        self,
        *,
        docs: list[RetrievalDocument],
        query_vector: list[float],
    ) -> list[tuple[RetrievalDocument, float]]:
        try:
            import chromadb
        except ImportError as exc:
            raise self._dependency_error("chromadb") from exc

        client = chromadb.Client()
        collection = cast(Any, client.create_collection(
            name=f"proofrag_{uuid.uuid4().hex}",
            metadata={"hnsw:space": "cosine"},
        ))
        ids = [str(index) for index in range(len(docs))]
        collection.add(
            ids=ids,
            documents=[doc.text for doc in docs],
            embeddings=[self.embedder.embed(_document_text(doc)) for doc in docs],
        )
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(self.top_k, len(docs)),
        )
        result_ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            (docs[int(doc_id)], _normalise_distance(float(distance)))
            for doc_id, distance in zip(result_ids, distances)
        ]


class LanceDBRetriever(_VectorRetrieverBase):
    """LanceDB-backed vector retriever."""

    backend_name = "lancedb"
    extra_name = "lancedb"

    def _check_dependencies(self) -> None:
        try:
            import lancedb  # noqa: F401
        except ImportError as exc:
            raise self._dependency_error("lancedb") from exc

    def _query_documents(
        self,
        *,
        docs: list[RetrievalDocument],
        query_vector: list[float],
    ) -> list[tuple[RetrievalDocument, float]]:
        try:
            import lancedb
        except ImportError as exc:
            raise self._dependency_error("lancedb") from exc

        with tempfile.TemporaryDirectory(prefix="proofrag_lancedb_") as tmpdir:
            db = lancedb.connect(tmpdir)
            table = db.create_table(
                "proofrag",
                data=[
                    {
                        "id": str(index),
                        "text": doc.text,
                        "vector": self.embedder.embed(_document_text(doc)),
                    }
                    for index, doc in enumerate(docs)
                ],
            )
            rows = table.search(query_vector).limit(min(self.top_k, len(docs))).to_list()
        ranked: list[tuple[RetrievalDocument, float]] = []
        for row in rows:
            index = int(row["id"])
            distance = float(row.get("_distance", 0.0))
            ranked.append((docs[index], _normalise_distance(distance)))
        return ranked


def _document_to_record(
    *,
    doc: RetrievalDocument,
    contract: EvidenceContract,
    confidence: float,
    backend_name: str,
) -> EvidenceRecord | None:
    supported_slots = _supported_slots(doc, contract)
    contradicts = [slot for slot in doc.contradicts if slot in contract.slot_ids]
    if not supported_slots and not contradicts:
        return None
    return EvidenceRecord(
        record_id=f"{backend_name}-{uuid.uuid4().hex[:8]}",
        source_id=doc.source_id,
        text=doc.text,
        supports_slots=supported_slots,
        contradicts=contradicts,
        evidence_strength=doc.evidence_strength,
        confidence=max(0.0, min(1.0, confidence)),
    )


def _supported_slots(
    doc: RetrievalDocument,
    contract: EvidenceContract,
) -> list[str]:
    if doc.supports_slots is not None:
        return [slot for slot in doc.supports_slots if slot in contract.slot_ids]
    doc_terms = set(_tokenise(_document_text(doc)))
    supported: list[str] = []
    for slot in contract.slots:
        if doc_terms & set(_tokenise(slot.description)):
            supported.append(slot.slot_id)
    return supported


def _document_text(doc: RetrievalDocument) -> str:
    return " ".join([doc.text, *doc.keywords])


def _tokenise(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def _normalise_similarity(score: float) -> float:
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _normalise_distance(distance: float) -> float:
    return 1.0 / (1.0 + max(0.0, distance))
