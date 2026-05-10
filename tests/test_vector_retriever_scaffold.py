from __future__ import annotations

import json
import types

from typer.testing import CliRunner

from proofrag.cli import app
from proofrag.contracts.schema import EvidenceContract, EvidenceSlot
from proofrag.retrieval.vector import (
    ChromaRetriever,
    FAISSRetriever,
    HashingEmbedder,
    LanceDBRetriever,
    OptionalVectorDependencyError,
)


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dimensions=16)

    first = embedder.embed("Tom asked LiHua")
    second = embedder.embed("Tom asked LiHua")

    assert first == second
    assert len(first) == 16
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9


def test_vector_retriever_reports_missing_optional_dependency_when_unavailable():
    try:
        FAISSRetriever()
    except OptionalVectorDependencyError as exc:
        message = str(exc)
    else:
        return

    assert "faiss retrieval requires optional dependency" in message
    assert "bm25" in message
    assert "hybrid" in message


def test_cli_vector_backend_reports_actionable_error():
    result = CliRunner().invoke(
        app,
        [
            "ask",
            "--question",
            "Who asked LiHua about the laptop warranty issue?",
            "--retriever",
            "faiss",
            "--json",
        ],
    )

    if result.exit_code == 0:
        return
    assert "faiss retrieval requires optional dependency" in result.output


def test_faiss_retriever_queries_optional_backend(tmp_path, monkeypatch):
    context_path = _write_vector_context(tmp_path)

    class FakeArray(list):
        @property
        def shape(self):
            return (len(self), len(self[0]) if self else 0)

    class FakeIndex:
        def __init__(self, dimensions):
            self.dimensions = dimensions

        def add(self, vectors):
            self.vectors = vectors

        def search(self, query, top_k):
            return [[1.0, 0.0][:top_k]], [[0, 1][:top_k]]

    monkeypatch.setitem(
        __import__("sys").modules,
        "faiss",
        types.SimpleNamespace(IndexFlatIP=FakeIndex),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "numpy",
        types.SimpleNamespace(array=lambda values, dtype=None: FakeArray(values)),
    )

    ledger = FAISSRetriever(context_path=context_path, top_k=1).retrieve(
        "Who asked LiHua?", _contract()
    )

    assert len(ledger.records) == 1
    assert ledger.records[0].source_id == "doc-1"
    assert ledger.records[0].supports_slots == ["who_asked"]
    assert ledger.records[0].confidence == 1.0


def test_chroma_retriever_queries_optional_backend(tmp_path, monkeypatch):
    context_path = _write_vector_context(tmp_path)

    class FakeCollection:
        def add(self, **kwargs):
            self.added = kwargs

        def query(self, **kwargs):
            return {"ids": [["0"]], "distances": [[0.0]]}

    class FakeClient:
        def create_collection(self, **kwargs):
            return FakeCollection()

    monkeypatch.setitem(
        __import__("sys").modules,
        "chromadb",
        types.SimpleNamespace(Client=FakeClient),
    )

    ledger = ChromaRetriever(context_path=context_path, top_k=1).retrieve(
        "Who asked LiHua?", _contract()
    )

    assert len(ledger.records) == 1
    assert ledger.records[0].source_id == "doc-1"
    assert ledger.records[0].confidence == 1.0


def test_lancedb_retriever_queries_optional_backend(tmp_path, monkeypatch):
    context_path = _write_vector_context(tmp_path)

    class FakeSearch:
        def limit(self, top_k):
            self.top_k = top_k
            return self

        def to_list(self):
            return [{"id": "0", "_distance": 0.0}]

    class FakeTable:
        def search(self, query_vector):
            self.query_vector = query_vector
            return FakeSearch()

    class FakeDB:
        def create_table(self, name, data):
            self.name = name
            self.data = data
            return FakeTable()

    monkeypatch.setitem(
        __import__("sys").modules,
        "lancedb",
        types.SimpleNamespace(connect=lambda path: FakeDB()),
    )

    ledger = LanceDBRetriever(context_path=context_path, top_k=1).retrieve(
        "Who asked LiHua?", _contract()
    )

    assert len(ledger.records) == 1
    assert ledger.records[0].source_id == "doc-1"
    assert ledger.records[0].confidence == 1.0


def _contract() -> EvidenceContract:
    return EvidenceContract(
        question="Who asked LiHua?",
        query_type="factoid",
        slots=[
            EvidenceSlot(
                slot_id="who_asked",
                description="who asked LiHua",
                evidence_type="factual",
            )
        ],
    )


def _write_vector_context(tmp_path):
    context_path = tmp_path / "context.json"
    context_path.write_text(
        json.dumps(
            [
                {
                    "source_id": "doc-1",
                    "text": "Tom asked LiHua about the laptop warranty.",
                    "keywords": ["Tom", "LiHua", "warranty"],
                    "supports_slots": ["who_asked"],
                },
                {
                    "source_id": "doc-2",
                    "text": "The lunch menu changed.",
                    "keywords": ["lunch"],
                    "supports_slots": [],
                },
            ]
        ),
        encoding="utf-8",
    )
    return context_path
