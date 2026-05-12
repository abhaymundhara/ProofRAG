"""LiHua source augmentation for normalized RAG exports."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


_STOPWORDS = {
    "about",
    "after",
    "answer",
    "before",
    "does",
    "from",
    "have",
    "lihua",
    "more",
    "question",
    "than",
    "that",
    "their",
    "them",
    "they",
    "this",
    "what",
    "when",
    "where",
    "with",
}

_QUERY_EXPANSIONS = {
    "achievement": ("progress", "improvement", "success", "stronger"),
    "achievements": ("progress", "improvement", "success", "stronger"),
    "fitness": (
        "body",
        "shape",
        "exercise",
        "figure",
        "gym",
        "training",
        "workout",
        "stronger",
    ),
    "healthy": ("health", "exercise", "fitness", "training", "workout"),
    "plan": ("schedule", "routine", "training"),
    "progress": ("improvement", "stronger", "training"),
}


@dataclass(frozen=True)
class LiHuaSourceDocument:
    source_id: str
    text: str
    tokens: list[str]


def augment_export_with_lihua_sources(
    *,
    input_path: str | Path,
    output_path: str | Path,
    data_dir: str | Path,
    top_k: int = 8,
) -> int:
    """Append top-ranked full LiHua source documents to each export row."""

    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    documents = load_lihua_source_documents(data_dir)
    ranker = _BM25Ranker(documents)
    rows_written = 0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(input_path).open("r", encoding="utf-8") as source, output.open(
        "w",
        encoding="utf-8",
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            existing_ids = {
                str(context.get("source_id", ""))
                for context in row.get("retrieved_context", [])
            }
            added = 0
            for score, document in ranker.rank(str(row.get("question", ""))):
                if _source_id_seen(document.source_id, existing_ids):
                    continue
                row.setdefault("retrieved_context", []).append(
                    {
                        "source_id": document.source_id,
                        "text": document.text,
                        "metadata": {
                            "retriever": "lihua_bm25",
                            "score": round(score, 4),
                        },
                    }
                )
                existing_ids.add(document.source_id)
                added += 1
                if added >= top_k:
                    break
            destination.write(json.dumps(row) + "\n")
            rows_written += 1
    return rows_written


def load_lihua_source_documents(data_dir: str | Path) -> list[LiHuaSourceDocument]:
    """Load text-like LiHua source files as BM25 documents."""

    root = Path(data_dir)
    documents: list[LiHuaSourceDocument] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        source_id = _source_id_from_text(path, text)
        documents.append(
            LiHuaSourceDocument(
                source_id=source_id,
                text=text,
                tokens=_tokens(text),
            )
        )
    return documents


class _BM25Ranker:
    def __init__(self, documents: list[LiHuaSourceDocument]) -> None:
        self.documents = documents
        self.average_length = (
            sum(len(document.tokens) for document in documents) / len(documents)
            if documents
            else 0.0
        )
        self.document_frequency: Counter[str] = Counter()
        for document in documents:
            self.document_frequency.update(set(document.tokens))

    def rank(self, question: str) -> list[tuple[float, LiHuaSourceDocument]]:
        query_tokens = _expanded_query_tokens(question)
        ranked: list[tuple[float, LiHuaSourceDocument]] = []
        for document in self.documents:
            score = self._score(query_tokens, document)
            if score > 0:
                ranked.append((score, document))
        return sorted(ranked, key=lambda item: item[0], reverse=True)

    def _score(self, query_tokens: list[str], document: LiHuaSourceDocument) -> float:
        if not self.documents or not document.tokens:
            return 0.0
        counts = Counter(document.tokens)
        score = 0.0
        for token in query_tokens:
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            doc_freq = self.document_frequency.get(token, 0)
            inverse_frequency = math.log(
                1 + (len(self.documents) - doc_freq + 0.5) / (doc_freq + 0.5)
            )
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * len(document.tokens) / self.average_length
            )
            score += inverse_frequency * (frequency * 2.5) / denominator
        return score


def _source_id_from_text(path: Path, text: str) -> str:
    match = re.search(r"Time:\s*([0-9]{8}[_:][0-9]{2}:?[0-9]{2})", text)
    if match:
        return match.group(1)
    return path.stem


def _source_id_seen(source_id: str, seen: set[str]) -> bool:
    variants = {
        source_id,
        source_id.replace(":", ""),
        source_id.replace(":", "_"),
        source_id.replace(":", "-"),
    }
    return bool(variants & seen)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    ]


def _expanded_query_tokens(text: str) -> list[str]:
    tokens = _tokens(text)
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(_QUERY_EXPANSIONS.get(token, ()))
    return expanded
