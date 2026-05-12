"""Strict verifier prompt helpers for evidence-contracted generation."""

from __future__ import annotations

import re

from proofrag.evidence.ledger import EvidenceRecord


_QUESTION_STOPWORDS = {
    "about",
    "after",
    "answer",
    "been",
    "before",
    "between",
    "does",
    "first",
    "from",
    "have",
    "into",
    "more",
    "name",
    "onto",
    "question",
    "than",
    "that",
    "their",
    "then",
    "they",
    "this",
    "time",
    "what",
    "when",
    "will",
    "with",
}

_YES_NO_STARTERS = {
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "was",
    "were",
}

_QUESTION_EXPANSIONS = {
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


def question_kind(question: str) -> str:
    """Return the strict verifier question kind."""

    first = (question.strip().split() or [""])[0].strip(":").lower()
    return "yesno" if first in _YES_NO_STARTERS else "factual"


def rank_evidence_records(
    question: str,
    records: list[EvidenceRecord],
    *,
    limit: int = 10,
) -> list[EvidenceRecord]:
    """Rank records by lexical question overlap for compact SLM prompts."""

    return sorted(
        records,
        key=lambda record: _record_relevance(question, record),
        reverse=True,
    )[:limit]


def build_strict_verifier_prompt(
    *,
    question: str,
    records: list[EvidenceRecord],
) -> str:
    """Build a compact evidence verifier prompt for small local models."""

    evidence = "\n\n".join(
        f"[{index}] source={record.source_id}\n{record.text}"
        for index, record in enumerate(records, start=1)
    )
    return "\n".join(
        [
            "You are ProofRAG's strict evidence verifier. Answer using ONLY the evidence snippets.",
            "",
            "Rules:",
            (
                '- For yes/no questions, answer exactly one of: "Yes", "No", '
                'or "Insufficient evidence".'
            ),
            (
                "- For factual questions, never answer Yes or No. Return only "
                'the requested person/place/date/time/object/fact, or "Insufficient evidence".'
            ),
            (
                "- If the question asks whether event A happened before/after "
                "event B, both events and their order must be explicit in the evidence."
            ),
            (
                "- If any required event, actor, object, date, time, or ordering is "
                'missing or ambiguous, answer "Insufficient evidence".'
            ),
            "- Do not infer from similar events. Do not use general topic similarity.",
            "- After the short answer, cite the specific evidence numbers in one sentence.",
            "",
            f"Question: {question}",
            "",
            "Evidence:",
            evidence,
            "",
            "Answer:",
        ]
    )


def build_compact_answer_prompt(
    *,
    question: str,
    records: list[EvidenceRecord],
) -> str:
    """Build a compact answer prompt over ranked evidence for small models."""

    evidence = "\n\n".join(
        f"[{record.source_id}]\n{record.text}" for record in records
    )
    return "\n".join(
        [
            "Answer the question using only the evidence snippets below.",
            'If the evidence is incomplete, answer exactly "Insufficient evidence."',
            "Keep the answer to one short sentence.",
            "Then add one citation sentence with only the supporting [source] ids.",
            "",
            f"Question: {question}",
            "",
            "Evidence:",
            evidence,
            "",
            "Answer:",
        ]
    )


def is_strict_abstention(answer: str) -> bool:
    """Return True when the verifier declined to answer."""

    return _normalized_answer(answer).startswith("insufficient evidence")


def is_uncertainty_abstention(answer: str) -> bool:
    """Return True when a generated answer admits missing evidence."""

    normalized = _normalized_answer(answer)
    uncertainty_patterns = (
        "no record",
        "no evidence",
        "no mention",
        "no information",
        "there is no record",
        "there is no evidence",
        "not enough information",
        "insufficient evidence",
        "cannot be confirmed",
        "cannot be determined",
    )
    return any(pattern in normalized for pattern in uncertainty_patterns)


def _normalized_answer(answer: str) -> str:
    return " ".join(answer.strip().lower().split())


def _record_relevance(question: str, record: EvidenceRecord) -> int:
    terms = _question_terms(question)
    text = record.text.lower()
    phrase_score = 0
    for length in (4, 3, 2):
        for index in range(max(0, len(terms) - length + 1)):
            phrase = " ".join(terms[index : index + length])
            if phrase and phrase in text:
                phrase_score += length * 2
    return sum(1 for term in terms if term in text) + phrase_score


def _question_terms(question: str) -> list[str]:
    terms = [
        term.lower()
        for term in re.findall(r"\w+", question)
        if len(term) > 3 and term.lower() not in _QUESTION_STOPWORDS
    ]
    expanded = list(terms)
    for term in terms:
        expanded.extend(_QUESTION_EXPANSIONS.get(term, ()))
    return expanded
