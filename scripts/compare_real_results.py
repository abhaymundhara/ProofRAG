#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify(row: dict[str, Any]) -> str:
    answer_allowed = bool(row.get("answer_allowed"))
    model_called = bool(row.get("model_called"))
    correct = bool(row.get("correct_when_answered"))
    contains_gold = bool(row.get("contains_gold_answer_raw"))
    missing_slots = row.get("missing_required_slots") or []
    contradiction_count = int(row.get("contradiction_count") or 0)

    if not answer_allowed:
        if contradiction_count > 0:
            return "blocked_contradiction"
        if missing_slots:
            return "blocked_missing_evidence"
        return "blocked_other"

    if answer_allowed and not model_called:
        return "allowed_but_model_not_called"

    if correct:
        return "correct"

    if contains_gold and not correct:
        return "gold_present_but_semantically_wrong"

    return "incorrect"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ProofRAG-over-MiniRAG real result JSONL files."
    )
    parser.add_argument("--input", required=True, help="Path to result JSONL file.")
    args = parser.parse_args()

    path = Path(args.input)
    rows = load_jsonl(path)

    buckets: dict[str, int] = {}
    for row in rows:
        label = classify(row)
        buckets[label] = buckets.get(label, 0) + 1

    total = len(rows)
    answered = sum(1 for row in rows if row.get("model_called"))
    allowed = sum(1 for row in rows if row.get("answer_allowed"))
    abstained = total - allowed
    correct = sum(1 for row in rows if row.get("correct_when_answered"))
    gold_present = sum(1 for row in rows if row.get("contains_gold_answer_raw"))

    accuracy_when_called = (correct / answered * 100.0) if answered else 0.0
    allow_rate = (allowed / total * 100.0) if total else 0.0

    print()
    print("=" * 92)
    print("  PROOFRAG OVER MINIRAG REAL RESULT COMPARISON")
    print("=" * 92)
    print(f"Input:                  {path}")
    print(f"Total:                  {total}")
    print(f"Allowed:                {allowed}")
    print(f"Abstained/Blocked:      {abstained}")
    print(f"Model Called:           {answered}")
    print(f"Correct When Answered:  {correct}")
    print(f"Gold String Present:    {gold_present}")
    print(f"Allow Rate:             {allow_rate:.1f}%")
    print(f"Accuracy When Called:   {accuracy_when_called:.1f}%")
    print("-" * 92)

    for key in sorted(buckets):
        print(f"{key:35} {buckets[key]}")

    print("-" * 92)
    print("ID                   | ALLOWED | CALLED | GOLD | CORRECT | CLASS")
    print("-" * 92)

    for row in rows:
        print(
            f"{row.get('id', ''):20} | "
            f"{str(row.get('answer_allowed')):7} | "
            f"{str(row.get('model_called')):6} | "
            f"{str(row.get('contains_gold_answer_raw')):4} | "
            f"{str(row.get('correct_when_answered')):7} | "
            f"{classify(row)}"
        )

    print("=" * 92)


if __name__ == "__main__":
    main()