#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from proofrag.evaluation.faithfulness import (
    FaithfulnessReport,
    claim_level_faithfulness,
    judge_faithfulness_with_llm,
)
from proofrag.evaluation.minirag_adapter import MiniRAGOutputAdapter
from proofrag.generation.base import BaseGenerator
from proofrag.generation.ollama import OllamaGenerator
from proofrag.generation.openai_compatible import OpenAICompatibleGenerator
from proofrag.generation.transformers import TransformersGenerator


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score claim-level or optional LLM-judge groundedness for MiniRAG "
            "and MiniRAG+ProofRAG answers against retrieved source snippets."
        )
    )
    parser.add_argument("--results", required=True, help="ProofRAG result JSONL.")
    parser.add_argument("--minirag-export", required=True, help="MiniRAG export JSONL.")
    parser.add_argument("--summary-json", required=True, help="Summary JSON output.")
    parser.add_argument("--table-md", required=True, help="Markdown table output.")
    parser.add_argument(
        "--scorer",
        choices=("claim", "llm-judge"),
        default="claim",
        help="Faithfulness scorer. `claim` is deterministic and default.",
    )
    parser.add_argument(
        "--judge-backend",
        choices=("ollama", "openai-compatible", "transformers"),
        default="ollama",
        help="Generator backend used when --scorer llm-judge.",
    )
    parser.add_argument("--judge-model", default="qwen3.5:4b")
    parser.add_argument("--judge-base-url", default="http://localhost:11434")
    parser.add_argument("--judge-api-key")
    parser.add_argument("--judge-timeout", type=int, default=120)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument(
        "--judge-endpoint-mode",
        choices=("chat", "generate"),
        default="chat",
        help="Ollama endpoint mode for --judge-backend ollama.",
    )
    args = parser.parse_args()

    result_rows = _load_jsonl(Path(args.results))
    evidence_by_id = _evidence_by_id(Path(args.minirag_export))
    judge_generator = _build_judge_generator(args) if args.scorer == "llm-judge" else None
    rows = [
        _score_row(
            row,
            evidence_by_id.get(str(row.get("id")), []),
            scorer=args.scorer,
            judge_generator=judge_generator,
        )
        for row in result_rows
    ]
    summary = _summarize(rows)

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "scorer": args.scorer,
                "judge_backend": args.judge_backend if judge_generator else None,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    table_path = Path(args.table_md)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_markdown(summary) + "\n", encoding="utf-8")

    print(f"Wrote faithfulness summary to {summary_path}")
    print(f"Wrote faithfulness table to {table_path}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _evidence_by_id(path: Path) -> dict[str, list[str]]:
    adapter = MiniRAGOutputAdapter()
    evidence: dict[str, list[str]] = {}
    for item in adapter.load_export(str(path)):
        snippets: list[str] = []
        for context in item.retrieved_context:
            for source_row in adapter.extract_source_rows(context.get("text", "")):
                content = source_row.get("content", "").strip()
                if content:
                    snippets.append(content)
        evidence[item.id] = snippets
    return evidence


def _score_row(
    row: dict[str, Any],
    evidence_texts: list[str],
    *,
    scorer: str = "claim",
    judge_generator: BaseGenerator | None = None,
) -> dict[str, Any]:
    baseline_answer = str(row.get("baseline_answer", ""))
    proofrag_answer = str(row.get("proofrag_generated_answer", ""))
    baseline = _score_answer(
        answer=baseline_answer,
        evidence_texts=evidence_texts,
        scorer=scorer,
        judge_generator=judge_generator,
    )
    proofrag = _score_answer(
        answer=proofrag_answer,
        evidence_texts=evidence_texts,
        scorer=scorer,
        judge_generator=judge_generator,
    )
    return {
        "id": row.get("id", ""),
        "evidence_count": len(evidence_texts),
        "scorer": scorer,
        "baseline_groundedness": baseline.groundedness,
        "baseline_unsupported_claims": baseline.unsupported_claim_count,
        "proofrag_groundedness": proofrag.groundedness,
        "proofrag_unsupported_claims": proofrag.unsupported_claim_count,
    }


def _score_answer(
    *,
    answer: str,
    evidence_texts: list[str],
    scorer: str,
    judge_generator: BaseGenerator | None,
) -> FaithfulnessReport:
    if scorer == "claim":
        return claim_level_faithfulness(answer=answer, evidence_texts=evidence_texts)
    if scorer == "llm-judge":
        if judge_generator is None:
            raise ValueError("--scorer llm-judge requires a judge generator")
        return judge_faithfulness_with_llm(
            answer=answer,
            evidence_texts=evidence_texts,
            generator=judge_generator,
        )
    raise ValueError(f"unknown scorer: {scorer}")


def _build_judge_generator(args: argparse.Namespace) -> BaseGenerator:
    if args.judge_backend == "ollama":
        return OllamaGenerator(
            model=args.judge_model,
            base_url=args.judge_base_url,
            timeout=args.judge_timeout,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            endpoint_mode=args.judge_endpoint_mode,
        )
    if args.judge_backend == "openai-compatible":
        return OpenAICompatibleGenerator(
            model=args.judge_model,
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            timeout=args.judge_timeout,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
        )
    if args.judge_backend == "transformers":
        return TransformersGenerator(
            model=args.judge_model,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
        )
    raise ValueError(f"unknown judge backend: {args.judge_backend}")


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "total": total,
        "baseline_mean_groundedness": _mean(
            row["baseline_groundedness"] for row in rows
        ),
        "proofrag_mean_groundedness": _mean(
            row["proofrag_groundedness"] for row in rows
        ),
        "baseline_unsupported_claims": sum(
            int(row["baseline_unsupported_claims"]) for row in rows
        ),
        "proofrag_unsupported_claims": sum(
            int(row["proofrag_unsupported_claims"]) for row in rows
        ),
    }


def _mean(values: Any) -> float:
    items = list(values)
    return sum(float(value) for value in items) / len(items) if items else 0.0


def _markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "| Method | Total | Mean Groundedness | Unsupported Claims |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| MiniRAG | {summary['total']} | "
                f"{summary['baseline_mean_groundedness']:.1%} | "
                f"{summary['baseline_unsupported_claims']} |"
            ),
            (
                f"| MiniRAG+ProofRAG | {summary['total']} | "
                f"{summary['proofrag_mean_groundedness']:.1%} | "
                f"{summary['proofrag_unsupported_claims']} |"
            ),
        ]
    )


if __name__ == "__main__":
    main()
