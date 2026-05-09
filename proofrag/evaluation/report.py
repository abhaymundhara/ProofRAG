from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from proofrag.evaluation.metrics import BenchmarkMetrics


class ExperimentLogSummary(BaseModel):
    """Aggregate summary for a ProofRAG ExperimentLogger JSONL file."""

    total_runs: int
    answer_allowed_count: int
    abstained_count: int
    answer_allowed_rate: float
    abstention_rate: float
    mean_coverage_score: float
    contradiction_count: int
    mean_contradictions: float
    retriever_backends: dict[str, int]
    generator_backends: dict[str, int]
    contract_inference_modes: dict[str, int]
    missing_slot_counts: dict[str, int]


def load_experiment_log(path: str | Path) -> list[dict[str, Any]]:
    """Load ExperimentLogger JSONL records."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def summarize_experiment_log(records: list[dict[str, Any]]) -> ExperimentLogSummary:
    """Summarize answer decisions, coverage, backends, and missing slots."""

    total = len(records)
    allowed = 0
    coverage_values: list[float] = []
    contradiction_values: list[int] = []
    retrievers: dict[str, int] = {}
    generators: dict[str, int] = {}
    contract_modes: dict[str, int] = {}
    missing_slots: dict[str, int] = {}

    for record in records:
        sufficiency = record.get("sufficiency") or {}
        summary = record.get("summary") or {}
        if bool(sufficiency.get("answer_allowed", summary.get("answer_allowed", False))):
            allowed += 1
        coverage_values.append(float(sufficiency.get("coverage_score", summary.get("coverage_score", 0.0))))
        contradictions = int(
            sufficiency.get("contradiction_count", summary.get("contradiction_count", 0))
        )
        contradiction_values.append(contradictions)

        _count(retrievers, str(summary.get("retriever_backend", "unknown")))
        _count(generators, str(summary.get("generator_backend", "unknown")))
        _count(contract_modes, str(summary.get("contract_inference", "unknown")))
        for slot in sufficiency.get("missing_required_slots", summary.get("missing_required_slots", [])):
            _count(missing_slots, str(slot))

    return ExperimentLogSummary(
        total_runs=total,
        answer_allowed_count=allowed,
        abstained_count=total - allowed,
        answer_allowed_rate=allowed / total if total else 0.0,
        abstention_rate=(total - allowed) / total if total else 0.0,
        mean_coverage_score=_mean(coverage_values),
        contradiction_count=sum(contradiction_values),
        mean_contradictions=_mean([float(value) for value in contradiction_values]),
        retriever_backends=retrievers,
        generator_backends=generators,
        contract_inference_modes=contract_modes,
        missing_slot_counts=missing_slots,
    )


def experiment_summary_markdown(summary: ExperimentLogSummary) -> str:
    """Render an experiment-log summary as Markdown."""

    lines = [
        "# ProofRAG Experiment Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total runs | {summary.total_runs} |",
        f"| Answer allowed | {summary.answer_allowed_count} |",
        f"| Abstained | {summary.abstained_count} |",
        f"| Answer allowed rate | {summary.answer_allowed_rate:.1%} |",
        f"| Abstention rate | {summary.abstention_rate:.1%} |",
        f"| Mean coverage score | {summary.mean_coverage_score:.3f} |",
        f"| Total contradictions | {summary.contradiction_count} |",
        f"| Mean contradictions | {summary.mean_contradictions:.3f} |",
        "",
        "## Backends",
        "",
        _counts_table("Retriever", summary.retriever_backends),
        "",
        _counts_table("Generator", summary.generator_backends),
        "",
        _counts_table("Contract inference", summary.contract_inference_modes),
        "",
        "## Missing Slots",
        "",
        _counts_table("Slot", summary.missing_slot_counts),
    ]
    return "\n".join(lines)


def _count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _counts_table(label: str, counts: dict[str, int]) -> str:
    lines = [f"| {label} | Count |", "| --- | ---: |"]
    if not counts:
        lines.append("| (none) | 0 |")
        return "\n".join(lines)
    for key in sorted(counts):
        lines.append(f"| {key} | {counts[key]} |")
    return "\n".join(lines)


def print_benchmark_report(results: list[dict], metrics: BenchmarkMetrics):
    print("\n" + "="*80)
    print("  PROOF-RAG TOY BENCHMARK REPORT")
    print("="*80)
    
    header = f"{'ID':<6} | {'EXP':<5} | {'ACT':<5} | {'COV':<5} | {'CONT':<4} | {'STATUS':<6} | {'MISSING SLOTS'}"
    print(header)
    print("-" * 80)
    
    for r in results:
        status = "PASS" if r["behavioural_pass"] else "FAIL"
        exp = "T" if r["expected_answer_allowed"] else "F"
        act = "T" if r["actual_answer_allowed"] else "F"
        missing = ", ".join(r["missing_slots"]) if r["missing_slots"] else "(none)"
        
        row = (f"{r['id']:<6} | {exp:<5} | {act:<5} | {r['coverage_score']:<5.1f} | "
               f"{r['contradiction_count']:<4} | {status:<6} | {missing}")
        print(row)
    
    print("\n" + "="*80)
    print("  AGGREGATE METRICS")
    print("="*80)
    print(f"Total Questions:         {metrics.total_questions}")
    print(f"Behavioural Pass Rate:   {metrics.behavioural_pass_rate:.1%} ({metrics.behavioural_pass_count}/{metrics.total_questions})")
    print(f"Answer Allowed Count:    {metrics.answer_allowed_count}")
    print(f"Abstained Count:         {metrics.abstained_count}")
    print(f"False Allow (Unsafe):    {metrics.false_allow_count} (Rate: {metrics.unsafe_answer_rate:.1%})")
    print(f"False Abstain:           {metrics.false_abstain_count}")
    print(f"Abstention Rate:         {metrics.abstention_rate:.1%}")
    print(f"Precision@Answered:      {metrics.precision_at_answered:.1%}")
    print(f"Mean Coverage Score:     {metrics.coverage_score_mean:.2f}")
    if metrics.latency_ms_mean:
        print(f"Mean Latency (ms):       {metrics.latency_ms_mean:.1f}")
    if metrics.total_tokens_mean:
        print(f"Mean Total Tokens:       {metrics.total_tokens_mean:.1f}")
    print("="*80 + "\n")
