"""
cli.py — ProofRAG command-line interface.

Usage:
    python -m proofrag.cli ask --question "Who asked LiHua about the laptop warranty issue?"

The ``ask`` command runs the full ProofRAG pipeline end-to-end:
  1. Infer or build an EvidenceContract for the question.
  2. Retrieve evidence via the configured retriever backend.
  3. Build an EvidenceLedger.
  4. Score sufficiency with RuleBasedSufficiencyScorer.
  5. Pack the strict context prompt.
  6. Generate with the configured generator backend, or abstain when required.
  7. Print all artefacts to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from proofrag.config import ProofRAGSettings, apply_cli_overrides, load_settings
from proofrag.contracts.adaptive import infer_adaptive_contract
from proofrag.contracts.infer import infer_contract_from_question
from proofrag.contracts.llm import infer_contract_with_llm
from proofrag.contracts.schema import EvidenceContract, EvidenceSlot
from proofrag.evidence.sufficiency import RuleBasedSufficiencyScorer
from proofrag.generation.dummy import DummyGenerator
from proofrag.generation.ollama import OllamaGenerator
from proofrag.generation.openai_compatible import OpenAICompatibleGenerator
from proofrag.generation.transformers import TransformersGenerator
from proofrag.logger import ExperimentLogger
from proofrag.packing.strict_context import StrictContextPacker
from proofrag.retrieval.base import BaseRetriever
from proofrag.retrieval.bm25 import BM25Retriever
from proofrag.retrieval.dummy import DummyRetriever
from proofrag.retrieval.hybrid import HybridRetriever
from proofrag.retrieval.iterative import ContractGapRetriever
from proofrag.retrieval.vector import (
    ChromaRetriever,
    FAISSRetriever,
    LanceDBRetriever,
    OptionalVectorDependencyError,
)

app = typer.Typer(
    name="proofrag",
    help="ProofRAG — evidence-contracted RAG for small language models (v0.1).",
    add_completion=False,
    no_args_is_help=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_DIVIDER = "=" * 72


def _print_section(title: str, content: str) -> None:
    typer.echo(f"\n{_DIVIDER}")
    typer.echo(f"  {title}")
    typer.echo(_DIVIDER)
    typer.echo(content)


def _build_contract(question: str) -> EvidenceContract:
    """Return a minimal hardcoded contract suitable for the demo question."""
    return EvidenceContract(
        question=question,
        query_type="factoid",
        slots=[
            EvidenceSlot(
                slot_id="who_asked",
                description="The person who asked LiHua about the laptop warranty issue",
                evidence_type="factual",
                required=True,
                min_sources=1,
            ),
            EvidenceSlot(
                slot_id="warranty_context",
                description="Context about the laptop warranty issue itself",
                evidence_type="contextual",
                required=True,
                min_sources=1,
            ),
        ],
        must_check_contradictions=True,
        strict_mode=True,
    )


def _resolve_contract(
    question: str,
    settings: ProofRAGSettings,
    generator: DummyGenerator | OllamaGenerator | OpenAICompatibleGenerator | TransformersGenerator | None = None,
) -> EvidenceContract:
    """Build the contract requested by runtime settings."""
    if settings.contract.inference == "rule_based":
        return infer_contract_from_question(question)
    if settings.contract.inference == "adaptive":
        return infer_adaptive_contract(question)
    if settings.contract.inference == "llm":
        if generator is None:
            raise typer.BadParameter("llm contract inference requires a generator")
        return infer_contract_with_llm(question=question, generator=generator)
    if settings.contract.inference != "demo":
        raise typer.BadParameter(
            "contract inference must be one of: demo, rule_based, adaptive, llm"
        )
    return _build_contract(question)


def _build_retriever(
    settings: ProofRAGSettings,
) -> BaseRetriever:
    """Build the configured retriever backend."""
    context_path = (
        Path(settings.retriever.context_path)
        if settings.retriever.context_path
        else None
    )
    if settings.retriever.backend == "dummy":
        return DummyRetriever(context_path=context_path)
    if settings.retriever.backend == "bm25":
        return BM25Retriever(
            context_path=context_path,
            top_k=settings.retriever.top_k,
        )
    if settings.retriever.backend == "hybrid":
        return HybridRetriever(
            context_path=context_path,
            top_k=settings.retriever.top_k,
            candidate_k=settings.retriever.candidate_k,
        )
    try:
        if settings.retriever.backend == "faiss":
            return FAISSRetriever(
                context_path=context_path,
                top_k=settings.retriever.top_k,
            )
        if settings.retriever.backend == "chroma":
            return ChromaRetriever(
                context_path=context_path,
                top_k=settings.retriever.top_k,
            )
        if settings.retriever.backend == "lancedb":
            return LanceDBRetriever(
                context_path=context_path,
                top_k=settings.retriever.top_k,
            )
    except OptionalVectorDependencyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    raise typer.BadParameter(
        "retriever backend must be one of: dummy, bm25, hybrid, faiss, chroma, lancedb"
    )


def _build_generator(
    settings: ProofRAGSettings,
) -> DummyGenerator | OllamaGenerator | OpenAICompatibleGenerator | TransformersGenerator:
    """Build the configured generator backend."""
    if settings.generator.backend == "dummy":
        return DummyGenerator()
    if settings.generator.backend == "ollama":
        return OllamaGenerator(
            model=settings.generator.model,
            base_url=settings.generator.base_url,
            timeout=settings.generator.timeout,
            temperature=settings.generator.temperature,
            max_tokens=settings.generator.max_tokens,
            endpoint_mode=settings.generator.endpoint_mode,
        )
    if settings.generator.backend in {"openai-compatible", "openai_compatible"}:
        return OpenAICompatibleGenerator(
            model=settings.generator.model,
            base_url=settings.generator.base_url,
            api_key=settings.generator.api_key,
            timeout=settings.generator.timeout,
            temperature=settings.generator.temperature,
            max_tokens=settings.generator.max_tokens,
        )
    if settings.generator.backend == "transformers":
        return TransformersGenerator(
            model=settings.generator.model,
            temperature=settings.generator.temperature,
            max_tokens=settings.generator.max_tokens,
        )
    raise typer.BadParameter(
        "generator backend must be one of: dummy, ollama, openai-compatible, transformers"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────


@app.command(name="version", hidden=True)
def _version_cmd() -> None:
    """Print ProofRAG version."""
    import proofrag
    typer.echo(f"ProofRAG {proofrag.__version__}")


@app.command()
def ask(
    question: Annotated[
        str,
        typer.Option("--question", "-q", help="The question to answer."),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Optional YAML/JSON runtime config file.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Override the JSONL experiment log path.",
        ),
    ] = None,
    retriever_backend: Annotated[
        str | None,
        typer.Option(
            "--retriever",
            help="Override retriever backend: dummy, bm25, hybrid, faiss, chroma, or lancedb.",
        ),
    ] = None,
    iterative_retrieval: Annotated[
        bool | None,
        typer.Option(
            "--iterative/--single-shot",
            help="Enable or disable contract-gap guided retrieval.",
        ),
    ] = None,
    max_retrieval_rounds: Annotated[
        int | None,
        typer.Option(
            "--max-retrieval-rounds",
            help="Maximum rounds when iterative retrieval is enabled.",
        ),
    ] = None,
    generator_backend: Annotated[
        str | None,
        typer.Option(
            "--generator",
            help="Override generator backend: dummy, ollama, openai-compatible, or transformers.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Override generator model name."),
    ] = None,
    contract_inference: Annotated[
        str | None,
        typer.Option(
            "--contract-inference",
            help="Override contract inference: demo, rule_based, adaptive, or llm.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print a compact machine-readable result."),
    ] = False,
) -> None:
    """Run the full ProofRAG pipeline for a given question."""

    settings = apply_cli_overrides(
        load_settings(config),
        output_path=str(output) if output else None,
        retriever_backend=retriever_backend,
        iterative_retrieval=iterative_retrieval,
        max_retrieval_rounds=max_retrieval_rounds,
        generator_backend=generator_backend,
        model=model,
        contract_inference=contract_inference,
    )

    if not json_output:
        typer.echo("\n🔍  ProofRAG v0.1  —  processing question...")

    generator = _build_generator(settings)

    # 1. Contract
    contract = _resolve_contract(question, settings, generator=generator)

    # 2 & 3. Retrieve → Ledger
    retriever = _build_retriever(settings)
    iterative_rounds: list[dict] = []
    if settings.retriever.iterative:
        iterative_result = ContractGapRetriever(
            retriever,
            max_rounds=settings.retriever.max_rounds,
        ).retrieve(question=question, contract=contract)
        ledger = iterative_result.ledger
        iterative_rounds = [
            retrieval_round.model_dump()
            for retrieval_round in iterative_result.rounds
        ]
    else:
        ledger = retriever.retrieve(question=question, contract=contract)

    # 4. Score
    scorer = RuleBasedSufficiencyScorer()
    report = scorer.score(contract=contract, ledger=ledger)

    # 5. Pack
    packer = StrictContextPacker()
    prompt = packer.pack(
        question=question,
        contract=contract,
        ledger=ledger,
        report=report,
    )

    # 6. Generate
    answer = generator.generate(prompt)

    # 7. Log experiment
    exp_logger = ExperimentLogger(output_path=Path(settings.logging.output_path))
    summary = {
        "answer_allowed": report.answer_allowed,
        "coverage_score": report.coverage_score,
        "missing_required_slots": report.missing_required_slots,
        "contradiction_count": report.contradiction_count,
        "retriever_total_docs": retriever.total_docs,
        "retriever_backend": settings.retriever.backend,
        "iterative_retrieval": settings.retriever.iterative,
        "retrieval_rounds": iterative_rounds,
        "generator_backend": settings.generator.backend,
        "contract_inference": settings.contract.inference,
    }
    run_id = exp_logger.log(
        question=question,
        contract=contract,
        ledger=ledger,
        report=report,
        packed_prompt=prompt if settings.logging.include_packed_prompt else "",
        answer=answer,
        run_config=settings.model_dump(),
        summary=summary,
    )

    # ── Print ────────────────────────────────────────────────────────────── #
    required_slot_ids = {s.slot_id for s in contract.required_slots}
    used_records = [
        rec for rec in ledger.records 
        if any(s in required_slot_ids for s in rec.supports_slots)
    ]
    ignored_count = retriever.total_docs - len(used_records)

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "run_id": run_id,
                    "question": question,
                    "answer": answer,
                    "sufficiency": report.model_dump(),
                    "summary": summary,
                    "log_path": str(exp_logger._path),
                },
                indent=2,
            )
        )
        return

    _print_section(
        "EVIDENCE CONTRACT",
        json.dumps(contract.model_dump(), indent=2),
    )

    _print_section(
        "SUFFICIENCY REPORT",
        json.dumps(report.model_dump(), indent=2),
    )

    _print_section("PACKED PROMPT", prompt)

    _print_section("DUMMY ANSWER", answer)

    typer.echo(f"\n{'─' * 72}")
    status = "✅  ANSWER ALLOWED" if report.answer_allowed else "🚫  ANSWER BLOCKED"
    typer.echo(f"  {status}  |  coverage={report.coverage_score:.0%}  |  "
               f"contradictions={report.contradiction_count}")
    typer.echo(f"  RETRIEVAL: total={retriever.total_docs} | used={len(used_records)} | ignored={ignored_count}")
    typer.echo(f"  run_id={run_id}")
    typer.echo(f"  logged → {exp_logger._path}")
    typer.echo(f"{'─' * 72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
