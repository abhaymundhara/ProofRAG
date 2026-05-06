"""
cli.py — ProofRAG command-line interface.

Usage:
    python -m proofrag.cli ask --question "Who asked LiHua about the laptop warranty issue?"

The ``ask`` command runs the full ProofRAG pipeline end-to-end:
  1. Build a hardcoded EvidenceContract for the question.
  2. Retrieve evidence via DummyRetriever.
  3. Build an EvidenceLedger.
  4. Score sufficiency with RuleBasedSufficiencyScorer.
  5. Pack the strict context prompt.
  6. Generate a placeholder answer with DummyGenerator.
  7. Print all artefacts to stdout.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer

from proofrag.contracts.schema import EvidenceContract, EvidenceSlot
from proofrag.evidence.sufficiency import RuleBasedSufficiencyScorer
from proofrag.generation.dummy import DummyGenerator
from proofrag.logger import ExperimentLogger
from proofrag.packing.strict_context import StrictContextPacker
from proofrag.retrieval.dummy import DummyRetriever

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
) -> None:
    """Run the full ProofRAG pipeline for a given question."""

    typer.echo(f"\n🔍  ProofRAG v0.1  —  processing question...")

    # 1. Contract
    contract = _build_contract(question)

    # 2 & 3. Retrieve → Ledger
    retriever = DummyRetriever()
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
    generator = DummyGenerator()
    answer = generator.generate(prompt)

    # 7. Log experiment
    exp_logger = ExperimentLogger()
    run_id = exp_logger.log(
        question=question,
        contract=contract,
        ledger=ledger,
        report=report,
        packed_prompt=prompt,
        answer=answer,
    )

    # ── Print ────────────────────────────────────────────────────────────── #
    required_slot_ids = {s.slot_id for s in contract.required_slots}
    used_records = [
        rec for rec in ledger.records 
        if any(s in required_slot_ids for s in rec.supports_slots)
    ]
    ignored_count = retriever.total_docs - len(used_records)

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
