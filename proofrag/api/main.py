"""
main.py — Optional FastAPI application factory.

FastAPI is an optional production dependency. Importing this module is safe in
the lightweight install; calling ``create_app`` requires FastAPI to be installed.
"""

from __future__ import annotations

from proofrag.api.schemas import AskRequest, AskResponse
from proofrag.cli import (
    _build_generator,
    _build_retriever,
    _resolve_contract,
)
from proofrag.config import apply_cli_overrides, load_settings
from proofrag.evidence.sufficiency import RuleBasedSufficiencyScorer
from proofrag.packing.strict_context import StrictContextPacker
from proofrag.retrieval.iterative import ContractGapRetriever


def create_app():
    """Create and return the FastAPI app."""

    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI server support is optional. Install the `api` extra to use it."
        ) from exc

    app = FastAPI(
        title="ProofRAG",
        description="Evidence-contracted RAG API",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        settings = apply_cli_overrides(
            load_settings(),
            retriever_backend=request.retriever,
            iterative_retrieval=request.iterative,
            max_retrieval_rounds=request.max_retrieval_rounds,
            generator_backend=request.generator,
            contract_inference=request.contract_inference,
        )
        generator = _build_generator(settings)
        contract = _resolve_contract(request.question, settings, generator=generator)
        retriever = _build_retriever(settings)
        if settings.retriever.iterative:
            retrieval_result = ContractGapRetriever(
                retriever,
                max_rounds=settings.retriever.max_rounds,
            ).retrieve(question=request.question, contract=contract)
            ledger = retrieval_result.ledger
            retrieval_rounds = [round_.model_dump() for round_ in retrieval_result.rounds]
        else:
            ledger = retriever.retrieve(question=request.question, contract=contract)
            retrieval_rounds = []

        scorer = RuleBasedSufficiencyScorer()
        report = scorer.score(contract=contract, ledger=ledger)
        prompt = StrictContextPacker().pack(
            question=request.question,
            contract=contract,
            ledger=ledger,
            report=report,
        )
        answer = generator.generate(prompt)
        summary = {
            "retriever_backend": settings.retriever.backend,
            "generator_backend": settings.generator.backend,
            "generator_model": settings.generator.model,
            "contract_inference": settings.contract.inference,
            "retrieval_rounds": retrieval_rounds,
            "coverage_score": report.coverage_score,
            "missing_required_slots": report.missing_required_slots,
            "contradiction_count": report.contradiction_count,
        }
        return AskResponse(
            question=request.question,
            answer=answer,
            answer_allowed=report.answer_allowed,
            sufficiency=report.model_dump(),
            summary=summary,
        )

    return app


try:
    app = create_app()
except RuntimeError:
    app = None
