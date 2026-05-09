"""
schemas.py — API request and response models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for the `/ask` endpoint."""

    question: str
    retriever: str = "hybrid"
    generator: str = "dummy"
    contract_inference: str = "adaptive"
    iterative: bool = True
    max_retrieval_rounds: int = Field(default=2, ge=1)


class AskResponse(BaseModel):
    """Response body for the `/ask` endpoint."""

    question: str
    answer: str
    answer_allowed: bool
    sufficiency: dict[str, Any]
    summary: dict[str, Any]

