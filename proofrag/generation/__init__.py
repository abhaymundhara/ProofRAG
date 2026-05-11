"""
generation — Pluggable generation backends.
"""

from proofrag.generation.dummy import DummyGenerator
from proofrag.generation.ollama import OllamaGenerator
from proofrag.generation.openai_compatible import OpenAICompatibleGenerator
from proofrag.generation.strict_verifier import (
    build_strict_verifier_prompt,
    is_strict_abstention,
    question_kind,
    rank_evidence_records,
)
from proofrag.generation.transformers import TransformersGenerator

__all__ = [
    "build_strict_verifier_prompt",
    "DummyGenerator",
    "is_strict_abstention",
    "OllamaGenerator",
    "OpenAICompatibleGenerator",
    "question_kind",
    "rank_evidence_records",
    "TransformersGenerator",
]
