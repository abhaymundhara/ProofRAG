"""
generation — Pluggable generation backends.
"""

from proofrag.generation.dummy import DummyGenerator
from proofrag.generation.ollama import OllamaGenerator
from proofrag.generation.openai_compatible import OpenAICompatibleGenerator
from proofrag.generation.transformers import TransformersGenerator

__all__ = [
    "DummyGenerator",
    "OllamaGenerator",
    "OpenAICompatibleGenerator",
    "TransformersGenerator",
]
