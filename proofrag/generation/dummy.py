"""
dummy.py — DummyGenerator.

Returns a fixed placeholder string.  In v0.2+ this will be replaced by a
real LLM call (e.g. via the transformers or openai client).
"""

from __future__ import annotations

from .base import BaseGenerator


class DummyGenerator(BaseGenerator):
    """A dummy generator that returns a fixed response for testing."""

    DEFAULT_RESPONSE = "Generated answer would go here."

    def __init__(self, placeholder: str | None = None) -> None:
        self._placeholder = placeholder or self.DEFAULT_RESPONSE

    def generate(self, prompt: str) -> str:
        """Accept a packed prompt and return a placeholder answer.

        Args:
            prompt: The evidence-gated prompt produced by StrictContextPacker.

        Returns:
            A placeholder string (real LLM response in future versions).
        """
        # The prompt parameter is intentionally unused in v0.1 — the signature
        # matches the interface that real generators will implement.
        _ = prompt
        return self._placeholder
