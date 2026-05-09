"""
dummy.py — DummyGenerator.

Returns a fixed string for deterministic tests and smoke runs. Production
pipelines should use the Ollama, OpenAI-compatible, or transformers backends.
"""

from __future__ import annotations

from .base import BaseGenerator


class DummyGenerator(BaseGenerator):
    """A dummy generator that returns a fixed response for testing."""

    DEFAULT_RESPONSE = "Generated answer would go here."

    def __init__(self, placeholder: str | None = None) -> None:
        self._placeholder = placeholder or self.DEFAULT_RESPONSE

    def generate(self, prompt: str) -> str:
        """Accept a packed prompt and return the configured fixed answer.

        Args:
            prompt: The evidence-gated prompt produced by StrictContextPacker.

        Returns:
            A deterministic string for tests and offline smoke runs.
        """
        # The prompt parameter is intentionally unused; the signature matches
        # the shared generator interface.
        _ = prompt
        return self._placeholder
