"""
transformers.py — Optional Hugging Face Transformers generator backend.

Transformers is intentionally optional so the default ProofRAG install remains
lightweight. Tests can inject a fake pipeline without importing transformers.
"""

from __future__ import annotations

from typing import Any

from proofrag.generation.base import BaseGenerator


class TransformersGenerator(BaseGenerator):
    """Local text-generation backend using `transformers.pipeline`."""

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        device: int | str | None = None,
        pipeline_obj: Any | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.device = device
        self._pipeline = pipeline_obj

    def generate(self, prompt: str) -> str:
        """Generate text for a packed ProofRAG prompt."""

        pipe = self._get_pipeline()
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature

        result = pipe(prompt, **kwargs)
        text = self._extract_text(result)
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text.strip()

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Transformers generation is optional. Install the `transformers` "
                "extra to use --generator transformers."
            ) from exc

        kwargs: dict[str, Any] = {"model": self.model}
        if self.device is not None:
            kwargs["device"] = self.device
        self._pipeline = pipeline("text-generation", **kwargs)
        return self._pipeline

    @staticmethod
    def _extract_text(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                return str(
                    first.get("generated_text")
                    or first.get("text")
                    or first.get("content")
                    or ""
                )
            return str(first)
        if isinstance(result, dict):
            return str(
                result.get("generated_text")
                or result.get("text")
                or result.get("content")
                or ""
            )
        return str(result or "")

