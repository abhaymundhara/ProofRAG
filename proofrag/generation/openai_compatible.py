"""
openai_compatible.py — OpenAI-compatible chat completion backend.

This backend uses the standard library HTTP client so ProofRAG does not need to
depend on a vendor SDK. It works with local or hosted servers that expose the
``/chat/completions`` API shape.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from proofrag.generation.base import BaseGenerator


class OpenAICompatibleGenerator(BaseGenerator):
    """Generator backend for OpenAI-compatible chat completion servers."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        timeout: int = 120,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Return generated message content for a packed ProofRAG prompt."""

        return self.generate_with_metadata(prompt)["content"]

    def generate_with_metadata(self, prompt: str) -> dict[str, Any]:
        """Call the chat completions endpoint and return content plus raw data."""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        raw = self._post(f"{self.base_url}/chat/completions", payload)
        choices = raw.get("choices") or []
        if not choices:
            return {"content": "", "raw": raw}
        message = choices[0].get("message") or {}
        return {
            "content": message.get("content", ""),
            "raw": raw,
            "usage": raw.get("usage", {}),
        }

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible HTTP error {exc.code}: {body or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"OpenAI-compatible request failed for {self.base_url}: {exc}"
            ) from exc

