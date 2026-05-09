from __future__ import annotations

import json
import urllib.error

import pytest

from proofrag.generation.openai_compatible import OpenAICompatibleGenerator


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_openai_compatible_payload_and_auth(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse(
            {
                "choices": [
                    {"message": {"content": "Tom asked LiHua."}}
                ],
                "usage": {"total_tokens": 12},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    generator = OpenAICompatibleGenerator(
        model="local-model",
        base_url="http://localhost:8000/v1",
        api_key="test-key",
        timeout=7,
        max_tokens=32,
    )
    result = generator.generate_with_metadata("Prompt")

    assert result["content"] == "Tom asked LiHua."
    assert result["usage"]["total_tokens"] == 12
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["timeout"] == 7
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "local-model"
    assert captured["payload"]["messages"][0]["content"] == "Prompt"
    assert captured["payload"]["max_tokens"] == 32


def test_openai_compatible_connection_error(monkeypatch):
    def fake_urlopen(req, timeout):
        _ = req, timeout
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    generator = OpenAICompatibleGenerator(model="local-model")
    with pytest.raises(RuntimeError, match="OpenAI-compatible request failed"):
        generator.generate("Prompt")

