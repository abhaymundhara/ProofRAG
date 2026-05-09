from __future__ import annotations

import pytest

from proofrag.api.schemas import AskRequest, AskResponse


def test_api_schema_defaults_are_lightweight():
    request = AskRequest(question="Who asked LiHua?")

    assert request.retriever == "hybrid"
    assert request.generator == "dummy"
    assert request.contract_inference == "adaptive"
    assert request.iterative is True


def test_api_response_schema():
    response = AskResponse(
        question="Q?",
        answer="A",
        answer_allowed=True,
        sufficiency={"answer_allowed": True},
        summary={"retriever_backend": "hybrid"},
    )

    assert response.answer_allowed is True
    assert response.summary["retriever_backend"] == "hybrid"


def test_create_app_when_fastapi_available():
    fastapi = pytest.importorskip("fastapi")
    _ = fastapi
    from fastapi.testclient import TestClient

    from proofrag.api.main import create_app

    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    response = client.post(
        "/ask",
        json={
            "question": "Who asked LiHua about the laptop warranty issue?",
            "retriever": "hybrid",
            "generator": "dummy",
            "contract_inference": "demo",
            "iterative": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer_allowed"] is True
    assert payload["summary"]["retriever_backend"] == "hybrid"


def test_api_supports_llm_contract_fallback_with_dummy_generator():
    fastapi = pytest.importorskip("fastapi")
    _ = fastapi
    from fastapi.testclient import TestClient

    from proofrag.api.main import create_app

    client = TestClient(create_app())
    response = client.post(
        "/ask",
        json={
            "question": "Who asked LiHua about the laptop warranty issue?",
            "retriever": "hybrid",
            "generator": "dummy",
            "contract_inference": "llm",
            "iterative": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["contract_inference"] == "llm"
    assert payload["answer_allowed"] is False
    assert "topic_context" in payload["sufficiency"]["missing_required_slots"]


def test_api_uses_environment_runtime_settings(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    _ = fastapi
    from fastapi.testclient import TestClient

    from proofrag.api.main import create_app

    monkeypatch.setenv("PROOFRAG_CONTEXT_PATH", "examples/context.json")
    monkeypatch.setenv("PROOFRAG_TOP_K", "2")
    monkeypatch.setenv("PROOFRAG_GENERATOR_MODEL", "env-model")

    client = TestClient(create_app())
    response = client.post(
        "/ask",
        json={
            "question": "Who asked LiHua about the laptop warranty issue?",
            "retriever": "hybrid",
            "generator": "dummy",
            "contract_inference": "demo",
            "iterative": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["retriever_backend"] == "hybrid"
    assert payload["summary"]["generator_model"] == "env-model"
    assert payload["answer_allowed"] is True
