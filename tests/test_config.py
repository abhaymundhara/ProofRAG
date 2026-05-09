from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from proofrag.cli import app
from proofrag.config import apply_cli_overrides, load_settings


def test_default_settings_are_lightweight():
    settings = load_settings()

    assert settings.retriever.backend == "dummy"
    assert settings.retriever.iterative is False
    assert settings.retriever.max_rounds == 2
    assert settings.generator.backend == "dummy"
    assert settings.contract.inference == "demo"
    assert settings.logging.output_path == "experiments/results/run_001.jsonl"


def test_loads_default_yaml_config():
    settings = load_settings("configs/default.yaml")

    assert settings.retriever.top_k == 5
    assert settings.retriever.iterative is False
    assert settings.generator.model == "qwen3.5:4b"
    assert settings.generator.endpoint_mode == "chat"
    assert settings.logging.include_packed_prompt is True


def test_loads_json_config(tmp_path: Path):
    config_path = tmp_path / "proofrag.json"
    config_path.write_text(
        json.dumps(
            {
                "generator": {"backend": "ollama", "model": "local-test"},
                "logging": {"output_path": str(tmp_path / "run.jsonl")},
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.generator.backend == "ollama"
    assert settings.generator.model == "local-test"
    assert settings.logging.output_path == str(tmp_path / "run.jsonl")


def test_load_settings_reads_config_path_from_environment(
    tmp_path: Path,
    monkeypatch,
):
    config_path = tmp_path / "proofrag.json"
    config_path.write_text(
        json.dumps({"retriever": {"backend": "bm25"}, "contract": {"inference": "adaptive"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROOFRAG_CONFIG", str(config_path))

    settings = load_settings()

    assert settings.retriever.backend == "bm25"
    assert settings.contract.inference == "adaptive"


def test_load_settings_applies_environment_scalar_overrides(monkeypatch):
    monkeypatch.setenv("PROOFRAG_RETRIEVER_BACKEND", "hybrid")
    monkeypatch.setenv("PROOFRAG_CONTEXT_PATH", "examples/context.json")
    monkeypatch.setenv("PROOFRAG_TOP_K", "3")
    monkeypatch.setenv("PROOFRAG_CANDIDATE_K", "7")
    monkeypatch.setenv("PROOFRAG_ITERATIVE_RETRIEVAL", "true")
    monkeypatch.setenv("PROOFRAG_MAX_RETRIEVAL_ROUNDS", "4")
    monkeypatch.setenv("PROOFRAG_GENERATOR_BACKEND", "ollama")
    monkeypatch.setenv("PROOFRAG_GENERATOR_MODEL", "qwen-local")
    monkeypatch.setenv("PROOFRAG_GENERATOR_BASE_URL", "http://localhost:11435")
    monkeypatch.setenv("PROOFRAG_GENERATOR_API_KEY", "secret")
    monkeypatch.setenv("PROOFRAG_GENERATOR_TIMEOUT", "30")
    monkeypatch.setenv("PROOFRAG_TEMPERATURE", "0.2")
    monkeypatch.setenv("PROOFRAG_MAX_TOKENS", "128")
    monkeypatch.setenv("PROOFRAG_ENDPOINT_MODE", "generate")
    monkeypatch.setenv("PROOFRAG_CONTRACT_INFERENCE", "llm")
    monkeypatch.setenv("PROOFRAG_OUTPUT_PATH", "experiments/results/env.jsonl")

    settings = load_settings()

    assert settings.retriever.backend == "hybrid"
    assert settings.retriever.context_path == "examples/context.json"
    assert settings.retriever.top_k == 3
    assert settings.retriever.candidate_k == 7
    assert settings.retriever.iterative is True
    assert settings.retriever.max_rounds == 4
    assert settings.generator.backend == "ollama"
    assert settings.generator.model == "qwen-local"
    assert settings.generator.base_url == "http://localhost:11435"
    assert settings.generator.api_key == "secret"
    assert settings.generator.timeout == 30
    assert settings.generator.temperature == 0.2
    assert settings.generator.max_tokens == 128
    assert settings.generator.endpoint_mode == "generate"
    assert settings.contract.inference == "llm"
    assert settings.logging.output_path == "experiments/results/env.jsonl"


def test_load_settings_rejects_invalid_boolean_environment(monkeypatch):
    monkeypatch.setenv("PROOFRAG_ITERATIVE_RETRIEVAL", "maybe")

    try:
        load_settings()
    except ValueError as exc:
        assert "PROOFRAG_ITERATIVE_RETRIEVAL" in str(exc)
    else:
        raise AssertionError("Expected invalid boolean environment value to fail")


def test_cli_overrides_do_not_mutate_original_settings():
    settings = load_settings()
    updated = apply_cli_overrides(
        settings,
        output_path="experiments/results/override.jsonl",
        retriever_backend="bm25",
        iterative_retrieval=True,
        max_retrieval_rounds=3,
        generator_backend="ollama",
        model="qwen-test",
        contract_inference="rule_based",
    )

    assert settings.generator.backend == "dummy"
    assert updated.retriever.backend == "bm25"
    assert updated.retriever.iterative is True
    assert updated.retriever.max_rounds == 3
    assert updated.generator.backend == "ollama"
    assert updated.generator.model == "qwen-test"
    assert updated.contract.inference == "rule_based"
    assert updated.logging.output_path == "experiments/results/override.jsonl"


def test_cli_overrides_take_precedence_over_environment(monkeypatch):
    monkeypatch.setenv("PROOFRAG_RETRIEVER_BACKEND", "bm25")
    monkeypatch.setenv("PROOFRAG_GENERATOR_MODEL", "env-model")

    updated = apply_cli_overrides(
        load_settings(),
        retriever_backend="hybrid",
        model="cli-model",
    )

    assert updated.retriever.backend == "hybrid"
    assert updated.generator.model == "cli-model"


def test_cli_ask_accepts_config_and_logs_runtime_metadata(tmp_path: Path):
    output_path = tmp_path / "ask.jsonl"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ask",
            "--question",
            "Who asked LiHua about the laptop warranty issue?",
            "--config",
            "configs/default.yaml",
            "--output",
            str(output_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert record["run_config"]["generator"]["backend"] == "dummy"
    assert record["summary"]["retriever_backend"] == "dummy"
    assert record["summary"]["iterative_retrieval"] is False
    assert record["summary"]["generator_backend"] == "dummy"
    assert record["summary"]["contract_inference"] == "demo"
    assert "sufficiency" in json.loads(result.output)


def test_cli_ask_can_enable_iterative_retrieval(tmp_path: Path):
    output_path = tmp_path / "iterative.jsonl"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ask",
            "--question",
            "Who asked LiHua about the laptop warranty issue?",
            "--config",
            "configs/default.yaml",
            "--iterative",
            "--max-retrieval-rounds",
            "2",
            "--output",
            str(output_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert record["summary"]["iterative_retrieval"] is True
    assert len(record["summary"]["retrieval_rounds"]) >= 1


def test_cli_llm_contract_inference_falls_back_with_dummy_generator(tmp_path: Path):
    output_path = tmp_path / "llm_contract.jsonl"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ask",
            "--question",
            "Who asked LiHua about the laptop warranty issue?",
            "--contract-inference",
            "llm",
            "--output",
            str(output_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert record["summary"]["contract_inference"] == "llm"
    assert "who_asked" in [slot["slot_id"] for slot in record["contract"]["slots"]]
