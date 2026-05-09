"""
config.py — Typed runtime configuration for ProofRAG.

The configuration layer intentionally keeps the default install lightweight.
YAML support uses PyYAML when it is installed and falls back to a small parser
for the simple nested mappings used by ``configs/default.yaml``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RetrieverSettings(BaseModel):
    """Retriever backend settings."""

    backend: str = "dummy"
    context_path: str | None = None
    top_k: int = Field(default=5, ge=1)
    candidate_k: int | None = Field(default=None, ge=1)
    iterative: bool = False
    max_rounds: int = Field(default=2, ge=1)


class GeneratorSettings(BaseModel):
    """Generator backend settings."""

    backend: str = "dummy"
    model: str = "qwen3.5:4b"
    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    timeout: int = Field(default=120, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(default=256, ge=1)
    endpoint_mode: str = "chat"


class ContractSettings(BaseModel):
    """Evidence contract inference settings."""

    inference: str = "demo"


class LoggingSettings(BaseModel):
    """Experiment logging settings."""

    output_path: str = "experiments/results/run_001.jsonl"
    include_packed_prompt: bool = True


class ProofRAGSettings(BaseModel):
    """Top-level ProofRAG settings."""

    retriever: RetrieverSettings = Field(default_factory=RetrieverSettings)
    generator: GeneratorSettings = Field(default_factory=GeneratorSettings)
    contract: ContractSettings = Field(default_factory=ContractSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def load_settings(config_path: str | Path | None = None) -> ProofRAGSettings:
    """Load settings from JSON/YAML/defaults plus ``PROOFRAG_*`` overrides."""

    if config_path is None:
        config_path = os.environ.get("PROOFRAG_CONFIG")
    if config_path is None:
        return _apply_env_overrides(ProofRAGSettings())

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        data = _load_yaml_mapping(raw)
    return _apply_env_overrides(ProofRAGSettings(**data))


def apply_cli_overrides(
    settings: ProofRAGSettings,
    *,
    output_path: str | None = None,
    retriever_backend: str | None = None,
    iterative_retrieval: bool | None = None,
    max_retrieval_rounds: int | None = None,
    generator_backend: str | None = None,
    model: str | None = None,
    contract_inference: str | None = None,
) -> ProofRAGSettings:
    """Return settings with CLI overrides applied."""

    data = settings.model_dump()
    if output_path is not None:
        data["logging"]["output_path"] = output_path
    if retriever_backend is not None:
        data["retriever"]["backend"] = retriever_backend
    if iterative_retrieval is not None:
        data["retriever"]["iterative"] = iterative_retrieval
    if max_retrieval_rounds is not None:
        data["retriever"]["max_rounds"] = max_retrieval_rounds
    if generator_backend is not None:
        data["generator"]["backend"] = generator_backend
    if model is not None:
        data["generator"]["model"] = model
    if contract_inference is not None:
        data["contract"]["inference"] = contract_inference
    return ProofRAGSettings(**data)


def _load_yaml_mapping(raw: str) -> dict[str, Any]:
    """Load a simple YAML mapping with an optional PyYAML fast path."""

    try:
        import yaml
    except ImportError:
        return _parse_simple_yaml(raw)

    parsed = yaml.safe_load(raw) or {}
    if not isinstance(parsed, dict):
        raise ValueError("ProofRAG config must be a mapping at the top level")
    return parsed


def _apply_env_overrides(settings: ProofRAGSettings) -> ProofRAGSettings:
    """Apply lightweight environment overrides without adding pydantic-settings."""

    environ = os.environ
    data = settings.model_dump()
    _set_if_present(data["logging"], "output_path", environ.get("PROOFRAG_OUTPUT_PATH"))
    _set_if_present(data["retriever"], "backend", environ.get("PROOFRAG_RETRIEVER_BACKEND"))
    _set_if_present(data["retriever"], "context_path", environ.get("PROOFRAG_CONTEXT_PATH"))
    _set_if_present(data["retriever"], "top_k", _env_int("PROOFRAG_TOP_K"))
    _set_if_present(data["retriever"], "candidate_k", _env_int("PROOFRAG_CANDIDATE_K"))
    _set_if_present(data["retriever"], "iterative", _env_bool("PROOFRAG_ITERATIVE_RETRIEVAL"))
    _set_if_present(data["retriever"], "max_rounds", _env_int("PROOFRAG_MAX_RETRIEVAL_ROUNDS"))
    _set_if_present(data["generator"], "backend", environ.get("PROOFRAG_GENERATOR_BACKEND"))
    _set_if_present(data["generator"], "model", environ.get("PROOFRAG_GENERATOR_MODEL"))
    _set_if_present(data["generator"], "base_url", environ.get("PROOFRAG_GENERATOR_BASE_URL"))
    _set_if_present(data["generator"], "api_key", environ.get("PROOFRAG_GENERATOR_API_KEY"))
    _set_if_present(data["generator"], "timeout", _env_int("PROOFRAG_GENERATOR_TIMEOUT"))
    _set_if_present(data["generator"], "temperature", _env_float("PROOFRAG_TEMPERATURE"))
    _set_if_present(data["generator"], "max_tokens", _env_int("PROOFRAG_MAX_TOKENS"))
    _set_if_present(data["generator"], "endpoint_mode", environ.get("PROOFRAG_ENDPOINT_MODE"))
    _set_if_present(data["contract"], "inference", environ.get("PROOFRAG_CONTRACT_INFERENCE"))
    return ProofRAGSettings(**data)


def _set_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def _env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value is not None else None


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value is not None else None


def _parse_simple_yaml(raw: str) -> dict[str, Any]:
    """Parse the limited YAML subset used by ProofRAG default configs."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_no, original_line in enumerate(raw.splitlines(), start=1):
        stripped = original_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML syntax on line {line_no}: {original_line}")

        indent = len(original_line) - len(original_line.lstrip(" "))
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid YAML indentation on line {line_no}")

        parent = stack[-1][1]
        if raw_value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value)

    return root


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
