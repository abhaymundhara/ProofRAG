from __future__ import annotations

import builtins

import pytest

from proofrag.generation.transformers import TransformersGenerator


def test_transformers_generator_uses_injected_pipeline_and_trims_prompt():
    calls = []

    def fake_pipeline(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return [{"generated_text": f"{prompt} Tom asked LiHua."}]

    generator = TransformersGenerator(
        model="tiny-local",
        max_tokens=12,
        temperature=0.0,
        pipeline_obj=fake_pipeline,
    )

    assert generator.generate("Prompt:") == "Tom asked LiHua."
    assert calls[0][1]["max_new_tokens"] == 12
    assert calls[0][1]["do_sample"] is False


def test_transformers_generator_supports_sampling_kwargs():
    def fake_pipeline(prompt, **kwargs):
        _ = prompt
        return [{"generated_text": "answer"}]

    generator = TransformersGenerator(
        model="tiny-local",
        temperature=0.7,
        pipeline_obj=fake_pipeline,
    )

    assert generator.generate("Prompt") == "answer"


def test_transformers_generator_missing_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    generator = TransformersGenerator(model="tiny-local")

    with pytest.raises(RuntimeError, match="Transformers generation is optional"):
        generator.generate("Prompt")

