"""Anthropic provider mapping drops ragas' default sampling params."""

from unittest.mock import MagicMock

from ragas.llms.base import InstructorLLM, InstructorModelArgs


def _llm(**kwargs) -> InstructorLLM:
    return InstructorLLM(
        client=MagicMock(), model="claude-haiku-5-5", provider="anthropic", **kwargs
    )


def test_default_sampling_params_dropped_for_anthropic():
    params = _llm()._map_provider_params()
    assert "temperature" not in params
    assert "top_p" not in params
    assert params["max_tokens"] == 1024


def test_explicit_kwargs_are_kept():
    params = _llm(temperature=0.3)._map_provider_params()
    assert params["temperature"] == 0.3
    assert "top_p" not in params


def test_explicit_model_args_are_kept():
    params = _llm(model_args=InstructorModelArgs(top_p=0.9))._map_provider_params()
    assert params["top_p"] == 0.9
    assert "temperature" not in params


def test_other_providers_unchanged():
    llm = InstructorLLM(client=MagicMock(), model="mistral-large", provider="mistral")
    params = llm._map_provider_params()
    assert params["temperature"] == 0.01
    assert params["top_p"] == 0.1


def test_llm_factory_anthropic_has_no_sampling_params():
    import pytest

    anthropic = pytest.importorskip("anthropic")
    from ragas.llms import llm_factory

    llm = llm_factory(
        "claude-haiku-5-5",
        provider="anthropic",
        client=anthropic.Anthropic(api_key="sk-test"),
    )
    params = llm._map_provider_params()
    assert "temperature" not in params and "top_p" not in params
