"""
Tests for _map_anthropic_params in InstructorLLM.

Regression tests for: https://github.com/vibrantlabsai/ragas/issues/2674

Anthropic deprecated top_p for models after Claude Opus 4.6. Sending top_p < 0.99
alongside temperature causes a 400 Bad Request error:
    'temperature and top_p cannot both be specified for this model'

These tests verify that _map_anthropic_params correctly strips top_p in all cases
where it would cause a rejection.
"""

from unittest.mock import MagicMock

import pytest

from ragas.llms.base import InstructorLLM, InstructorModelArgs


def _make_anthropic_llm(**model_args_overrides) -> InstructorLLM:
    """Create an InstructorLLM with anthropic provider and a mock client."""
    mock_client = MagicMock()
    mock_client.__class__.__name__ = "Instructor"

    base_args = InstructorModelArgs()
    args_dict = base_args.model_dump()
    args_dict.update(model_args_overrides)

    llm = InstructorLLM.__new__(InstructorLLM)
    llm.client = mock_client
    llm.model = "claude-sonnet-4-6"
    llm.provider = "anthropic"
    llm.model_args = {k: v for k, v in args_dict.items() if k != "system_prompt"}
    llm.system_prompt = None
    llm.cache = None
    llm.is_async = False
    return llm


class TestMapAnthropicParams:
    def test_default_top_p_is_removed(self):
        """Default top_p=0.1 must be stripped — it causes 400 on current Anthropic models."""
        llm = _make_anthropic_llm()
        assert llm.model_args["top_p"] == 0.1  # confirm default is present before mapping

        result = llm._map_anthropic_params()

        assert "top_p" not in result, (
            "top_p=0.1 was not removed — this would cause a 400 Bad Request on Anthropic"
        )

    def test_top_p_below_threshold_is_removed(self):
        """Any top_p < 0.99 must be removed."""
        for value in [0.0, 0.1, 0.5, 0.98]:
            llm = _make_anthropic_llm(top_p=value)
            result = llm._map_anthropic_params()
            assert "top_p" not in result, (
                f"top_p={value} was not removed — Anthropic rejects values below 0.99"
            )

    def test_top_p_at_threshold_is_kept(self):
        """top_p=0.99 is explicitly allowed by Anthropic for backwards compatibility."""
        llm = _make_anthropic_llm(top_p=0.99)
        result = llm._map_anthropic_params()
        assert result["top_p"] == 0.99

    def test_top_p_above_threshold_is_kept(self):
        """top_p=1.0 should also be kept."""
        llm = _make_anthropic_llm(top_p=1.0)
        result = llm._map_anthropic_params()
        assert result["top_p"] == 1.0

    def test_temperature_and_other_params_preserved(self):
        """Stripping top_p must not affect temperature, max_tokens, or other params."""
        llm = _make_anthropic_llm(temperature=0.5, max_tokens=2048)
        result = llm._map_anthropic_params()
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 2048

    def test_no_top_p_in_model_args(self):
        """If top_p was never set, _map_anthropic_params should not crash."""
        llm = _make_anthropic_llm()
        llm.model_args.pop("top_p", None)  # remove it entirely
        result = llm._map_anthropic_params()
        assert "top_p" not in result

    def test_map_provider_params_routes_to_anthropic(self):
        """_map_provider_params must route 'anthropic' provider to _map_anthropic_params."""
        llm = _make_anthropic_llm()
        result = llm._map_provider_params()
        # top_p=0.1 default must have been stripped by the Anthropic-specific path
        assert "top_p" not in result, (
            "_map_provider_params did not route to _map_anthropic_params — "
            "top_p was not stripped for anthropic provider"
        )
