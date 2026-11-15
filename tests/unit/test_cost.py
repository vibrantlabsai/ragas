import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from ragas.cost import (
    CostCallbackHandler,
    TokenUsage,
    get_token_usage_for_anthropic,
    get_token_usage_for_azure_ai,
    get_token_usage_for_bedrock,
    get_token_usage_for_openai,
)

"""
TODO: things to test
- get usage from LLM Result
- estimate cost works for different API providers 
- openai with multiple n
- anthropic
- anthropic with multiple n
"""


def test_token_usage():
    x = TokenUsage(input_tokens=10, output_tokens=20)
    y = TokenUsage(input_tokens=5, output_tokens=15)
    assert (x + y).input_tokens == 15
    assert (x + y).output_tokens == 35

    with pytest.raises(ValueError):
        x.model = "openai"
        y.model = "gpt3"
        _ = x + y

    # test equals
    assert x == x
    assert y != x
    z = TokenUsage(input_tokens=10, output_tokens=20)
    z_with_model = TokenUsage(input_tokens=10, output_tokens=20, model="openai")
    z_same_with_model = TokenUsage(input_tokens=10, output_tokens=20, model="openai")
    assert z_with_model != z
    assert z_same_with_model == z_with_model

    # test same model
    assert z_with_model.is_same_model(z_same_with_model)
    assert not z_with_model.is_same_model(z)


def test_token_usage_cost():
    x = TokenUsage(input_tokens=10, output_tokens=20)
    assert x.cost(cost_per_input_token=0.1, cost_per_output_token=0.2) == 5.0


openai_llm_result = LLMResult(
    generations=[[ChatGeneration(message=AIMessage(content="Hello, world!"))]],
    llm_output={
        "token_usage": {
            "completion_tokens": 10,
            "prompt_tokens": 10,
            "total_tokens": 20,
        },
        "model_name": "gpt-4o",
        "system_fingerprint": "fp_2eie",
    },
)

anthropic_llm_result = LLMResult(
    generations=[
        [
            ChatGeneration(
                message=AIMessage(
                    content="Hello, world!",
                    response_metadata={
                        "id": "msg_01UHjFfUr",
                        "model": "claude-3-opus-20240229",
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 9, "output_tokens": 12},
                    },
                )
            )
        ]
    ],
    llm_output={},
)

bedrock_llama_result = LLMResult(
    generations=[
        [
            ChatGeneration(
                text="Hello, world!",
                message=AIMessage(
                    content="Hello, world!",
                    response_metadata={
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 10,
                            "total_tokens": 20,
                        },
                        "stop_reason": "stop",
                        "model_id": "us.meta.llama3-1-70b-instruct-v1:0",
                    },
                ),
            )
        ]
    ],
    llm_output={},
)

bedrock_claude_result = LLMResult(
    generations=[
        [
            ChatGeneration(
                text="Hello, world!",
                message=AIMessage(
                    content="Hello, world!",
                    response_metadata={
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 10,
                            "total_tokens": 20,
                        },
                        "stop_reason": "end_turn",
                        "model_id": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                    },
                ),
            )
        ]
    ],
    llm_output={},
)

azure_ai_result = LLMResult(
    generations=[[ChatGeneration(message=AIMessage(content="Hello, world!"))]],
    llm_output={
        "token_usage": {
            "input_tokens": 10,
            "output_tokens": 10,
            "total_tokens": 20,
        },
        "model_name": "mistral-small-2503",
    },
)


def test_parse_llm_results():
    # openai
    token_usage = get_token_usage_for_openai(openai_llm_result)
    assert token_usage == TokenUsage(input_tokens=10, output_tokens=10, model="gpt-4o")

    # anthropic
    token_usage = get_token_usage_for_anthropic(anthropic_llm_result)
    assert token_usage == TokenUsage(
        input_tokens=9, output_tokens=12, model="claude-3-opus-20240229"
    )

    # Bedrock LLaMa
    token_usage = get_token_usage_for_bedrock(bedrock_llama_result)
    assert token_usage == TokenUsage(
        input_tokens=10, output_tokens=10, model="us.meta.llama3-1-70b-instruct-v1:0"
    )

    # Bedrock Claude
    token_usage = get_token_usage_for_bedrock(bedrock_claude_result)
    assert token_usage == TokenUsage(
        input_tokens=10,
        output_tokens=10,
        model="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    )

    # Azure AI
    token_usage = get_token_usage_for_azure_ai(azure_ai_result)
    assert token_usage == TokenUsage(
        input_tokens=10, output_tokens=10, model="mistral-small-2503"
    )


def test_azure_ai_edge_cases():
    # Test with None llm_output
    empty_result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="Hello, world!"))]],
        llm_output=None,
    )
    token_usage = get_token_usage_for_azure_ai(empty_result)
    assert token_usage == TokenUsage(input_tokens=0, output_tokens=0)

    # Test with empty llm_output
    empty_llm_output_result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="Hello, world!"))]],
        llm_output={},
    )
    token_usage = get_token_usage_for_azure_ai(empty_llm_output_result)
    assert token_usage == TokenUsage(input_tokens=0, output_tokens=0)

    # Test with missing token_usage field
    no_token_usage_result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="Hello, world!"))]],
        llm_output={"model_name": "mistral-small-2503"},
    )
    token_usage = get_token_usage_for_azure_ai(no_token_usage_result)
    assert token_usage == TokenUsage(
        input_tokens=0, output_tokens=0, model="mistral-small-2503"
    )

    # Test with partial token_usage field
    partial_token_usage_result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="Hello, world!"))]],
        llm_output={
            "token_usage": {"input_tokens": 15},  # missing output_tokens
            "model_name": "mistral-small-2503",
        },
    )
    token_usage = get_token_usage_for_azure_ai(partial_token_usage_result)
    assert token_usage == TokenUsage(
        input_tokens=15, output_tokens=0, model="mistral-small-2503"
    )


def test_cost_callback_handler():
    cost_cb = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)
    cost_cb.on_llm_end(openai_llm_result)

    # cost
    assert cost_cb.total_tokens() == TokenUsage(
        input_tokens=10, output_tokens=10, model="gpt-4o"
    )

    assert cost_cb.total_cost(0.1) == 2.0
    assert (
        cost_cb.total_cost(cost_per_input_token=0.1, cost_per_output_token=0.1) == 2.0
    )


class _FakePrice:
    def __init__(self, input_per_token: float, output_per_token: float):
        self.input_per_token = input_per_token
        self.output_per_token = output_per_token


def _fake_llm_prices(table):
    import types

    module = types.ModuleType("llm_prices")
    module.lookup = lambda model: table.get(model)  # type: ignore[attr-defined]
    return module


def test_total_cost_looks_up_prices_when_none_given(monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "llm_prices",
        _fake_llm_prices(
            {
                "gpt-4o": _FakePrice(0.1, 0.2),
                "claude-3-opus-20240229": _FakePrice(1.0, 3.0),
            }
        ),
    )
    cost_cb = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)
    cost_cb.on_llm_end(openai_llm_result)
    assert cost_cb.total_cost() == pytest.approx(10 * 0.1 + 10 * 0.2)

    # one price per model when several models were used
    cost_cb.usage_data.append(
        TokenUsage(input_tokens=2, output_tokens=1, model="claude-3-opus-20240229")
    )
    assert cost_cb.total_cost() == pytest.approx(3.0 + 2 * 1.0 + 1 * 3.0)

    # explicit costs still win over the lookup
    assert cost_cb.total_cost(per_model_costs={"gpt-4o": (1.0, 1.0)}) == 20.0


def test_total_cost_unknown_model_raises(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "llm_prices", _fake_llm_prices({}))
    cost_cb = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)
    cost_cb.on_llm_end(openai_llm_result)
    with pytest.raises(ValueError, match="gpt-4o"):
        cost_cb.total_cost()


def test_total_cost_without_llm_prices_raises(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "llm_prices", None)
    cost_cb = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)
    cost_cb.on_llm_end(openai_llm_result)
    with pytest.raises(ValueError, match="No cost table or cost per token provided"):
        cost_cb.total_cost()


def test_total_cost_with_llm_prices_package():
    llm_prices = pytest.importorskip("llm_prices")

    price = llm_prices.lookup("gpt-4o")
    cost_cb = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)
    cost_cb.on_llm_end(openai_llm_result)
    assert cost_cb.total_cost() == pytest.approx(
        10 * price.input_per_token + 10 * price.output_per_token
    )
