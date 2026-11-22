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
                    usage_metadata={
                        "input_tokens": 10,
                        "output_tokens": 10,
                        "total_tokens": 20,
                    },
                    response_metadata={
                        "model_provider": "bedrock",
                        "model_name": "us.meta.llama3-1-70b-instruct-v1:0",
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
                    usage_metadata={
                        "input_tokens": 10,
                        "output_tokens": 10,
                        "total_tokens": 20,
                    },
                    response_metadata={
                        "model_provider": "bedrock",
                        "model_name": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                    },
                ),
            )
        ]
    ],
    llm_output={},
)

# Bedrock legacy response format
bedrock_legacy_result = LLMResult(
    generations=[
        [
            ChatGeneration(
                message=AIMessage(
                    content="Hello",
                    response_metadata={
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 9,
                        },
                        "model_id": "legacy-bedrock-model",
                    },
                )
            )
        ]
    ],
    llm_output={},
)

# Bedrock additional_kwargs response format
bedrock_additional_kwargs_result = LLMResult(
    generations=[
        [
            ChatGeneration(
                message=AIMessage(
                    content="Hello, world!",
                    additional_kwargs={
                        "usage": {
                            "prompt_tokens": 15,
                            "completion_tokens": 7,
                        }
                    },
                    response_metadata={
                        "model_name": "test-bedrock-model",
                    },
                )
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

    # Bedrock additional_kwargs usage format
    token_usage = get_token_usage_for_bedrock(bedrock_additional_kwargs_result)

    assert token_usage == TokenUsage(
        input_tokens=15,
        output_tokens=7,
        model="test-bedrock-model",
    )

    # Bedrock legacy response format
    token_usage = get_token_usage_for_bedrock(bedrock_legacy_result)

    assert token_usage == TokenUsage(
        input_tokens=12,
        output_tokens=9,
        model="legacy-bedrock-model",
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
