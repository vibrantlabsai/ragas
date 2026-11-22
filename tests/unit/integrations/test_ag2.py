"""Tests for AG2 integration message conversion."""

import pytest

from ragas.integrations.ag2 import convert_to_ragas_messages
from ragas.messages import AIMessage, HumanMessage, ToolMessage


class TestConvertToRagasMessages:
    """Test AG2 message conversion to Ragas format."""

    def test_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = convert_to_ragas_messages(messages)
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello"

    def test_assistant_message(self):
        messages = [
            {"role": "assistant", "content": "I can help with that.", "name": "helper"}
        ]
        result = convert_to_ragas_messages(messages)
        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "I can help with that."

    def test_assistant_with_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "name": "researcher",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_stock_info",
                            "arguments": '{"ticker": "AAPL"}',
                        },
                    }
                ],
            }
        ]
        result = convert_to_ragas_messages(messages)
        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0].name == "get_stock_info"
        assert result[0].tool_calls[0].args == {"ticker": "AAPL"}

    def test_tool_message(self):
        messages = [
            {
                "role": "tool",
                "content": '{"price": 185.5}',
                "tool_call_id": "call_abc123",
                "name": "get_stock_info",
            }
        ]
        result = convert_to_ragas_messages(messages)
        assert len(result) == 1
        assert isinstance(result[0], ToolMessage)
        assert result[0].content == '{"price": 185.5}'

    def test_system_message_skipped(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ]
        result = convert_to_ragas_messages(messages)
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)

    def test_full_conversation(self):
        messages = [
            {"role": "user", "content": "What is AAPL price?"},
            {
                "role": "assistant",
                "content": "",
                "name": "analyst",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_stock_info",
                            "arguments": '{"ticker": "AAPL"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"ticker": "AAPL", "price": 185.5}',
                "tool_call_id": "call_1",
                "name": "get_stock_info",
            },
            {
                "role": "assistant",
                "content": "AAPL is currently trading at $185.50.",
                "name": "analyst",
            },
        ]
        result = convert_to_ragas_messages(messages)
        assert len(result) == 4
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        assert isinstance(result[2], ToolMessage)
        assert isinstance(result[3], AIMessage)

    def test_metadata_enabled(self):
        messages = [{"role": "assistant", "content": "Hello", "name": "greeter"}]
        result = convert_to_ragas_messages(messages, metadata=True)
        assert result[0].metadata == {"agent_name": "greeter"}

    def test_metadata_disabled_by_default(self):
        messages = [{"role": "assistant", "content": "Hello", "name": "greeter"}]
        result = convert_to_ragas_messages(messages, metadata=False)
        assert result[0].metadata is None

    def test_missing_role_raises_key_error(self):
        messages = [{"content": "No role here"}]
        with pytest.raises(KeyError, match="role"):
            convert_to_ragas_messages(messages)

    def test_invalid_role_raises_value_error(self):
        messages = [{"role": "unknown", "content": "Bad role"}]
        with pytest.raises(ValueError, match="unknown"):
            convert_to_ragas_messages(messages)

    def test_empty_messages(self):
        result = convert_to_ragas_messages([])
        assert result == []

    def test_none_content_handled(self):
        messages = [{"role": "assistant", "content": None, "name": "agent"}]
        result = convert_to_ragas_messages(messages)
        assert result[0].content == ""

    def test_multiple_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "AI"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": '{"expr": "2+2"}',
                        },
                    },
                ],
            }
        ]
        result = convert_to_ragas_messages(messages)
        assert len(result[0].tool_calls) == 2
        assert result[0].tool_calls[0].name == "search"
        assert result[0].tool_calls[1].name == "calculate"

    def test_malformed_json_arguments(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "broken",
                            "arguments": "not valid json{",
                        },
                    }
                ],
            }
        ]
        result = convert_to_ragas_messages(messages)
        assert result[0].tool_calls[0].args == {"raw": "not valid json{"}
