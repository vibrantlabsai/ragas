import json
from typing import List, Union, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import ragas.messages as r
from ragas.integrations.langgraph import convert_to_ragas_messages


def test_human_message_conversion():
    """Test conversion of HumanMessage with valid string content"""
    messages = [
        HumanMessage(content="Hello, add 4 and 5"),
        ToolMessage(content="9", tool_call_id="1"),
    ]
    result = convert_to_ragas_messages(messages)

    assert len(result) == 2
    assert isinstance(result[0], r.HumanMessage)
    assert result[0].content == "Hello, add 4 and 5"


def test_human_message_invalid_content():
    """Test HumanMessage with invalid content type raises TypeError"""
    messages: List[Union[HumanMessage, SystemMessage, AIMessage, ToolMessage]] = [
        HumanMessage(content=["invalid", "content"])
    ]

    with pytest.raises(TypeError) as exc_info:
        convert_to_ragas_messages(messages)
    assert "HumanMessage content must be a string" in str(exc_info.value)


def test_ai_message_conversion():
    """Test conversion of AIMessage with valid string content"""
    messages: List[Union[HumanMessage, SystemMessage, AIMessage, ToolMessage]] = [
        AIMessage(content="I'm doing well, thanks!")
    ]
    result = convert_to_ragas_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], r.AIMessage)
    assert result[0].content == "I'm doing well, thanks!"
    assert result[0].tool_calls is None


def test_ai_message_with_tool_calls():
    """Test conversion of AIMessage with tool calls"""

    tool_calls = [
        {
            "function": {
                "arguments": '{"metal_name": "gold"}',
                "name": "get_metal_price",
            }
        },
        {
            "function": {
                "arguments": '{"metal_name": "silver"}',
                "name": "get_metal_price",
            }
        },
    ]

    messages: List[Union[HumanMessage, SystemMessage, AIMessage, ToolMessage]] = [
        AIMessage(
            content="Find the difference in the price of gold and silver?",
            additional_kwargs={"tool_calls": tool_calls},
        )
    ]

    result = convert_to_ragas_messages(messages)
    assert len(result) == 1
    assert isinstance(result[0], r.AIMessage)
    assert result[0].content == "Find the difference in the price of gold and silver?"
    assert result[0].tool_calls is not None
    assert len(result[0].tool_calls) == 2
    assert result[0].tool_calls[0].name == "get_metal_price"
    assert result[0].tool_calls[0].args == {"metal_name": "gold"}
    assert result[0].tool_calls[1].name == "get_metal_price"
    assert result[0].tool_calls[1].args == {"metal_name": "silver"}


def test_tool_message_conversion():
    """Test conversion of ToolMessage with valid string content"""
    messages = [
        HumanMessage(content="Hello, add 4 and 5"),
        ToolMessage(content="9", tool_call_id="2"),
    ]
    result = convert_to_ragas_messages(messages)

    assert len(result) == 2
    assert isinstance(result[1], r.ToolMessage)
    assert result[1].content == "9"


def test_system_message_skipped():
    """Test that SystemMessages are properly skipped"""
    messages = [SystemMessage(content="System prompt"), HumanMessage(content="Hello")]
    result = convert_to_ragas_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], r.HumanMessage)
    assert result[0].content == "Hello"


def test_unsupported_message_type():
    """Test that unsupported message types raise ValueError"""

    class CustomMessage:
        content = "test"

    messages = cast(
        List[Union[HumanMessage, SystemMessage, AIMessage, ToolMessage]],
        [CustomMessage()],
    )

    with pytest.raises(ValueError) as exc_info:
        convert_to_ragas_messages(messages)
    assert "Unsupported message type: CustomMessage" in str(exc_info.value)


def test_empty_message_list():
    """Test conversion of empty message list"""
    messages = []
    result = convert_to_ragas_messages(messages)
    assert result == []


def test_invalid_tool_calls_json():
    """Test handling of invalid JSON in tool calls"""
    tool_calls = [{"function": {"name": "search", "arguments": "invalid json"}}]

    messages: List[Union[HumanMessage, SystemMessage, AIMessage, ToolMessage]] = [
        AIMessage(content="Test", additional_kwargs={"tool_calls": tool_calls})
    ]

    with pytest.raises(json.JSONDecodeError):
        convert_to_ragas_messages(messages)


def test_messages_metadata_conversion():
    """Test metadata conversion for HumanMessage, AIMessage, and ToolMessage when metadata=True"""
    messages = [
        HumanMessage(content="Hello", id="msg_1"),
        AIMessage(content="Hi", id="msg_2"),
        ToolMessage(content="Result", tool_call_id="1", id="msg_3"),
    ]
    result = convert_to_ragas_messages(messages, metadata=True)

    assert len(result) == 3
    assert result[0].metadata is not None and result[0].metadata.get("id") == "msg_1"
    assert result[1].metadata is not None and result[1].metadata.get("id") == "msg_2"
    assert result[2].metadata is not None and result[2].metadata.get("id") == "msg_3"
