import json
from typing import List, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import ragas.messages as r


def convert_to_ragas_messages(
    messages: List[Union[HumanMessage, SystemMessage, AIMessage, ToolMessage]],
    metadata: bool = False,
) -> List[Union[r.HumanMessage, r.AIMessage, r.ToolMessage]]:
    """
    Convert LangChain messages into Ragas messages with metadata for agent evaluation.

    Parameters
    ----------
    messages : List[Union[HumanMessage, SystemMessage, AIMessage, ToolMessage]]
        List of LangChain message objects to be converted.
    metadata : bool, optional (default=False)
        Whether to include metadata in the converted messages.

    Returns
    -------
    List[Union[r.HumanMessage, r.AIMessage, r.ToolMessage]]
        List of corresponding Ragas message objects with metadata.

    Raises
    ------
    ValueError
        If an unsupported message type is encountered.
    TypeError
        If message content is not a string.

    Notes
    -----
    SystemMessages are skipped in the conversion process.
    """

    def _validate_string_content(message, message_type: str) -> str:
        if not isinstance(message.content, str):
            raise TypeError(
                f"{message_type} content must be a string, got {type(message.content).__name__}. "
                f"Content: {message.content}"
            )
        return message.content

    def _extract_metadata(message) -> dict:
        return {k: v for k, v in message.__dict__.items() if k != "content"}

    if metadata:
        MESSAGE_TYPE_MAP = {
            HumanMessage: lambda m: r.HumanMessage(
                content=_validate_string_content(m, "HumanMessage"),
                metadata=_extract_metadata(m),
            ),
            ToolMessage: lambda m: r.ToolMessage(
                content=_validate_string_content(m, "ToolMessage"),
                metadata=_extract_metadata(m),
            ),
        }
    else:
        MESSAGE_TYPE_MAP = {
            HumanMessage: lambda m: r.HumanMessage(
                content=_validate_string_content(m, "HumanMessage")
            ),
            ToolMessage: lambda m: r.ToolMessage(
                content=_validate_string_content(m, "ToolMessage")
            ),
        }

    def _extract_tool_calls(message: AIMessage) -> List[r.ToolCall]:
        if message.tool_calls:
            tool_calls = message.tool_calls
            return [
                r.ToolCall(
                    name=tool_call["name"],
                    args=tool_call["args"],
                )
                for tool_call in tool_calls
            ]

        tool_calls = message.additional_kwargs.get("tool_calls", [])

        return [
            r.ToolCall(
                name=tool_call["function"]["name"],
                args=json.loads(tool_call["function"]["arguments"]),
            )
            for tool_call in tool_calls
        ]

    def _convert_ai_message(message: AIMessage, metadata: bool) -> r.AIMessage:
        tool_calls = _extract_tool_calls(message) if message.additional_kwargs or message.tool_calls else None
        if metadata:
            return r.AIMessage(
                content=_validate_string_content(message, "AIMessage"),
                tool_calls=tool_calls,
                metadata=_extract_metadata(message),
            )
        else:
            return r.AIMessage(
                content=_validate_string_content(message, "AIMessage"),
                tool_calls=tool_calls,
            )

    def _convert_message(message, metadata: bool = False):
        if isinstance(message, SystemMessage):
            return None  # Skip SystemMessages
        if isinstance(message, AIMessage):
            return _convert_ai_message(message, metadata)
        converter = MESSAGE_TYPE_MAP.get(type(message))
        if converter is None:
            raise ValueError(f"Unsupported message type: {type(message).__name__}")
        return converter(message)

    return [
        converted
        for message in messages
        if (converted := _convert_message(message)) is not None
    ]
