"""Integration for AG2 (formerly AutoGen) multi-agent framework.

AG2 is an open-source framework for building multi-agent AI applications
with 500K+ monthly PyPI downloads. This module converts AG2 conversation
messages to Ragas message format for evaluation.

See: https://ag2.ai/
"""

import json
import typing as t

from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage


def convert_to_ragas_messages(
    messages: t.List[t.Dict[str, t.Any]],
    metadata: bool = False,
) -> t.List[t.Union[HumanMessage, AIMessage, ToolMessage]]:
    """Convert AG2 conversation messages to Ragas message format.

    AG2 agents store conversation history as lists of message dicts
    with OpenAI-compatible format. This function converts them into
    typed Ragas message objects for evaluation.

    Parameters
    ----------
    messages : List[Dict[str, Any]]
        List of AG2 message dictionaries from agent chat history.
        Each dict must have a "role" key with value "user", "assistant",
        or "tool". Messages are typically obtained from
        ``agent.chat_messages`` or the result of ``run().process()``.
    metadata : bool, optional
        If True, include extra message attributes (like agent name)
        in the metadata field of AIMessage. Default is False.

    Returns
    -------
    List[Union[HumanMessage, AIMessage, ToolMessage]]
        List of converted Ragas format messages.

    Raises
    ------
    KeyError
        If a message is missing the required 'role' key.
    ValueError
        If a message has an unsupported role.

    Examples
    --------
    >>> from ragas.integrations.ag2 import convert_to_ragas_messages
    >>> messages = [
    ...     {"role": "user", "content": "Analyze AAPL stock"},
    ...     {"role": "assistant", "content": "AAPL is trading at $185.",
    ...      "name": "analyst"},
    ... ]
    >>> ragas_msgs = convert_to_ragas_messages(messages)
    >>> len(ragas_msgs)
    2
    """

    def _parse_tool_calls(
        tool_calls_data: t.List[t.Dict[str, t.Any]],
    ) -> t.List[ToolCall]:
        """Parse AG2 tool calls into Ragas ToolCall objects."""
        result = []
        for tc in tool_calls_data:
            func = tc.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {"raw": args_str}
            result.append(ToolCall(name=name, args=args))
        return result

    def _extract_metadata(
        message: t.Dict[str, t.Any],
    ) -> t.Optional[t.Dict[str, t.Any]]:
        """Extract metadata from an AG2 message."""
        if not metadata:
            return None
        meta = {}
        if "name" in message:
            meta["agent_name"] = message["name"]
        if "tool_call_id" in message:
            meta["tool_call_id"] = message["tool_call_id"]
        return meta if meta else None

    def _handle_assistant(msg: t.Dict[str, t.Any]) -> AIMessage:
        """Convert assistant message to Ragas AIMessage."""
        tool_calls = (
            _parse_tool_calls(msg["tool_calls"]) if msg.get("tool_calls") else []
        )
        content = msg.get("content") or ""
        meta = _extract_metadata(msg)
        return AIMessage(content=content, tool_calls=tool_calls, metadata=meta)

    def _handle_tool(msg: t.Dict[str, t.Any]) -> ToolMessage:
        """Convert tool message to Ragas ToolMessage."""
        content = msg.get("content") or ""
        return ToolMessage(content=content)

    def _handle_user(msg: t.Dict[str, t.Any]) -> HumanMessage:
        """Convert user message to Ragas HumanMessage."""
        content = msg.get("content") or ""
        return HumanMessage(content=content)

    converted: t.List[t.Union[HumanMessage, AIMessage, ToolMessage]] = []

    for message in messages:
        role = message.get("role")
        if role is None:
            raise KeyError("'role' key not present in message")

        if role == "assistant":
            converted.append(_handle_assistant(message))
        elif role == "tool":
            converted.append(_handle_tool(message))
        elif role == "user":
            converted.append(_handle_user(message))
        elif role == "system":
            # AG2 system messages are skipped (not relevant for evaluation)
            continue
        else:
            raise ValueError(
                f"Role must be one of ['assistant', 'user', 'tool', 'system'], "
                f"but found '{role}'"
            )

    return converted
