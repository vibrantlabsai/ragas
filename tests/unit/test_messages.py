import warnings

from ragas.messages import AIMessage, ToolCall


def test_ai_message_to_dict_serialises_tool_calls_without_deprecation():
    """to_dict() must not rely on the removed-in-V3 Pydantic ``dict`` API."""
    message = AIMessage(
        content="let me check the weather",
        tool_calls=[ToolCall(name="get_weather", args={"city": "Paris"})],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        payload = message.to_dict()

    deprecations = [
        str(record.message)
        for record in caught
        if issubclass(record.category, DeprecationWarning)
    ]

    assert deprecations == []
    assert payload == {
        "content": {
            "text": "let me check the weather",
            "tool_calls": [{"name": "get_weather", "args": {"city": "Paris"}}],
        },
        "type": "ai",
    }
