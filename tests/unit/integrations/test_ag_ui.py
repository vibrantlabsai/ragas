"""Tests for AG-UI integration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ragas.messages import AIMessage, HumanMessage, ToolMessage

# Check if ag_ui is available
try:
    from ag_ui.core import (
        AssistantMessage,
        EventType,
        MessagesSnapshotEvent,
        RunFinishedEvent,
        RunStartedEvent,
        StepFinishedEvent,
        StepStartedEvent,
        TextMessageChunkEvent,
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
        ToolCallArgsEvent,
        ToolCallChunkEvent,
        ToolCallEndEvent,
        ToolCallResultEvent,
        ToolCallStartEvent,
        UserMessage,
    )

    AG_UI_AVAILABLE = True
except ImportError:
    AG_UI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not AG_UI_AVAILABLE, reason="ag-ui-protocol not installed"
)


# Mock event class for non-message events
class MockEvent:
    """Simple mock for non-message events like STATE_SNAPSHOT."""

    def __init__(self, event_type: str, **kwargs):
        self.type = event_type
        self.timestamp = kwargs.get("timestamp", 1234567890)
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def basic_text_message_events():
    """Create a basic streaming text message event sequence."""
    return [
        RunStartedEvent(run_id="run-123", thread_id="thread-456"),
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Hello"),
        TextMessageContentEvent(message_id="msg-1", delta=" world"),
        TextMessageEndEvent(message_id="msg-1"),
        TextMessageStartEvent(message_id="msg-2", role="assistant"),
        TextMessageContentEvent(message_id="msg-2", delta="Hi"),
        TextMessageContentEvent(message_id="msg-2", delta=" there!"),
        TextMessageEndEvent(message_id="msg-2"),
    ]


@pytest.fixture
def tool_call_events():
    """Create events with tool calls."""
    return [
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Let me check the weather"),
        TextMessageEndEvent(message_id="msg-1"),
        ToolCallStartEvent(
            tool_call_id="tc-1", tool_call_name="get_weather", parent_message_id="msg-1"
        ),
        ToolCallArgsEvent(tool_call_id="tc-1", delta='{"city": "San Francisco"'),
        ToolCallArgsEvent(tool_call_id="tc-1", delta=', "units": "fahrenheit"}'),
        ToolCallEndEvent(tool_call_id="tc-1"),
        ToolCallResultEvent(
            tool_call_id="tc-1",
            message_id="result-1",
            content="Temperature: 72°F, Conditions: Sunny",
        ),
        TextMessageStartEvent(message_id="msg-2", role="assistant"),
        TextMessageContentEvent(
            message_id="msg-2", delta="It's sunny and 72°F in San Francisco"
        ),
        TextMessageEndEvent(message_id="msg-2"),
    ]


def test_import_error_without_ag_ui_protocol():
    """Test that appropriate error is raised without ag-ui-protocol package."""
    from ragas.integrations.ag_ui import _import_ag_ui_core

    # Mock the actual ag_ui import
    with patch.dict("sys.modules", {"ag_ui": None, "ag_ui.core": None}):
        with pytest.raises(
            ImportError, match="AG-UI integration requires the ag-ui-protocol package"
        ):
            _import_ag_ui_core()


def test_basic_text_message_conversion(basic_text_message_events):
    """Test converting basic streaming text messages."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    messages = convert_to_ragas_messages(basic_text_message_events)

    assert len(messages) == 2
    assert isinstance(messages[0], AIMessage)
    assert messages[0].content == "Hello world"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "Hi there!"


def test_message_with_metadata(basic_text_message_events):
    """Test that metadata is included when requested."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    messages = convert_to_ragas_messages(basic_text_message_events, metadata=True)

    assert len(messages) == 2
    assert messages[0].metadata is not None
    assert "message_id" in messages[0].metadata
    assert messages[0].metadata["message_id"] == "msg-1"
    assert "run_id" in messages[0].metadata
    assert messages[0].metadata["run_id"] == "run-123"
    assert "thread_id" in messages[0].metadata
    assert messages[0].metadata["thread_id"] == "thread-456"


def test_message_without_metadata(basic_text_message_events):
    """Test that metadata is excluded when not requested."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    messages = convert_to_ragas_messages(basic_text_message_events, metadata=False)

    assert len(messages) == 2
    assert messages[0].metadata is None
    assert messages[1].metadata is None


def test_tool_call_conversion(tool_call_events):
    """Test converting tool calls with arguments and results."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    messages = convert_to_ragas_messages(tool_call_events)

    # Should have: AI message, Tool result, AI message
    assert len(messages) == 3

    # First message: AI initiating tool call
    assert isinstance(messages[0], AIMessage)
    assert messages[0].content == "Let me check the weather"

    # Second message: Tool result
    assert isinstance(messages[1], ToolMessage)
    assert "72°F" in messages[1].content

    # Third message: AI with response
    assert isinstance(messages[2], AIMessage)
    assert "sunny" in messages[2].content.lower()


def test_tool_call_with_metadata(tool_call_events):
    """Test that tool call metadata is preserved."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    messages = convert_to_ragas_messages(tool_call_events, metadata=True)

    tool_message = next(msg for msg in messages if isinstance(msg, ToolMessage))
    assert tool_message.metadata is not None
    assert "tool_call_id" in tool_message.metadata
    assert tool_message.metadata["tool_call_id"] == "tc-1"


def test_step_context_in_metadata():
    """Test that step context is included in metadata."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        RunStartedEvent(run_id="run-1", thread_id="thread-1"),
        StepStartedEvent(step_name="analyze_query"),
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Processing..."),
        TextMessageEndEvent(message_id="msg-1"),
        StepFinishedEvent(step_name="analyze_query"),
    ]

    messages = convert_to_ragas_messages(events, metadata=True)

    assert len(messages) == 1
    assert "step_name" in messages[0].metadata
    assert messages[0].metadata["step_name"] == "analyze_query"


def test_messages_snapshot_conversion():
    """Test converting MessagesSnapshotEvent."""
    from ragas.integrations.ag_ui import convert_messages_snapshot

    snapshot = MessagesSnapshotEvent(
        messages=[
            UserMessage(id="msg-1", content="What's 2+2?"),
            AssistantMessage(id="msg-2", content="4"),
            UserMessage(id="msg-3", content="Thanks!"),
        ]
    )

    messages = convert_messages_snapshot(snapshot)

    assert len(messages) == 3
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "What's 2+2?"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "4"
    assert isinstance(messages[2], HumanMessage)
    assert messages[2].content == "Thanks!"


def test_snapshot_with_metadata():
    """Test that snapshot conversion includes metadata when requested."""
    from ragas.integrations.ag_ui import convert_messages_snapshot

    snapshot = MessagesSnapshotEvent(
        messages=[UserMessage(id="msg-1", content="Hello")]
    )

    messages = convert_messages_snapshot(snapshot, metadata=True)

    assert messages[0].metadata is not None
    assert "message_id" in messages[0].metadata
    assert messages[0].metadata["message_id"] == "msg-1"


def test_non_message_events_filtered():
    """Test that non-message events are silently filtered."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        RunStartedEvent(run_id="run-1", thread_id="thread-1"),
        MockEvent(EventType.STATE_SNAPSHOT, snapshot={"key": "value"}),
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Hello"),
        TextMessageEndEvent(message_id="msg-1"),
        MockEvent("RUN_FINISHED", result="success"),
    ]

    messages = convert_to_ragas_messages(events)

    # Should only have the text message, other events filtered
    assert len(messages) == 1
    assert messages[0].content == "Hello"


def test_incomplete_message_stream(caplog):
    """Test handling of incomplete message streams."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    # Message with content but no end event
    events = [
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Hello"),
        # Missing TextMessageEndEvent
    ]

    messages = convert_to_ragas_messages(events)

    # Should not create message without end event
    assert len(messages) == 0


def test_orphaned_content_event(caplog):
    """Test handling of content event without corresponding start."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        # Content event without start
        TextMessageContentEvent(message_id="msg-unknown", delta="Orphaned content"),
    ]

    messages = convert_to_ragas_messages(events)

    assert len(messages) == 0


def test_tool_call_argument_parsing_error(caplog):
    """Test handling of invalid JSON in tool arguments."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Using tool"),
        ToolCallStartEvent(tool_call_id="tc-1", tool_call_name="broken_tool"),
        ToolCallArgsEvent(tool_call_id="tc-1", delta="{invalid json"),
        ToolCallEndEvent(tool_call_id="tc-1"),
        TextMessageEndEvent(message_id="msg-1"),  # Message ends AFTER tool call
    ]

    messages = convert_to_ragas_messages(events)

    # Should still create message with tool call containing raw_args
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert messages[0].tool_calls is not None
    assert len(messages[0].tool_calls) == 1
    assert messages[0].tool_calls[0].name == "broken_tool"
    # Invalid JSON should be stored in raw_args
    assert "raw_args" in messages[0].tool_calls[0].args
    assert messages[0].tool_calls[0].args["raw_args"] == "{invalid json"


def test_tool_call_result_retroactive_attachment():
    """
    Tests that ToolCallResultEvent correctly finds the previous AIMessage
    and attaches the tool call specification if it was missing.

    This can happen when ToolCallEndEvent arrives before TextMessageEndEvent,
    causing tool_calls to be cleared from _completed_tool_calls before the
    AIMessage is created.
    """
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    # Scenario: TextMessageEnd arrives AFTER ToolCallEnd, so the tool call
    # is already cleared from _completed_tool_calls when the AIMessage is created
    events = [
        # AI message starts
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Let me check that"),
        # Tool call happens
        ToolCallStartEvent(tool_call_id="tc-1", tool_call_name="search_tool"),
        ToolCallArgsEvent(tool_call_id="tc-1", delta='{"query": "weather"}'),
        ToolCallEndEvent(tool_call_id="tc-1"),
        # Message ends AFTER tool call ends
        TextMessageEndEvent(message_id="msg-1"),
        # Tool result arrives
        ToolCallResultEvent(
            tool_call_id="tc-1", message_id="result-1", content="Sunny, 75F"
        ),
    ]

    messages = convert_to_ragas_messages(events)

    # Should have AI message with tool call, then Tool message
    assert len(messages) == 2
    assert isinstance(messages[0], AIMessage)
    assert isinstance(messages[1], ToolMessage)

    # The AIMessage should have the tool_calls attached (either from normal flow
    # or retroactively attached by _handle_tool_call_result)
    assert messages[0].tool_calls is not None
    assert len(messages[0].tool_calls) >= 1
    # At least one tool call should be present (could be synthetic if needed)
    assert any(
        tc.name in ["search_tool", "unknown_tool"] for tc in messages[0].tool_calls
    )

    # Tool message should contain the result
    assert messages[1].content == "Sunny, 75F"


def test_event_collector_reuse(basic_text_message_events):
    """Test that AGUIEventCollector can be cleared and reused."""
    from ragas.integrations.ag_ui import AGUIEventCollector

    collector = AGUIEventCollector()

    # Process first batch
    for event in basic_text_message_events[:5]:  # First message
        collector.process_event(event)

    messages1 = collector.get_messages()
    assert len(messages1) == 1

    # Clear and process second batch
    collector.clear()
    for event in basic_text_message_events[5:]:  # Second message
        collector.process_event(event)

    messages2 = collector.get_messages()
    assert len(messages2) == 1
    assert messages2[0].content != messages1[0].content


def test_multiple_tool_calls_in_sequence():
    """Test handling multiple tool calls in sequence."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        ToolCallStartEvent(tool_call_id="tc-1", tool_call_name="tool1"),
        ToolCallArgsEvent(tool_call_id="tc-1", delta='{"param": "value1"}'),
        ToolCallEndEvent(tool_call_id="tc-1"),
        ToolCallStartEvent(tool_call_id="tc-2", tool_call_name="tool2"),
        ToolCallArgsEvent(tool_call_id="tc-2", delta='{"param": "value2"}'),
        ToolCallEndEvent(tool_call_id="tc-2"),
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Done"),
        TextMessageEndEvent(message_id="msg-1"),
    ]

    messages = convert_to_ragas_messages(events)

    # Should create AI message with both tool calls
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert messages[0].tool_calls is not None
    assert len(messages[0].tool_calls) == 2
    assert messages[0].tool_calls[0].name == "tool1"
    assert messages[0].tool_calls[1].name == "tool2"


def test_empty_event_list():
    """Test handling of empty event list."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    messages = convert_to_ragas_messages([])
    assert len(messages) == 0


def test_wrong_snapshot_type_error():
    """Test that convert_messages_snapshot validates input type."""
    from ragas.integrations.ag_ui import convert_messages_snapshot

    with pytest.raises(TypeError, match="Expected MessagesSnapshotEvent"):
        convert_messages_snapshot(MockEvent("WRONG_TYPE"))


def test_role_mapping():
    """Test that different roles map correctly to Ragas message types."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        TextMessageStartEvent(message_id="msg-1", role="user"),
        TextMessageContentEvent(message_id="msg-1", delta="User message"),
        TextMessageEndEvent(message_id="msg-1"),
        TextMessageStartEvent(message_id="msg-2", role="assistant"),
        TextMessageContentEvent(message_id="msg-2", delta="Assistant message"),
        TextMessageEndEvent(message_id="msg-2"),
    ]

    messages = convert_to_ragas_messages(events)

    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "User message"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "Assistant message"


def test_complex_conversation_flow():
    """Test a complex multi-turn conversation with tool calls."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        RunStartedEvent(run_id="run-1", thread_id="thread-1"),
        # User asks
        TextMessageStartEvent(message_id="msg-1", role="user"),
        TextMessageContentEvent(message_id="msg-1", delta="What's the weather?"),
        TextMessageEndEvent(message_id="msg-1"),
        # Assistant responds and calls tool
        TextMessageStartEvent(message_id="msg-2", role="assistant"),
        TextMessageContentEvent(message_id="msg-2", delta="Let me check"),
        TextMessageEndEvent(message_id="msg-2"),
        ToolCallStartEvent(tool_call_id="tc-1", tool_call_name="weather_api"),
        ToolCallArgsEvent(tool_call_id="tc-1", delta='{"location": "SF"}'),
        ToolCallEndEvent(tool_call_id="tc-1"),
        # Tool returns result
        ToolCallResultEvent(
            tool_call_id="tc-1", message_id="result-1", content="Sunny, 70F"
        ),
        # Assistant responds with answer
        TextMessageStartEvent(message_id="msg-3", role="assistant"),
        TextMessageContentEvent(message_id="msg-3", delta="It's sunny and 70F"),
        TextMessageEndEvent(message_id="msg-3"),
        # User thanks
        TextMessageStartEvent(message_id="msg-4", role="user"),
        TextMessageContentEvent(message_id="msg-4", delta="Thanks!"),
        TextMessageEndEvent(message_id="msg-4"),
    ]

    messages = convert_to_ragas_messages(events, metadata=True)

    # Should have: Human, AI (with tool_calls), Tool, AI, Human
    assert len(messages) == 5
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], ToolMessage)
    assert isinstance(messages[3], AIMessage)
    assert isinstance(messages[4], HumanMessage)

    # Check content
    assert "weather" in messages[0].content.lower()
    assert "check" in messages[1].content.lower()
    assert "sunny" in messages[2].content.lower()
    assert "sunny" in messages[3].content.lower()
    assert "thanks" in messages[4].content.lower()

    # Check metadata
    assert all(msg.metadata is not None for msg in messages)
    assert all("run_id" in msg.metadata for msg in messages)


def test_text_message_chunk():
    """Test TEXT_MESSAGE_CHUNK event handling."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        TextMessageChunkEvent(
            message_id="msg-1", role="assistant", delta="Complete message"
        ),
    ]

    messages = convert_to_ragas_messages(events)

    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert messages[0].content == "Complete message"


def test_tool_call_chunk():
    """Test TOOL_CALL_CHUNK event handling."""
    from ragas.integrations.ag_ui import convert_to_ragas_messages

    events = [
        ToolCallChunkEvent(
            tool_call_id="tc-1", tool_call_name="search", delta='{"query": "test"}'
        ),
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Done"),
        TextMessageEndEvent(message_id="msg-1"),
    ]

    messages = convert_to_ragas_messages(events)

    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert messages[0].tool_calls is not None
    assert len(messages[0].tool_calls) == 1
    assert messages[0].tool_calls[0].name == "search"
    assert messages[0].tool_calls[0].args == {"query": "test"}


def test_tool_call_chunk_with_dict_delta():
    """
    Test that _handle_tool_call_chunk can handle delta as dict.

    While the AG-UI protocol specifies delta as a string, the handler code
    defensively handles dict deltas. We test this by directly calling the
    handler with a mock event object.
    """
    from ragas.integrations.ag_ui import AGUIEventCollector

    collector = AGUIEventCollector()

    # Create a mock event with dict delta (bypassing Pydantic validation)
    class MockToolCallChunkEvent:
        type = "TOOL_CALL_CHUNK"
        tool_call_id = "tc-1"
        tool_call_name = "calculate"
        delta = {"operation": "add", "values": [1, 2, 3]}  # dict instead of string
        timestamp = "2025-01-01T00:00:00Z"

    # Process the mock event directly
    collector._handle_tool_call_chunk(MockToolCallChunkEvent())

    # Now add an AI message to pick up the tool call
    from ag_ui.core import (
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
    )

    collector.process_event(TextMessageStartEvent(message_id="msg-1", role="assistant"))
    collector.process_event(
        TextMessageContentEvent(message_id="msg-1", delta="Result is 6")
    )
    collector.process_event(TextMessageEndEvent(message_id="msg-1"))

    messages = collector.get_messages()

    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert messages[0].tool_calls is not None
    assert len(messages[0].tool_calls) == 1
    assert messages[0].tool_calls[0].name == "calculate"
    assert messages[0].tool_calls[0].args == {"operation": "add", "values": [1, 2, 3]}


# ===== FastAPI Integration Tests =====


# Helper to check if FastAPI dependencies are available
def _has_fastapi_deps():
    try:
        import httpx  # noqa: F401

        return AG_UI_AVAILABLE
    except ImportError:
        return False


@pytest.mark.skipif(
    not _has_fastapi_deps(), reason="httpx or ag-ui-protocol not installed"
)
@pytest.mark.asyncio
async def test_call_ag_ui_endpoint():
    """Test HTTP client helper for calling AG-UI endpoints."""
    from unittest.mock import AsyncMock, MagicMock

    from ragas.integrations.ag_ui import _call_ag_ui_endpoint

    # Mock SSE response data
    sse_lines = [
        'data: {"type": "RUN_STARTED", "run_id": "run-1", "thread_id": "thread-1", "timestamp": 1234567890}',
        "",
        'data: {"type": "TEXT_MESSAGE_START", "message_id": "msg-1", "role": "assistant", "timestamp": 1234567891}',
        "",
        'data: {"type": "TEXT_MESSAGE_CONTENT", "message_id": "msg-1", "delta": "Hello!", "timestamp": 1234567892}',
        "",
        'data: {"type": "TEXT_MESSAGE_END", "message_id": "msg-1", "timestamp": 1234567893}',
        "",
        'data: {"type": "RUN_FINISHED", "run_id": "run-1", "thread_id": "thread-1", "timestamp": 1234567894}',
        "",
    ]

    # Create async iterator for SSE lines
    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    # Mock httpx response
    mock_response = MagicMock()
    mock_response.aiter_lines = mock_aiter_lines
    mock_response.raise_for_status = MagicMock()

    # Mock httpx client
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock()
    mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        events = await _call_ag_ui_endpoint(
            endpoint_url="http://localhost:8000/agent",
            user_input="Hello",
        )

    # Should have collected 5 events
    assert len(events) == 5
    assert events[0].type == "RUN_STARTED"
    assert events[1].type == "TEXT_MESSAGE_START"
    assert events[2].type == "TEXT_MESSAGE_CONTENT"
    assert events[3].type == "TEXT_MESSAGE_END"
    assert events[4].type == "RUN_FINISHED"


@pytest.mark.skipif(
    not _has_fastapi_deps(), reason="httpx or ag-ui-protocol not installed"
)
@pytest.mark.asyncio
async def test_call_ag_ui_endpoint_with_config():
    """Test HTTP client with thread_id and agent_config."""
    from unittest.mock import AsyncMock, MagicMock

    from ragas.integrations.ag_ui import _call_ag_ui_endpoint

    sse_lines = [
        'data: {"type": "RUN_STARTED", "run_id": "run-1", "thread_id": "my-thread", "timestamp": 1234567890}',
        "",
        'data: {"type": "RUN_FINISHED", "run_id": "run-1", "thread_id": "my-thread", "timestamp": 1234567891}',
        "",
    ]

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    mock_response = MagicMock()
    mock_response.aiter_lines = mock_aiter_lines
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock()
    mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        events = await _call_ag_ui_endpoint(
            endpoint_url="http://localhost:8000/agent",
            user_input="Test query",
            thread_id="my-thread",
            agent_config={"temperature": 0.7},
        )

    assert len(events) == 2
    # Check that thread_id was passed through
    assert events[0].thread_id == "my-thread"


@pytest.mark.skipif(
    not _has_fastapi_deps(), reason="httpx or ag-ui-protocol not installed"
)
@pytest.mark.asyncio
async def test_call_ag_ui_endpoint_malformed_json():
    """Test HTTP client handles malformed JSON gracefully."""
    from unittest.mock import AsyncMock, MagicMock

    from ragas.integrations.ag_ui import _call_ag_ui_endpoint

    sse_lines = [
        'data: {"type": "RUN_STARTED", "run_id": "run-1", "thread_id": "thread-1", "timestamp": 1234567890}',
        "",
        "data: {invalid json}",  # Malformed
        "",
        'data: {"type": "RUN_FINISHED", "run_id": "run-1", "thread_id": "thread-1", "timestamp": 1234567891}',
        "",
    ]

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    mock_response = MagicMock()
    mock_response.aiter_lines = mock_aiter_lines
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock()
    mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        events = await _call_ag_ui_endpoint(
            endpoint_url="http://localhost:8000/agent",
            user_input="Test",
        )

    # Should skip malformed event but collect valid ones
    assert len(events) == 2
    assert events[0].type == "RUN_STARTED"
    assert events[1].type == "RUN_FINISHED"


@pytest.mark.skipif(
    not _has_fastapi_deps(), reason="httpx or ag-ui-protocol not installed"
)
@pytest.mark.asyncio
async def test_evaluate_ag_ui_agent():
    """Test batch evaluation of AG-UI agent endpoint."""
    from ragas.dataset_schema import EvaluationDataset, EvaluationResult, SingleTurnSample
    from ragas.integrations.ag_ui import (
        MISSING_CONTEXT_PLACEHOLDER,
        MISSING_RESPONSE_PLACEHOLDER,
        evaluate_ag_ui_agent,
    )

    # Create mock dataset
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input="What's the weather?",
                reference="Check weather API",
            ),
            SingleTurnSample(
                user_input="Tell me a joke",
                reference="Respond with humor",
            ),
        ]
    )

    # Mock events for first query (weather)
    weather_events = [
        RunStartedEvent(run_id="run-1", thread_id="thread-1"),
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="It's sunny and 72F"),
        TextMessageEndEvent(message_id="msg-1"),
        RunFinishedEvent(run_id="run-1", thread_id="thread-1"),
    ]

    # Mock events for second query (joke)
    joke_events = [
        RunStartedEvent(run_id="run-2", thread_id="thread-2"),
        TextMessageStartEvent(message_id="msg-2", role="assistant"),
        TextMessageContentEvent(
            message_id="msg-2", delta="Why don't scientists trust atoms?"
        ),
        TextMessageContentEvent(message_id="msg-2", delta=" They make up everything!"),
        TextMessageEndEvent(message_id="msg-2"),
        RunFinishedEvent(run_id="run-2", thread_id="thread-2"),
    ]

    # Mock _call_ag_ui_endpoint to return different events based on input
    async def mock_call_endpoint(endpoint_url, user_input, **kwargs):
        if "weather" in user_input.lower():
            return weather_events
        else:
            return joke_events

    with patch(
        "ragas.integrations.ag_ui._call_ag_ui_endpoint",
        side_effect=mock_call_endpoint,
    ):
        result = await evaluate_ag_ui_agent(
            endpoint_url="http://localhost:8000/agent",
            dataset=dataset,
            metrics=[],  # Empty for testing
        )

    # Check that dataset was populated
    assert dataset.samples[0].response == "It's sunny and 72F"
    assert (
        dataset.samples[1].response
        == "Why don't scientists trust atoms? They make up everything!"
    )

    # Check that result is an EvaluationResult
    assert isinstance(result, EvaluationResult)
    # With empty metrics, scores should be empty dicts
    assert result.scores == [{}, {}]


@pytest.mark.skipif(
    not _has_fastapi_deps(), reason="httpx or ag-ui-protocol not installed"
)
@pytest.mark.asyncio
async def test_evaluate_ag_ui_agent_with_tool_calls():
    """Test evaluation with tool calls in response."""
    from ragas.dataset_schema import EvaluationDataset, EvaluationResult, SingleTurnSample
    from ragas.integrations.ag_ui import (
        MISSING_CONTEXT_PLACEHOLDER,
        MISSING_RESPONSE_PLACEHOLDER,
        evaluate_ag_ui_agent,
    )

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input="Search for Python tutorials",
            ),
        ]
    )

    # Mock events with tool call
    search_events = [
        RunStartedEvent(run_id="run-1", thread_id="thread-1"),
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Let me search for that"),
        TextMessageEndEvent(message_id="msg-1"),
        ToolCallStartEvent(tool_call_id="tc-1", tool_call_name="search"),
        ToolCallArgsEvent(tool_call_id="tc-1", delta='{"query": "Python tutorials"}'),
        ToolCallEndEvent(tool_call_id="tc-1"),
        ToolCallResultEvent(
            tool_call_id="tc-1",
            message_id="result-1",
            content="Found: tutorial1.com, tutorial2.com",
        ),
        RunFinishedEvent(run_id="run-1", thread_id="thread-1"),
    ]

    async def mock_call_endpoint(endpoint_url, user_input, **kwargs):
        return search_events

    with patch(
        "ragas.integrations.ag_ui._call_ag_ui_endpoint",
        side_effect=mock_call_endpoint,
    ):
        result = await evaluate_ag_ui_agent(
            endpoint_url="http://localhost:8000/agent",
            dataset=dataset,
            metrics=[],
        )

    # Check that response was extracted
    assert dataset.samples[0].response == "Let me search for that"
    # Check that tool results are in retrieved_contexts
    assert dataset.samples[0].retrieved_contexts is not None
    assert len(dataset.samples[0].retrieved_contexts) == 1
    assert "tutorial1.com" in dataset.samples[0].retrieved_contexts[0]
    # Check result type
    assert isinstance(result, EvaluationResult)


@pytest.mark.skipif(
    not _has_fastapi_deps(), reason="httpx or ag-ui-protocol not installed"
)
@pytest.mark.asyncio
async def test_evaluate_ag_ui_agent_handles_failures():
    """Test evaluation handles HTTP failures gracefully."""
    import math

    from ragas.dataset_schema import EvaluationDataset, EvaluationResult, SingleTurnSample
    from ragas.integrations.ag_ui import (
        MISSING_CONTEXT_PLACEHOLDER,
        MISSING_RESPONSE_PLACEHOLDER,
        evaluate_ag_ui_agent,
    )

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(user_input="Query 1"),
            SingleTurnSample(user_input="Query 2"),
        ]
    )

    # Mock events - first succeeds, second fails (returns NaN from executor)
    success_events = [
        RunStartedEvent(run_id="run-1", thread_id="thread-1"),
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Success response"),
        TextMessageEndEvent(message_id="msg-1"),
        RunFinishedEvent(run_id="run-1", thread_id="thread-1"),
    ]

    # Mock Executor to handle the exception
    class MockExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, func, *args, **kwargs):
            pass

        def results(self):
            # First result succeeds, second is NaN (failed)
            return [success_events, math.nan]

        async def aresults(self):
            return self.results()

    with patch(
        "ragas.integrations.ag_ui.Executor",
        MockExecutor,
    ):
        result = await evaluate_ag_ui_agent(
            endpoint_url="http://localhost:8000/agent",
            dataset=dataset,
            metrics=[],
        )

    # First sample should have response, second should use placeholders
    assert dataset.samples[0].response == "Success response"
    assert dataset.samples[1].response == MISSING_RESPONSE_PLACEHOLDER
    assert dataset.samples[1].retrieved_contexts == [MISSING_CONTEXT_PLACEHOLDER]
    # Check result type
    assert isinstance(result, EvaluationResult)


# ============================================================================
# Multi-turn evaluation tests
# ============================================================================


def test_convert_ragas_messages_to_ag_ui():
    """Test converting Ragas messages to AG-UI format."""
    from ragas.integrations.ag_ui import _convert_ragas_messages_to_ag_ui
    from ragas.messages import ToolCall

    messages = [
        HumanMessage(content="What's the weather?"),
        AIMessage(
            content="Let me check",
            tool_calls=[ToolCall(name="get-weather", args={"location": "SF"})],
        ),
        HumanMessage(content="Thanks!"),
    ]

    ag_ui_messages = _convert_ragas_messages_to_ag_ui(messages)

    assert len(ag_ui_messages) == 3

    # Check UserMessage
    assert ag_ui_messages[0].id == "1"
    assert ag_ui_messages[0].content == "What's the weather?"

    # Check AssistantMessage with tool calls
    assert ag_ui_messages[1].id == "2"
    assert ag_ui_messages[1].content == "Let me check"
    assert ag_ui_messages[1].tool_calls is not None
    assert len(ag_ui_messages[1].tool_calls) == 1
    assert ag_ui_messages[1].tool_calls[0].function.name == "get-weather"
    assert '"location": "SF"' in ag_ui_messages[1].tool_calls[0].function.arguments

    # Check second UserMessage
    assert ag_ui_messages[2].id == "3"
    assert ag_ui_messages[2].content == "Thanks!"


@pytest.mark.asyncio
async def test_evaluate_multi_turn_basic():
    """Test basic multi-turn evaluation."""
    from unittest.mock import patch

    from ragas.dataset_schema import EvaluationDataset, EvaluationResult, MultiTurnSample
    from ragas.integrations.ag_ui import evaluate_ag_ui_agent
    from ragas.messages import ToolCall

    # Create multi-turn sample
    sample = MultiTurnSample(
        user_input=[HumanMessage(content="What's the weather in SF?")],
        reference_tool_calls=[ToolCall(name="get-weather", args={"location": "SF"})],
    )

    dataset = EvaluationDataset(samples=[sample])

    # Mock events that agent would return
    # Note: Tool calls are completed before message, so they attach to the next AIMessage
    agent_events = [
        RunStartedEvent(run_id="run-1", thread_id="thread-1"),
        ToolCallStartEvent(
            tool_call_id="tc-1",
            tool_call_name="get-weather",
            parent_message_id="msg-1",
        ),
        ToolCallArgsEvent(tool_call_id="tc-1", delta='{"location": "SF"}'),
        ToolCallEndEvent(tool_call_id="tc-1"),
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Let me check the weather"),
        TextMessageEndEvent(message_id="msg-1"),
        ToolCallResultEvent(
            tool_call_id="tc-1",
            message_id="result-1",
            content="Temperature: 72°F",
        ),
        RunFinishedEvent(run_id="run-1", thread_id="thread-1"),
    ]

    # Mock Executor
    class MockExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, func, *args, **kwargs):
            pass

        def results(self):
            return [agent_events]

        async def aresults(self):
            return self.results()

    with patch(
        "ragas.integrations.ag_ui.Executor",
        MockExecutor,
    ):
        result = await evaluate_ag_ui_agent(
            endpoint_url="http://localhost:8000/agent",
            dataset=dataset,
            metrics=[],
        )

    # Verify that agent responses were appended to conversation
    assert len(sample.user_input) > 1  # Should have original + agent responses

    # Check that we have AIMessage and ToolMessage appended
    ai_messages = [msg for msg in sample.user_input if isinstance(msg, AIMessage)]
    tool_messages = [msg for msg in sample.user_input if isinstance(msg, ToolMessage)]

    assert len(ai_messages) >= 1  # At least one AI message
    assert len(tool_messages) >= 1  # At least one tool message

    # Verify tool calls in AIMessage (tool calls completed before message, so attached to it)
    assert ai_messages[0].tool_calls is not None
    assert len(ai_messages[0].tool_calls) > 0
    assert ai_messages[0].tool_calls[0].name == "get-weather"

    # Check result type
    assert isinstance(result, EvaluationResult)


@pytest.mark.asyncio
async def test_evaluate_multi_turn_with_existing_conversation():
    """Test multi-turn evaluation with pre-existing conversation."""
    from unittest.mock import patch

    from ragas.dataset_schema import EvaluationDataset, EvaluationResult, MultiTurnSample
    from ragas.integrations.ag_ui import evaluate_ag_ui_agent
    from ragas.messages import ToolCall

    # Create sample with existing conversation
    sample = MultiTurnSample(
        user_input=[
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="What's the weather in SF?"),
        ],
        reference_tool_calls=[ToolCall(name="get-weather", args={"location": "SF"})],
    )

    original_length = len(sample.user_input)
    dataset = EvaluationDataset(samples=[sample])

    # Mock agent events
    agent_events = [
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Let me check the weather"),
        TextMessageEndEvent(message_id="msg-1"),
        ToolCallStartEvent(
            tool_call_id="tc-1",
            tool_call_name="get-weather",
            parent_message_id="msg-1",
        ),
        ToolCallArgsEvent(tool_call_id="tc-1", delta='{"location": "SF"}'),
        ToolCallEndEvent(tool_call_id="tc-1"),
    ]

    class MockExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, func, *args, **kwargs):
            pass

        def results(self):
            return [agent_events]

    with patch(
        "ragas.integrations.ag_ui.Executor",
        MockExecutor,
    ):
        result = await evaluate_ag_ui_agent(
            endpoint_url="http://localhost:8000/agent",
            dataset=dataset,
            metrics=[],
        )

    # Verify conversation was extended, not replaced
    assert len(sample.user_input) > original_length

    # First 3 messages should be unchanged
    assert isinstance(sample.user_input[0], HumanMessage)
    assert sample.user_input[0].content == "Hello"
    assert isinstance(sample.user_input[1], AIMessage)
    assert sample.user_input[1].content == "Hi there!"
    assert isinstance(sample.user_input[2], HumanMessage)
    assert sample.user_input[2].content == "What's the weather in SF?"

    # New messages should be appended
    new_messages = sample.user_input[original_length:]
    assert len(new_messages) > 0
    assert any(isinstance(msg, AIMessage) for msg in new_messages)

    # Check result type
    assert isinstance(result, EvaluationResult)


@pytest.mark.asyncio
async def test_evaluate_multi_turn_failed_query():
    """Test multi-turn evaluation handles failed queries correctly."""
    import math
    from unittest.mock import patch

    from ragas.dataset_schema import EvaluationDataset, EvaluationResult, MultiTurnSample
    from ragas.integrations.ag_ui import evaluate_ag_ui_agent

    # Create multi-turn sample
    sample = MultiTurnSample(
        user_input=[HumanMessage(content="Test query")],
        reference_tool_calls=[],
    )

    original_length = len(sample.user_input)
    dataset = EvaluationDataset(samples=[sample])

    class MockExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, func, *args, **kwargs):
            pass

        def results(self):
            # Return NaN to simulate failure
            return [math.nan]

        async def aresults(self):
            return self.results()

    with patch(
        "ragas.integrations.ag_ui.Executor",
        MockExecutor,
    ):
        result = await evaluate_ag_ui_agent(
            endpoint_url="http://localhost:8000/agent",
            dataset=dataset,
            metrics=[],
        )

    # Conversation should remain unchanged after failure
    assert len(sample.user_input) == original_length
    assert isinstance(sample.user_input[0], HumanMessage)
    assert sample.user_input[0].content == "Test query"

    # Check result type
    assert isinstance(result, EvaluationResult)


@pytest.mark.skipif(
    not _has_fastapi_deps(), reason="httpx or ag-ui-protocol not installed"
)
@pytest.mark.asyncio
async def test_evaluate_ag_ui_agent_trace_structure():
    """Test that evaluation creates proper trace structure for debugging."""
    from unittest.mock import patch

    from ragas.callbacks import ChainType
    from ragas.dataset_schema import EvaluationDataset, EvaluationResult, SingleTurnSample
    from ragas.integrations.ag_ui import evaluate_ag_ui_agent

    # Create dataset with 2 samples
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(user_input="Query 1", reference="Ref 1"),
            SingleTurnSample(user_input="Query 2", reference="Ref 2"),
        ]
    )

    # Mock events
    events = [
        RunStartedEvent(run_id="run-1", thread_id="thread-1"),
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Response"),
        TextMessageEndEvent(message_id="msg-1"),
        RunFinishedEvent(run_id="run-1", thread_id="thread-1"),
    ]

    class MockExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, func, *args, **kwargs):
            pass

        def results(self):
            return [events, events]

    with patch("ragas.integrations.ag_ui.Executor", MockExecutor):
        result = await evaluate_ag_ui_agent(
            endpoint_url="http://localhost:8000/agent",
            dataset=dataset,
            metrics=[],  # No metrics for this test
        )

    # Verify result type
    assert isinstance(result, EvaluationResult)

    # Verify trace structure exists
    assert hasattr(result, "ragas_traces")
    traces = result.ragas_traces

    # Should have 3 traces: 1 evaluation + 2 rows (no metrics)
    assert len(traces) == 3

    # Find root evaluation trace
    root_traces = [t for t in traces.values() if t.parent_run_id is None]
    assert len(root_traces) == 1
    root = root_traces[0]

    # Verify root trace properties
    assert root.name == "ag_ui_evaluation"
    assert root.metadata["type"] == ChainType.EVALUATION.value
    assert "endpoint_url" in root.inputs
    assert len(root.children) == 2  # 2 row runs

    # Verify row traces
    for row_id in root.children:
        row_trace = traces[row_id]
        assert row_trace.parent_run_id == root.run_id
        assert row_trace.name.startswith("row ")
        assert row_trace.metadata["type"] == ChainType.ROW.value
        assert "row_index" in row_trace.metadata
