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

    from ragas.integrations.ag_ui import call_ag_ui_endpoint

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
        events = await call_ag_ui_endpoint(
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

    from ragas.integrations.ag_ui import call_ag_ui_endpoint

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
        events = await call_ag_ui_endpoint(
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

    from ragas.integrations.ag_ui import call_ag_ui_endpoint

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
        events = await call_ag_ui_endpoint(
            endpoint_url="http://localhost:8000/agent",
            user_input="Test",
        )

    # Should skip malformed event but collect valid ones
    assert len(events) == 2
    assert events[0].type == "RUN_STARTED"
    assert events[1].type == "RUN_FINISHED"


# ============================================================================
# Experiment-based evaluation tests (new @experiment pattern)
# ============================================================================


def test_convert_ragas_messages_to_ag_ui():
    """Test converting Ragas messages to AG-UI format."""
    from ragas.integrations.ag_ui import convert_messages_to_ag_ui
    from ragas.messages import ToolCall

    messages = [
        HumanMessage(content="What's the weather?"),
        AIMessage(
            content="Let me check",
            tool_calls=[ToolCall(name="get-weather", args={"location": "SF"})],
        ),
        HumanMessage(content="Thanks!"),
    ]

    ag_ui_messages = convert_messages_to_ag_ui(messages)

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


# ---------------------------------------------------------------------------
# Tests for extraction helpers
# ---------------------------------------------------------------------------


def test_extract_response():
    """Test extract_response extracts AI message content."""
    from ragas.integrations.ag_ui import extract_response

    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! "),
        AIMessage(content="How can I help?"),
        ToolMessage(content="Tool result"),
    ]

    response = extract_response(messages)
    assert response == "Hi there! How can I help?"


def test_extract_response_empty():
    """Test extract_response returns empty string when no AI content."""
    from ragas.integrations.ag_ui import extract_response

    messages = [
        HumanMessage(content="Hello"),
        ToolMessage(content="Tool result"),
    ]

    response = extract_response(messages)
    assert response == ""


def test_extract_tool_calls():
    """Test extract_tool_calls extracts tool calls from AI messages."""
    from ragas.integrations.ag_ui import extract_tool_calls
    from ragas.messages import ToolCall

    messages = [
        AIMessage(
            content="Let me check",
            tool_calls=[
                ToolCall(name="get_weather", args={"location": "SF"}),
                ToolCall(name="get_time", args={"timezone": "PST"}),
            ],
        ),
        AIMessage(
            content="More info",
            tool_calls=[ToolCall(name="search", args={"query": "test"})],
        ),
    ]

    tool_calls = extract_tool_calls(messages)
    assert len(tool_calls) == 3
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[1].name == "get_time"
    assert tool_calls[2].name == "search"


def test_extract_tool_calls_empty():
    """Test extract_tool_calls returns empty list when no tool calls."""
    from ragas.integrations.ag_ui import extract_tool_calls

    messages = [
        AIMessage(content="Just a response"),
        HumanMessage(content="Question"),
    ]

    tool_calls = extract_tool_calls(messages)
    assert tool_calls == []


def test_extract_contexts():
    """Test extract_contexts extracts tool message content."""
    from ragas.integrations.ag_ui import extract_contexts

    messages = [
        AIMessage(content="Let me check"),
        ToolMessage(content="Weather: Sunny, 72F"),
        AIMessage(content="The weather is nice"),
        ToolMessage(content="Time: 3:00 PM"),
    ]

    contexts = extract_contexts(messages)
    assert len(contexts) == 2
    assert contexts[0] == "Weather: Sunny, 72F"
    assert contexts[1] == "Time: 3:00 PM"


def test_extract_contexts_empty():
    """Test extract_contexts returns empty list when no tool messages."""
    from ragas.integrations.ag_ui import extract_contexts

    messages = [
        AIMessage(content="Response"),
        HumanMessage(content="Question"),
    ]

    contexts = extract_contexts(messages)
    assert contexts == []


# ---------------------------------------------------------------------------
# Tests for build_sample
# ---------------------------------------------------------------------------


def test_build_sample_single_turn():
    """Test build_sample creates SingleTurnSample for simple input."""
    from ragas.dataset_schema import SingleTurnSample
    from ragas.integrations.ag_ui import build_sample

    messages = [
        AIMessage(content="The answer is 42."),
    ]

    sample = build_sample(
        user_input="What is the meaning of life?",
        messages=messages,
        reference="42 is the answer.",
    )

    assert isinstance(sample, SingleTurnSample)
    assert sample.user_input == "What is the meaning of life?"
    assert sample.response == "The answer is 42."
    assert sample.reference == "42 is the answer."


def test_build_sample_multi_turn_with_list_input():
    """Test build_sample creates MultiTurnSample when user_input is a list."""
    from ragas.dataset_schema import MultiTurnSample
    from ragas.integrations.ag_ui import build_sample

    user_input = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
        HumanMessage(content="What's the weather?"),
    ]
    messages = [AIMessage(content="It's sunny!")]

    sample = build_sample(
        user_input=user_input,
        messages=messages,
        reference="Weather info",
    )

    assert isinstance(sample, MultiTurnSample)
    # Conversation should include original + agent response
    assert len(sample.user_input) == 4


def test_build_sample_multi_turn_with_tool_calls():
    """Test build_sample creates MultiTurnSample when reference_tool_calls provided."""
    from ragas.dataset_schema import MultiTurnSample
    from ragas.integrations.ag_ui import build_sample
    from ragas.messages import ToolCall

    messages = [
        AIMessage(
            content="Checking weather",
            tool_calls=[ToolCall(name="get_weather", args={"location": "SF"})],
        ),
    ]
    reference_tool_calls = [ToolCall(name="get_weather", args={"location": "SF"})]

    sample = build_sample(
        user_input="What's the weather in SF?",
        messages=messages,
        reference_tool_calls=reference_tool_calls,
    )

    assert isinstance(sample, MultiTurnSample)
    assert sample.reference_tool_calls == reference_tool_calls


# ---------------------------------------------------------------------------
# Tests for run_ag_ui_row
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_fastapi_deps(), reason="httpx or ag-ui-protocol not installed"
)
@pytest.mark.asyncio
async def test_run_ag_ui_row_processes_row():
    """Test that run_ag_ui_row processes rows correctly."""
    from ragas.integrations.ag_ui import run_ag_ui_row

    # Mock events
    events = [
        RunStartedEvent(run_id="run-1", thread_id="thread-1"),
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Hello! I'm here to help."),
        TextMessageEndEvent(message_id="msg-1"),
        RunFinishedEvent(run_id="run-1", thread_id="thread-1"),
    ]

    async def mock_call_endpoint(endpoint_url, user_input, **kwargs):
        return events

    with patch(
        "ragas.integrations.ag_ui.call_ag_ui_endpoint",
        side_effect=mock_call_endpoint,
    ):
        result = await run_ag_ui_row(
            {"user_input": "Hello", "reference": "Test reference"},
            endpoint_url="http://localhost:8000/agent",
        )

    # Check result structure
    assert "user_input" in result
    assert "response" in result
    assert "messages" in result
    assert "tool_calls" in result
    assert "contexts" in result
    assert "reference" in result
    assert result["user_input"] == "Hello"
    assert result["response"] == "Hello! I'm here to help."
    assert result["reference"] == "Test reference"
    assert len(result["messages"]) == 1  # One AIMessage


@pytest.mark.skipif(
    not _has_fastapi_deps(), reason="httpx or ag-ui-protocol not installed"
)
@pytest.mark.asyncio
async def test_run_ag_ui_row_extracts_tool_results():
    """Test that run_ag_ui_row extracts tool results into contexts."""
    from ragas.integrations.ag_ui import run_ag_ui_row

    # Mock events with tool call
    events = [
        RunStartedEvent(run_id="run-1", thread_id="thread-1"),
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="Let me check"),
        ToolCallStartEvent(tool_call_id="tc-1", tool_call_name="get_weather"),
        ToolCallArgsEvent(tool_call_id="tc-1", delta='{"location": "SF"}'),
        ToolCallEndEvent(tool_call_id="tc-1"),
        TextMessageEndEvent(message_id="msg-1"),
        ToolCallResultEvent(
            tool_call_id="tc-1",
            message_id="result-1",
            content="Sunny, 72F",
        ),
        RunFinishedEvent(run_id="run-1", thread_id="thread-1"),
    ]

    async def mock_call_endpoint(endpoint_url, user_input, **kwargs):
        return events

    with patch(
        "ragas.integrations.ag_ui.call_ag_ui_endpoint",
        side_effect=mock_call_endpoint,
    ):
        result = await run_ag_ui_row(
            {"user_input": "What's the weather?", "reference": "Weather info"},
            endpoint_url="http://localhost:8000/agent",
        )

    # Check that tool results were extracted to contexts
    assert "contexts" in result
    assert len(result["contexts"]) > 0
    # Tool result content should be in contexts
    assert "Sunny, 72F" in result["contexts"][0]
    # Tool calls should also be extracted
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0].name == "get_weather"


@pytest.mark.skipif(
    not _has_fastapi_deps(), reason="httpx or ag-ui-protocol not installed"
)
@pytest.mark.asyncio
async def test_run_ag_ui_row_handles_empty_user_input():
    """Test that run_ag_ui_row handles empty user_input."""
    from ragas.integrations.ag_ui import MISSING_RESPONSE_PLACEHOLDER, run_ag_ui_row

    # Mock endpoint that returns empty response
    async def mock_call_endpoint(endpoint_url, user_input, **kwargs):
        # Return minimal events with no content
        return [
            RunStartedEvent(run_id="run-1", thread_id="thread-1"),
            RunFinishedEvent(run_id="run-1", thread_id="thread-1"),
        ]

    with patch(
        "ragas.integrations.ag_ui.call_ag_ui_endpoint",
        side_effect=mock_call_endpoint,
    ):
        result = await run_ag_ui_row(
            {"user_input": "", "reference": "Test"},
            endpoint_url="http://localhost:8000/agent",
        )

    # With empty user_input but successful endpoint call, response is the placeholder
    assert result["response"] == MISSING_RESPONSE_PLACEHOLDER
    assert result["user_input"] == ""


@pytest.mark.skipif(
    not _has_fastapi_deps(), reason="httpx or ag-ui-protocol not installed"
)
@pytest.mark.asyncio
async def test_run_ag_ui_row_handles_none_user_input():
    """Test that run_ag_ui_row handles None user_input."""
    from ragas.integrations.ag_ui import MISSING_RESPONSE_PLACEHOLDER, run_ag_ui_row

    # Call with None user_input (no mocking - should return immediately)
    result = await run_ag_ui_row(
        {"reference": "Test"},
        endpoint_url="http://localhost:8000/agent",
    )

    # Should return placeholder response when user_input is missing
    assert result["response"] == MISSING_RESPONSE_PLACEHOLDER
    assert result.get("user_input") is None


@pytest.mark.skipif(
    not _has_fastapi_deps(), reason="httpx or ag-ui-protocol not installed"
)
@pytest.mark.asyncio
async def test_run_ag_ui_row_handles_multi_turn_input():
    """Test that run_ag_ui_row handles multi-turn conversation input."""
    from ragas.integrations.ag_ui import run_ag_ui_row

    # Mock events for agent response
    events = [
        RunStartedEvent(run_id="run-1", thread_id="thread-1"),
        TextMessageStartEvent(message_id="msg-1", role="assistant"),
        TextMessageContentEvent(message_id="msg-1", delta="It's sunny!"),
        TextMessageEndEvent(message_id="msg-1"),
        RunFinishedEvent(run_id="run-1", thread_id="thread-1"),
    ]

    async def mock_call_endpoint(endpoint_url, user_input, **kwargs):
        return events

    # Multi-turn input as list of messages
    conversation = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
        HumanMessage(content="What's the weather?"),
    ]

    with patch(
        "ragas.integrations.ag_ui.call_ag_ui_endpoint",
        side_effect=mock_call_endpoint,
    ):
        result = await run_ag_ui_row(
            {"user_input": conversation, "reference": "Weather info"},
            endpoint_url="http://localhost:8000/agent",
        )

    # Response should be extracted from agent events
    assert result["response"] == "It's sunny!"
    # Original conversation is preserved in result
    assert "user_input" in result
    assert len(result["user_input"]) == len(conversation)


@pytest.mark.skipif(
    not _has_fastapi_deps(), reason="httpx or ag-ui-protocol not installed"
)
@pytest.mark.asyncio
async def test_run_ag_ui_row_with_extra_headers():
    """Test that extra headers are passed to the endpoint."""
    from ragas.integrations.ag_ui import run_ag_ui_row

    captured_kwargs = {}

    async def mock_call_endpoint(endpoint_url, user_input, **kwargs):
        captured_kwargs.update(kwargs)
        return [
            RunStartedEvent(run_id="run-1", thread_id="thread-1"),
            TextMessageStartEvent(message_id="msg-1", role="assistant"),
            TextMessageContentEvent(message_id="msg-1", delta="Response"),
            TextMessageEndEvent(message_id="msg-1"),
            RunFinishedEvent(run_id="run-1", thread_id="thread-1"),
        ]

    with patch(
        "ragas.integrations.ag_ui.call_ag_ui_endpoint",
        side_effect=mock_call_endpoint,
    ):
        await run_ag_ui_row(
            {"user_input": "Test", "reference": "Ref"},
            endpoint_url="http://localhost:8000/agent",
            extra_headers={"Authorization": "Bearer test-token"},
        )

    # Check that extra headers were passed
    assert "extra_headers" in captured_kwargs
    assert captured_kwargs["extra_headers"]["Authorization"] == "Bearer test-token"


@pytest.mark.skipif(
    not _has_fastapi_deps(), reason="httpx or ag-ui-protocol not installed"
)
@pytest.mark.asyncio
async def test_run_ag_ui_row_handles_endpoint_failure():
    """Test that run_ag_ui_row handles endpoint failures gracefully."""
    from ragas.integrations.ag_ui import (
        MISSING_CONTEXT_PLACEHOLDER,
        MISSING_RESPONSE_PLACEHOLDER,
        run_ag_ui_row,
    )

    async def mock_call_endpoint_failure(endpoint_url, user_input, **kwargs):
        raise Exception("Connection refused")

    with patch(
        "ragas.integrations.ag_ui.call_ag_ui_endpoint",
        side_effect=mock_call_endpoint_failure,
    ):
        # Should return result with placeholder values instead of raising
        result = await run_ag_ui_row(
            {"user_input": "Test", "reference": "Ref"},
            endpoint_url="http://localhost:8000/agent",
        )

    # Verify graceful failure handling
    assert result["response"] == MISSING_RESPONSE_PLACEHOLDER
    assert result["contexts"] == [MISSING_CONTEXT_PLACEHOLDER]
    assert result["user_input"] == "Test"
    assert result["reference"] == "Ref"
    assert result["messages"] == []
    assert result["tool_calls"] == []
