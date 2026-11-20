"""
Comprehensive test suite for tracing integrations.

Tests both Langfuse and MLflow integrations with proper mocking
to avoid external dependencies in tests.
"""

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLangfuseIntegration:
    """Test suite for Langfuse tracing integration."""

    def test_langfuse_imports_with_missing_dependency(self):
        """Test that imports work gracefully when langfuse is not available."""
        with patch.dict("sys.modules", {"langfuse": None, "langfuse.api": None}):
            # This should not raise an ImportError
            from ragas.integrations.tracing.langfuse import (
                LangfuseTrace,
                observe,
                sync_trace,
            )

            assert callable(observe)
            assert LangfuseTrace is not None
            assert callable(sync_trace)

    def test_langfuse_imports_with_dependency_available(self):
        """Test imports when langfuse is available."""
        # Mock langfuse modules
        mock_langfuse = MagicMock()
        mock_api = MagicMock()

        with patch.dict(
            "sys.modules", {"langfuse": mock_langfuse, "langfuse.api": mock_api}
        ):
            from ragas.integrations.tracing.langfuse import LangfuseTrace, observe

            assert LangfuseTrace is not None
            assert callable(observe)

    def test_observe_decorator_fallback(self):
        """Test that observe decorator works as a no-op when langfuse unavailable."""
        with patch.dict("sys.modules", {"langfuse": None}):
            from ragas.integrations.tracing.langfuse import observe

            @observe()
            def test_function():
                return "test_result"

            result = test_function()
            assert result == "test_result"

    def test_langfuse_trace_initialization(self):
        """Test LangfuseTrace initialization with mock trace."""
        from ragas.integrations.tracing.langfuse import LangfuseTrace

        # Use MagicMock instead of trying to instantiate the real class
        mock_trace = MagicMock()
        mock_trace.id = "test-trace-id"
        mock_trace.timestamp = datetime.now()
        mock_trace.htmlPath = "test-path"
        mock_trace.latency = 100
        mock_trace.totalCost = 0.01

        langfuse_trace = LangfuseTrace(mock_trace)
        assert langfuse_trace.trace == mock_trace

    def test_langfuse_trace_filters_observations_by_name(self):
        """Test that filtering returns only observations with the requested name."""
        from ragas.integrations.tracing.langfuse import LangfuseTrace

        matching_observation = MagicMock()
        matching_observation.name = "retrieval"
        other_observation = MagicMock()
        other_observation.name = "generation"
        mock_trace = MagicMock()
        mock_trace.observations = [matching_observation, other_observation]

        langfuse_trace = LangfuseTrace(mock_trace)

        assert langfuse_trace.filter("retrieval") == [matching_observation]

    @pytest.mark.asyncio
    async def test_sync_trace_with_trace_id(self):
        """Test sync_trace function with explicit trace ID."""
        from ragas.integrations.tracing.langfuse import sync_trace

        # Mock the Langfuse client
        with patch(
            "ragas.integrations.tracing.langfuse.Langfuse"
        ) as mock_langfuse_class:
            mock_client = MagicMock()
            mock_trace = MagicMock()
            mock_client.api.trace.get.return_value = mock_trace
            mock_langfuse_class.return_value = mock_client

            result = await sync_trace(
                trace_id="test-trace-id", max_retries=1, delay=0.1
            )

            assert result.trace is mock_trace
            mock_client.api.trace.get.assert_called_once_with("test-trace-id")

    @pytest.mark.asyncio
    async def test_sync_trace_without_trace_id(self):
        """Test sync_trace function without trace ID (uses current trace)."""
        from ragas.integrations.tracing.langfuse import sync_trace

        with patch(
            "ragas.integrations.tracing.langfuse.Langfuse"
        ) as mock_langfuse_class:
            mock_client = MagicMock()
            mock_client.get_current_trace_id.return_value = "current-trace-id"
            mock_trace = MagicMock()
            mock_client.api.trace.get.return_value = mock_trace
            mock_langfuse_class.return_value = mock_client

            result = await sync_trace(max_retries=1, delay=0.1)

            assert result.trace is mock_trace
            mock_client.get_current_trace_id.assert_called_once()
            mock_client.api.trace.get.assert_called_once_with("current-trace-id")

    @pytest.mark.asyncio
    async def test_sync_trace_retries_transient_failures(self):
        """Test that sync_trace retries when a trace is not available yet."""
        from ragas.integrations.tracing.langfuse import sync_trace

        with (
            patch(
                "ragas.integrations.tracing.langfuse.Langfuse"
            ) as mock_langfuse_class,
            patch(
                "ragas.integrations.tracing.langfuse.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            mock_client = MagicMock()
            mock_trace = MagicMock()
            mock_client.api.trace.get.side_effect = [
                RuntimeError("trace is still syncing"),
                mock_trace,
            ]
            mock_langfuse_class.return_value = mock_client

            result = await sync_trace(
                trace_id="test-trace-id", max_retries=2, delay=0.25
            )

            assert result.trace is mock_trace
            assert mock_client.api.trace.get.call_count == 2
            mock_sleep.assert_awaited_once_with(0.25)

    @pytest.mark.asyncio
    async def test_sync_trace_raises_after_retries_are_exhausted(self):
        """Test that sync_trace reports when the trace never becomes available."""
        from ragas.integrations.tracing.langfuse import sync_trace

        with (
            patch(
                "ragas.integrations.tracing.langfuse.Langfuse"
            ) as mock_langfuse_class,
            patch(
                "ragas.integrations.tracing.langfuse.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            mock_client = MagicMock()
            mock_client.api.trace.get.side_effect = RuntimeError(
                "trace is still syncing"
            )
            mock_langfuse_class.return_value = mock_client

            with pytest.raises(
                ValueError, match="Trace test-trace-id not found after 2 attempts"
            ):
                await sync_trace(trace_id="test-trace-id", max_retries=2, delay=0.25)

            assert mock_client.api.trace.get.call_count == 2
            mock_sleep.assert_awaited_once_with(0.25)

    @pytest.mark.asyncio
    async def test_sync_trace_no_trace_found(self):
        """Test sync_trace raises ValueError when no trace is found."""
        from ragas.integrations.tracing.langfuse import sync_trace

        with patch(
            "ragas.integrations.tracing.langfuse.Langfuse"
        ) as mock_langfuse_class:
            mock_client = MagicMock()
            mock_client.get_current_trace_id.return_value = None
            mock_langfuse_class.return_value = mock_client

            with pytest.raises(ValueError, match="No trace id found"):
                await sync_trace(max_retries=1, delay=0.1)

    @pytest.mark.asyncio
    async def test_sync_trace_without_langfuse(self):
        """Test sync_trace reports the missing dependency for an explicit trace ID."""
        from ragas.integrations.tracing.langfuse import sync_trace

        with patch("ragas.integrations.tracing.langfuse.LANGFUSE_AVAILABLE", False):
            with pytest.raises(ImportError, match="pip install langfuse"):
                await sync_trace(trace_id="test-trace-id", max_retries=1)

    def test_add_query_param(self):
        """Test URL query parameter addition utility."""
        from ragas.integrations.tracing.langfuse import add_query_param

        base_url = "https://example.com/trace"
        result = add_query_param(base_url, "param", "value")

        assert "param=value" in result
        assert result.startswith("https://example.com/trace")

    def test_add_query_param_existing_params(self):
        """Test URL query parameter addition with existing parameters."""
        from ragas.integrations.tracing.langfuse import add_query_param

        base_url = "https://example.com/trace?existing=param"
        result = add_query_param(base_url, "new", "value")

        assert "existing=param" in result
        assert "new=value" in result


class TestMLflowIntegration:
    """Test suite for MLflow tracing integration."""

    def test_mlflow_imports_with_missing_dependency(self):
        """Test that imports work gracefully when mlflow is not available."""
        with patch.dict("sys.modules", {"mlflow": None, "mlflow.entities": None}):
            from ragas.integrations.tracing.mlflow import MLflowTrace, sync_trace

            assert MLflowTrace is not None
            assert callable(sync_trace)

    def test_mlflow_imports_with_dependency_available(self):
        """Test imports when mlflow is available."""
        mock_mlflow = MagicMock()
        mock_entities = MagicMock()

        with patch.dict(
            "sys.modules", {"mlflow": mock_mlflow, "mlflow.entities": mock_entities}
        ):
            from ragas.integrations.tracing.mlflow import MLflowTrace

            assert MLflowTrace is not None

    def test_mlflow_trace_initialization(self):
        """Test MLflowTrace initialization with mock trace."""
        from ragas.integrations.tracing.mlflow import MLflowTrace

        # Use MagicMock instead of trying to instantiate the real class
        mock_trace = MagicMock()
        mlflow_trace = MLflowTrace(mock_trace)
        assert mlflow_trace.trace == mock_trace

    def test_mlflow_trace_get_url_with_env(self):
        """Test MLflowTrace URL generation with MLFLOW_HOST set."""
        from ragas.integrations.tracing.mlflow import MLflowTrace

        # Use MagicMock for the trace object
        mock_trace = MagicMock()
        mock_trace.info = MagicMock()
        mock_trace.info.request_id = "test-request-id"
        mock_trace.info.experiment_id = "test-experiment-id"

        with patch.dict(os.environ, {"MLFLOW_HOST": "https://mlflow.example.com/"}):
            mlflow_trace = MLflowTrace(mock_trace)
            url = mlflow_trace.get_url()

            assert "https://mlflow.example.com" in url
            assert "test-request-id" in url
            assert "test-experiment-id" in url

    def test_mlflow_trace_get_url_no_env(self):
        """Test MLflowTrace URL generation without MLFLOW_HOST."""
        from ragas.integrations.tracing.mlflow import MLflowTrace

        # Use MagicMock for the trace object
        mock_trace = MagicMock()
        mlflow_trace = MLflowTrace(mock_trace)

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="MLFLOW_HOST environment variable is not set"
            ):
                mlflow_trace.get_url()

    def test_mlflow_trace_filter(self):
        """Test MLflowTrace span filtering."""
        from ragas.integrations.tracing.mlflow import MLflowTrace

        # Use MagicMock for both span and trace objects
        mock_span = MagicMock()
        mock_span.name = "test-span"

        mock_trace = MagicMock()
        mock_trace.search_spans = MagicMock(return_value=[mock_span])

        mlflow_trace = MLflowTrace(mock_trace)
        filtered_spans = mlflow_trace.get_filter("test-span")

        assert len(filtered_spans) == 1
        assert filtered_spans[0] == mock_span
        mock_trace.search_spans.assert_called_once_with(name="test-span")

    @pytest.mark.asyncio
    async def test_mlflow_sync_trace_success(self):
        """Test successful MLflow trace synchronization."""
        from ragas.integrations.tracing.mlflow import sync_trace

        with (
            patch(
                "ragas.integrations.tracing.mlflow.get_last_active_trace_id"
            ) as mock_get_id,
            patch("ragas.integrations.tracing.mlflow.get_trace") as mock_get_trace,
        ):
            mock_get_id.return_value = "test-trace-id"
            mock_trace = MagicMock()
            mock_get_trace.return_value = mock_trace

            result = await sync_trace()

            assert result is not None
            assert result.trace == mock_trace
            mock_get_id.assert_called_once()
            mock_get_trace.assert_called_once_with("test-trace-id")

    @pytest.mark.asyncio
    async def test_mlflow_sync_trace_no_active_trace(self):
        """Test MLflow sync_trace when no active trace exists."""
        from ragas.integrations.tracing.mlflow import sync_trace

        with patch(
            "ragas.integrations.tracing.mlflow.get_last_active_trace_id"
        ) as mock_get_id:
            mock_get_id.return_value = None

            with pytest.raises(ValueError, match="No active trace found"):
                await sync_trace()

    @pytest.mark.asyncio
    async def test_mlflow_sync_trace_not_found(self):
        """Test MLflow sync_trace when trace is not found."""
        from ragas.integrations.tracing.mlflow import sync_trace

        with (
            patch(
                "ragas.integrations.tracing.mlflow.get_last_active_trace_id"
            ) as mock_get_id,
            patch("ragas.integrations.tracing.mlflow.get_trace") as mock_get_trace,
        ):
            mock_get_id.return_value = "test-trace-id"
            mock_get_trace.return_value = None

            with pytest.raises(ValueError, match="Trace not found"):
                await sync_trace()


class TestTracingIntegrationInitModule:
    """Test the tracing integration __init__ module."""

    def test_lazy_import_langfuse_functions(self):
        """Test lazy imports for Langfuse functions."""
        from ragas.integrations.tracing import LangfuseTrace, observe, sync_trace

        assert callable(observe)
        assert callable(sync_trace)
        assert LangfuseTrace is not None

    def test_lazy_import_mlflow_classes(self):
        """Test lazy imports for MLflow classes."""
        from ragas.integrations.tracing import MLflowTrace

        assert MLflowTrace is not None

    def test_invalid_attribute_access(self):
        """Test that accessing non-existent attributes raises AttributeError."""
        import ragas.integrations.tracing as tracing

        with pytest.raises(AttributeError, match="has no attribute 'non_existent'"):
            _ = tracing.non_existent


class TestTracingWithCallbackSystem:
    """Test tracing integrations with the existing callback system."""

    def test_tracing_with_ragas_tracer(self):
        """Test that tracing can work alongside RagasTracer."""
        from ragas.callbacks import RagasTracer
        from ragas.integrations.tracing.langfuse import observe

        tracer = RagasTracer()

        @observe()
        def traced_function():
            return "test_result"

        # Should work without conflicts
        result = traced_function()
        assert result == "test_result"

        # Tracer should still be functional
        assert isinstance(tracer.traces, dict)

    def test_callback_manager_compatibility(self):
        """Test compatibility with LangChain callback manager."""
        from langchain_core.callbacks import CallbackManager

        from ragas.callbacks import RagasTracer
        from ragas.integrations.tracing.langfuse import observe

        tracer = RagasTracer()
        callback_manager = CallbackManager([tracer])

        @observe()
        def evaluation_function():
            return {"score": 0.85}

        result = evaluation_function()
        assert result["score"] == 0.85

        # Should not interfere with callback functionality
        assert len(callback_manager.handlers) == 1


if __name__ == "__main__":
    pytest.main([__file__])
