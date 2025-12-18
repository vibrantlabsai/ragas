import asyncio
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAsyncUtilsControl:
    """Test async utils environment detection and execution."""

    def test_run_without_event_loop(self):
        """Test run function works when no event loop is running."""
        from ragas.async_utils import run

        async def test_func():
            return "test"

        result = run(test_func)
        assert result == "test"

    def test_is_jupyter_environment_in_standard_python(self):
        """Test is_jupyter_environment returns False in standard Python."""
        from ragas.async_utils import is_jupyter_environment

        result = is_jupyter_environment()
        assert result is False


class TestEvaluateAsyncControl:
    """Test the sync evaluate function."""

    def test_evaluate_uses_run_utility(self):
        """Test evaluate uses the run utility for async execution."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*coroutine.*was never awaited",
            )

            with patch("ragas.async_utils.run") as mock_run:
                mock_run.return_value = MagicMock()

                from ragas import evaluate

                evaluate(
                    dataset=MagicMock(),
                    metrics=[MagicMock()],
                    show_progress=False,
                )

        # Should call run() which handles environment detection
        mock_run.assert_called_once()


class TestAevaluateImport:
    """Test that aevaluate can be imported and is async."""

    def test_aevaluate_importable(self):
        """Test that aevaluate can be imported."""
        from ragas import aevaluate

        assert callable(aevaluate)
        assert asyncio.iscoroutinefunction(aevaluate)

    def test_evaluate_signature_updated(self):
        """Test that evaluate function no longer has allow_nest_asyncio parameter."""
        import inspect

        from ragas import evaluate

        sig = inspect.signature(evaluate)
        # allow_nest_asyncio parameter should be removed
        assert "allow_nest_asyncio" not in sig.parameters


class TestAevaluateAsyncBehavior:
    """Test that aevaluate works properly in async contexts."""

    @pytest.mark.asyncio
    async def test_aevaluate_is_truly_async(self):
        """Test that aevaluate is a proper async function."""
        from ragas import aevaluate

        # Verify it's a coroutine function
        assert asyncio.iscoroutinefunction(aevaluate)


class TestAsyncIntegration:
    """Basic integration tests for async scenarios."""

    @pytest.mark.asyncio
    async def test_aevaluate_in_running_loop(self):
        """Test aevaluate can be called when an event loop is already running."""
        # This test runs with pytest-asyncio, so an event loop is running
        from ragas import aevaluate

        # Just test that the function can be called without RuntimeError
        # We'll mock everything to avoid API calls
        with patch("ragas.evaluation.EvaluationDataset"):
            with patch("ragas.evaluation.validate_required_columns"):
                with patch("ragas.evaluation.validate_supported_metrics"):
                    with patch("ragas.evaluation.Executor") as mock_executor_class:
                        with patch("ragas.evaluation.new_group"):
                            mock_executor = MagicMock()
                            mock_executor.aresults = AsyncMock(return_value=[])
                            mock_executor_class.return_value = mock_executor

                            try:
                                await aevaluate(
                                    dataset=MagicMock(),
                                    metrics=[],
                                    show_progress=False,
                                )
                                # Should not raise RuntimeError about event loop
                            except Exception as e:
                                # We expect other exceptions due to mocking, but not RuntimeError
                                assert "event loop" not in str(e).lower()
                                assert "nest_asyncio" not in str(e).lower()
