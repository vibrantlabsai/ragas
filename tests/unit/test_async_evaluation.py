import asyncio
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAsyncUtilsControl:
    """Test async utils behavior."""

    def test_run_executes_coroutine(self):
        """Test run function executes coroutine successfully."""
        from ragas.async_utils import run

        async def test_func():
            return "test"

        # When running without a loop
        result = run(test_func)
        assert result == "test"

    @pytest.mark.asyncio
    async def test_run_raises_in_loop(self):
        """Test run function raises RuntimeError when called inside a loop."""
        from ragas.async_utils import run

        async def test_func():
            return "test"

        # When running inside a loop (pytest-asyncio provides one)
        with pytest.raises(RuntimeError, match="Event loop is already running"):
            run(test_func)


class TestEvaluateAsyncControl:
    """Test the sync evaluate function behavior."""

    def test_evaluate_calls_run(self):
        """Test evaluate calls run (which handles the loop check)."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*coroutine.*was never awaited",
            )
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message="evaluate.*deprecated",
            )

            with patch("ragas.async_utils.run") as mock_run:
                mock_run.return_value = MagicMock()

                from ragas import evaluate

                evaluate(
                    dataset=MagicMock(),
                    metrics=[MagicMock()],
                    show_progress=False,
                )

        # Should use run()
        mock_run.assert_called_once()


class TestAevaluateImport:
    """Test that aevaluate can be imported and is async."""

    def test_aevaluate_importable(self):
        """Test that aevaluate can be imported."""
        from ragas import aevaluate

        assert callable(aevaluate)
        assert asyncio.iscoroutinefunction(aevaluate)

    def test_evaluate_has_deprecated_param(self):
        """Test that evaluate function still has the parameter but it is deprecated."""
        import inspect

        from ragas import evaluate

        sig = inspect.signature(evaluate)
        assert "allow_nest_asyncio" in sig.parameters


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

