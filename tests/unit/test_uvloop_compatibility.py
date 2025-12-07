"""Test uvloop compatibility and async execution."""

import asyncio
import sys

import pytest


class TestUvloopCompatibility:
    """Test that ragas works with uvloop event loops."""

    @pytest.mark.skipif(sys.version_info < (3, 8), reason="uvloop requires Python 3.8+")
    def test_is_event_loop_running_with_uvloop(self):
        """Test that is_event_loop_running works with uvloop."""
        uvloop = pytest.importorskip("uvloop")

        from ragas.async_utils import is_event_loop_running

        async def test_func():
            result = is_event_loop_running()
            return result

        uvloop.install()
        try:
            result = asyncio.run(test_func())
            assert result is True
        finally:
            asyncio.set_event_loop_policy(None)

    @pytest.mark.skipif(sys.version_info < (3, 8), reason="uvloop requires Python 3.8+")
    def test_run_with_uvloop_and_running_loop(self):
        """Test that run() raises clear error with uvloop in running event loop (Jupyter scenario)."""
        uvloop = pytest.importorskip("uvloop")

        from ragas.async_utils import run

        async def inner_task():
            return "success"

        async def outer_task():
            with pytest.raises(RuntimeError, match="Cannot execute nested async code"):
                run(inner_task)

        uvloop.install()
        try:
            asyncio.run(outer_task())
        finally:
            asyncio.set_event_loop_policy(None)

    @pytest.mark.skipif(sys.version_info < (3, 8), reason="uvloop requires Python 3.8+")
    def test_run_async_tasks_with_uvloop(self):
        """Test that run_async_tasks works with uvloop."""
        uvloop = pytest.importorskip("uvloop")

        from ragas.async_utils import run_async_tasks

        async def task(n):
            return n * 2

        tasks = [task(i) for i in range(5)]

        uvloop.install()
        try:
            results = run_async_tasks(tasks, show_progress=False)
            assert sorted(results) == [0, 2, 4, 6, 8]
        finally:
            asyncio.set_event_loop_policy(None)

    def test_is_jupyter_environment_returns_false(self):
        """Test that is_jupyter_environment returns False in non-Jupyter environments."""
        from ragas.async_utils import is_jupyter_environment

        result = is_jupyter_environment()
        assert result is False

    def test_run_without_running_loop(self):
        """Test that run() works when no event loop is running."""
        from ragas.async_utils import run

        async def task():
            return "success"

        result = run(task)
        assert result == "success"
