"""Test uvloop compatibility with nest_asyncio."""

import asyncio
import sys

import pytest


class TestUvloopCompatibility:
    """Test that ragas works with uvloop event loops."""

    @pytest.mark.skipif(sys.version_info < (3, 8), reason="uvloop requires Python 3.8+")
    def test_run_with_uvloop_and_running_loop(self):
        """Test that run() raises clear error with uvloop in running event loop."""
        uvloop = pytest.importorskip("uvloop")

        from ragas.async_utils import run

        async def inner_task():
            return "success"

        async def outer_task():
            # Should raise RuntimeError because loop is running
            with pytest.raises(RuntimeError, match="Event loop is already running"):
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
            # this works because we are calling run_async_tasks from outside a loop (via run which calls asyncio.run)
            # Wait, run_async_tasks calls run() internally.
            # If we call it here synchronously, it calls run().
            # run() checks if loop is running.
            # Here we are NOT in a loop (sync context).
            results = run_async_tasks(tasks, show_progress=False)
            assert sorted(results) == [0, 2, 4, 6, 8]
        finally:
            asyncio.set_event_loop_policy(None)

    def test_run_with_standard_asyncio_and_running_loop(self):
        """Test that run() raises RuntimeError with standard asyncio in a running loop."""
        from ragas.async_utils import run

        async def inner_task():
            return "nested_success"

        async def outer_task():
            # This used to work with nest_asyncio, now must fail
            with pytest.raises(RuntimeError, match="Event loop is already running"):
                run(inner_task)

        asyncio.run(outer_task())
