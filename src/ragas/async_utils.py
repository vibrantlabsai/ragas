"""Async utils."""

import asyncio
import logging
import typing as t

logger = logging.getLogger(__name__)


def is_event_loop_running() -> bool:
    """
    Check if an event loop is currently running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    else:
        return loop.is_running()


def is_jupyter_environment() -> bool:
    """
    Check if code is running in a Jupyter-like environment.

    Returns:
        bool: True if running in Jupyter, False otherwise
    """
    try:
        # Check for IPython/Jupyter kernel
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return False

        # Check if it's a kernel-based IPython (Jupyter)
        return hasattr(ipython, "kernel")
    except ImportError:
        return False


def as_completed(
    coroutines: t.Sequence[t.Coroutine],
    max_workers: int = -1,
    *,
    cancel_check: t.Optional[t.Callable[[], bool]] = None,
    cancel_pending: bool = True,
) -> t.Iterator[asyncio.Future]:
    """
    Wrap coroutines with a semaphore if max_workers is specified.

    Returns an iterator of futures that completes as tasks finish.
    """
    if max_workers == -1:
        tasks = [asyncio.create_task(coro) for coro in coroutines]
    else:
        semaphore = asyncio.Semaphore(max_workers)

        async def sema_coro(coro):
            async with semaphore:
                return await coro

        tasks = [asyncio.create_task(sema_coro(coro)) for coro in coroutines]

    ac_iter = asyncio.as_completed(tasks)

    if cancel_check is None:
        return ac_iter

    def _iter_with_cancel():
        for future in ac_iter:
            if cancel_check():
                if cancel_pending:
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                break
            yield future

    return _iter_with_cancel()


async def process_futures(
    futures: t.Iterator[asyncio.Future],
) -> t.AsyncGenerator[t.Any, None]:
    """
    Process futures with optional progress tracking.

    Args:
        futures: Iterator of asyncio futures to process (e.g., from asyncio.as_completed)

    Yields:
        Results from completed futures as they finish
    """
    # Process completed futures as they finish
    for future in futures:
        try:
            result = await future
        except asyncio.CancelledError:
            raise  # Re-raise CancelledError to ensure proper cancellation
        except Exception as e:
            result = e
        yield result


def run(
    async_func: t.Union[
        t.Callable[[], t.Coroutine[t.Any, t.Any, t.Any]],
        t.Coroutine[t.Any, t.Any, t.Any],
    ],
) -> t.Any:
    """
    Run an async function, handling both Jupyter and standard environments.

    This function automatically detects the execution environment:
    - In Jupyter notebooks: schedules coroutine on the existing event loop
    - In standard Python: creates a new event loop with asyncio.run()

    Parameters
    ----------
    async_func : Callable or Coroutine
        The async function or coroutine to run

    Returns
    -------
    Any
        The result of the async function

    Raises
    ------
    RuntimeError
        If an event loop is running in a non-Jupyter environment
    """
    coro = async_func() if callable(async_func) else async_func

    if is_event_loop_running():
        # Check if we're in a Jupyter environment
        if is_jupyter_environment():
            # In Jupyter, schedule on the existing loop
            import asyncio

            loop = asyncio.get_running_loop()
            # Create a task and run it
            task = loop.create_task(coro)
            # Use a simple polling approach to wait for the task
            while not task.done():
                loop._run_once()
                if task.done():
                    break
            return task.result()
        else:
            # In non-Jupyter with running loop, this is an error
            loop = asyncio.get_running_loop()
            loop_type = type(loop).__name__
            raise RuntimeError(
                f"Cannot execute nested async code with {loop_type}. "
                f"An event loop is already running in this context. "
                f"Please use 'await' instead of calling this synchronous wrapper, "
                f"or refactor your code to avoid nested async calls."
            )

    return asyncio.run(coro)


def run_async_tasks(
    tasks: t.Sequence[t.Coroutine],
    batch_size: t.Optional[int] = None,
    show_progress: bool = True,
    progress_bar_desc: str = "Running async tasks",
    max_workers: int = -1,
    *,
    cancel_check: t.Optional[t.Callable[[], bool]] = None,
) -> t.List[t.Any]:
    """
    Execute async tasks with optional batching and progress tracking.

    NOTE: Order of results is not guaranteed!

    Args:
        tasks: Sequence of coroutines to execute
        batch_size: Optional size for batching tasks. If None, runs all concurrently
        show_progress: Whether to display progress bars
        max_workers: Maximum number of concurrent tasks (-1 for unlimited)
    """
    from ragas.utils import ProgressBarManager, batched

    async def _run():
        total_tasks = len(tasks)
        results = []
        first_exception = None
        pbm = ProgressBarManager(progress_bar_desc, show_progress)

        if not batch_size:
            with pbm.create_single_bar(total_tasks) as pbar:
                async for result in process_futures(
                    as_completed(tasks, max_workers, cancel_check=cancel_check)
                ):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Task failed with {type(result).__name__}: {result}",
                            exc_info=False,
                        )
                        # Store first exception to raise after all tasks complete
                        if first_exception is None:
                            first_exception = result
                    results.append(result)
                    pbar.update(1)
        else:
            total_tasks = len(tasks)
            batches = batched(tasks, batch_size)
            overall_pbar, batch_pbar, n_batches = pbm.create_nested_bars(
                total_tasks, batch_size
            )
            with overall_pbar, batch_pbar:
                for i, batch in enumerate(batches, 1):
                    pbm.update_batch_bar(batch_pbar, i, n_batches, len(batch))
                    async for result in process_futures(
                        as_completed(batch, max_workers, cancel_check=cancel_check)
                    ):
                        if isinstance(result, Exception):
                            logger.error(
                                f"Task failed with {type(result).__name__}: {result}",
                                exc_info=False,
                            )
                            # Store first exception to raise after all tasks complete
                            if first_exception is None:
                                first_exception = result
                        results.append(result)
                        batch_pbar.update(1)
                    overall_pbar.update(len(batch))

        # Raise the first exception encountered to fail fast with clear error message
        if first_exception is not None:
            raise first_exception

        return results

    return run(_run)
