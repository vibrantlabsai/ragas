from __future__ import annotations

import logging
import threading
import typing as t
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from ragas.async_utils import as_completed, process_futures, run
from ragas.run_config import RunConfig
from ragas.utils import ProgressBarManager, batched

logger = logging.getLogger(__name__)


@dataclass
class Executor:
    """
    Executor class for running asynchronous jobs with progress tracking and error handling.

    Attributes
    ----------
    desc : str
        Description for the progress bar
    show_progress : bool
        Whether to show the progress bar
    keep_progress_bar : bool
        Whether to keep the progress bar after completion
    jobs : List[Any]
        List of jobs to execute
    raise_exceptions : bool
        Whether to raise exceptions or log them
    batch_size : int
        Whether to batch (large) lists of tasks
    run_config : RunConfig
        Configuration for the run
    _nest_asyncio_applied : bool
        Whether nest_asyncio has been applied
    _cancel_event : threading.Event
        Event to signal cancellation
    """

    desc: str = "Evaluating"
    show_progress: bool = True
    keep_progress_bar: bool = True
    jobs: t.List[t.Any] = field(default_factory=list, repr=False)
    raise_exceptions: bool = False
    batch_size: t.Optional[int] = None
    run_config: t.Optional[RunConfig] = field(default=None, repr=False)
    pbar: t.Optional[tqdm] = None
    _jobs_processed: int = field(default=0, repr=False)
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def cancel(self) -> None:
        """Cancel the execution of all jobs."""
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        """Check if the execution has been cancelled."""
        return self._cancel_event.is_set()

    def wrap_callable_with_index(
        self, callable: t.Callable, counter: int
    ) -> t.Callable:
        async def wrapped_callable_async(*args, **kwargs) -> t.Tuple[int, t.Any]:
            try:
                result = await callable(*args, **kwargs)
                return counter, result
            except Exception as e:
                if self.raise_exceptions:
                    raise e
                else:
                    exec_name = type(e).__name__
                    exec_message = str(e)
                    logger.error(
                        "Exception raised in Job[%s]: %s(%s)",
                        counter,
                        exec_name,
                        exec_message,
                        exc_info=False,
                    )
                return counter, np.nan

        return wrapped_callable_async

    def submit(
        self,
        callable: t.Callable,
        *args,
        name: t.Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Submit a job to be executed, wrapping the callable with error handling and indexing to keep track of the job index.
        """
        # Use _jobs_processed for consistent indexing across multiple runs
        callable_with_index = self.wrap_callable_with_index(
            callable, self._jobs_processed
        )
        self.jobs.append((callable_with_index, args, kwargs, name))
        self._jobs_processed += 1

    def clear_jobs(self) -> None:
        """Clear all submitted jobs and reset counter."""
        self.jobs.clear()
        self._jobs_processed = 0

    async def _process_jobs(self) -> t.List[t.Any]:
        """Execute jobs with optional progress tracking."""
        if not self.jobs:
            return []

        # Make a copy of jobs to process and clear the original list to prevent re-execution
        jobs_to_process = self.jobs.copy()
        self.jobs.clear()

        max_workers = (
            self.run_config.max_workers
            if self.run_config and hasattr(self.run_config, "max_workers")
            else -1
        )
        results = []
        pbm = ProgressBarManager(self.desc, self.show_progress)

        if not self.batch_size:
            # Use external progress bar if provided, otherwise create one
            if self.pbar is None:
                with pbm.create_single_bar(len(jobs_to_process)) as internal_pbar:
                    await self._process_coroutines(
                        jobs_to_process, internal_pbar, results, max_workers
                    )
            else:
                await self._process_coroutines(
                    jobs_to_process, self.pbar, results, max_workers
                )
            return results

        # Process jobs in batches with nested progress bars
        await self._process_batched_jobs(jobs_to_process, pbm, max_workers, results)
        return results

    async def _process_batched_jobs(
        self, jobs_to_process, progress_manager, max_workers, results
    ):
        """Process jobs in batches with nested progress tracking."""
        batch_size = self.batch_size or len(jobs_to_process)
        batches = batched(jobs_to_process, batch_size)
        overall_pbar, batch_pbar, n_batches = progress_manager.create_nested_bars(
            len(jobs_to_process), batch_size
        )

        with overall_pbar, batch_pbar:
            for i, batch in enumerate(batches, 1):
                # Check for cancellation before processing each batch
                if self.is_cancelled():
                    break

                progress_manager.update_batch_bar(batch_pbar, i, n_batches, len(batch))

                # Create coroutines per batch
                coroutines = [
                    afunc(*args, **kwargs) for afunc, args, kwargs, _ in batch
                ]

                async for result in process_futures(
                    as_completed(
                        coroutines, max_workers, cancel_check=self.is_cancelled
                    )
                ):
                    # If jobs are configured to raise exceptions, propagate immediately
                    if isinstance(result, Exception) and self.raise_exceptions:
                        raise result
                    results.append(result)
                    batch_pbar.update(1)
                # Update overall progress bar for all futures in this batch
                overall_pbar.update(len(batch))

    async def _process_coroutines(self, jobs, pbar, results, max_workers):
        """Helper function to process coroutines and update the progress bar."""
        coroutines = [afunc(*args, **kwargs) for afunc, args, kwargs, _ in jobs]

        async for result in process_futures(
            as_completed(coroutines, max_workers, cancel_check=self.is_cancelled)
        ):
            # If jobs are configured to raise exceptions, propagate immediately
            if isinstance(result, Exception) and self.raise_exceptions:
                raise result
            results.append(result)
            pbar.update(1)

    async def aresults(self) -> t.List[t.Any]:
        """
        Execute all submitted jobs and return their results asynchronously.
        The results are returned in the order of job submission.

        This is the async entry point for executing async jobs when already in an async context.
        """
        results = await self._process_jobs()
        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]

    def results(self) -> t.List[t.Any]:
        """
        Execute all submitted jobs and return their results. The results are returned in the order of job submission.

        This is the main sync entry point for executing async jobs.
        """

        async def _async_wrapper():
            return await self.aresults()

        return run(_async_wrapper)


def run_async_batch(
    desc: str,
    func: t.Callable,
    kwargs_list: t.List[t.Dict],
    batch_size: t.Optional[int] = None,
):
    """
    Provide functionality to run the same async function with different arguments in parallel.
    """
    run_config = RunConfig()
    executor = Executor(
        desc=desc,
        keep_progress_bar=False,
        raise_exceptions=True,
        run_config=run_config,
        batch_size=batch_size,
    )

    for kwargs in kwargs_list:
        executor.submit(func, **kwargs)

    return executor.results()
