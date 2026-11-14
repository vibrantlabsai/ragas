"""Experiments hold the results of an experiment against a dataset."""

__all__ = [
    "Experiment",
    "ExperimentExecutionConfig",
    "ExperimentRunSummary",
    "experiment",
    "version_experiment",
]

import asyncio
import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

from ragas.async_utils import (
    FailurePolicy,
    RowExecutionError,
    run_indexed_batch,
    validate_batch_execution_options,
)
from ragas.backends.base import BaseBackend
from ragas.dataset import Dataset, DataTable
from ragas.utils import find_git_root, memorable_names

logger = logging.getLogger(__name__)


@dataclass
class ExperimentExecutionConfig:
    """Execution controls for running an experiment over a dataset.

    Parameters
    ----------
    max_concurrency : int, optional
        Maximum number of experiment rows to execute at the same time.
        ``None`` preserves the default unbounded behavior for the provided
        dataset size.
    failure_policy : {"skip", "record", "raise"}, default "skip"
        How row failures are handled. ``"skip"`` keeps the historical partial
        success behavior by omitting failed rows from saved results and storing
        structured errors on ``experiment.run_errors``. ``"record"`` keeps the
        same successful experiment rows while recording failures in
        ``experiment.run_errors`` and ``experiment.run_summary``. ``"raise"``
        stops execution and raises ``ragas.async_utils.BatchExecutionError``.
    preserve_order : bool, default False
        Whether successful rows are appended in original dataset order.
        ``False`` preserves the historical completion-order append behavior.
    """

    max_concurrency: t.Optional[int] = None
    failure_policy: FailurePolicy = "skip"
    preserve_order: bool = False

    def __post_init__(self):
        validate_batch_execution_options(self.max_concurrency, self.failure_policy)


@dataclass
class ExperimentRunSummary:
    """Non-persisted summary of an experiment run.

    Attributes
    ----------
    total_rows : int
        Number of dataset rows submitted to the experiment runner.
    succeeded : int
        Number of successful, non-``None`` results appended to the experiment.
    failed : int
        Number of rows that raised an exception during execution.
    skipped : int
        Number of failed rows omitted because ``failure_policy="skip"`` was
        used.
    failure_policy : {"skip", "record", "raise"}
        Failure policy used for the run.
    """

    total_rows: int
    succeeded: int
    failed: int
    skipped: int
    failure_policy: FailurePolicy


class Experiment(DataTable):
    """Experiment results stored in a ``DataTable``.

    Attributes
    ----------
    run_errors : List[RowExecutionError]
        Structured errors from the most recent in-memory run. These errors are
        not persisted to experiment backends.
    run_summary : ExperimentRunSummary, optional
        Summary of the most recent in-memory run. This metadata is not
        persisted to experiment backends.
    """

    DATATABLE_TYPE = "Experiment"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_errors: t.List[RowExecutionError] = []
        self.run_summary: t.Optional[ExperimentRunSummary] = None


def version_experiment(
    experiment_name: str,
    commit_message: t.Optional[str] = None,
    repo_path: t.Union[str, Path, None] = None,
    create_branch: bool = True,
    stage_all: bool = False,
) -> str:
    """Version control the current state of the codebase for an experiment.

    This function requires GitPython to be installed. You can install it with:
        pip install ragas[git]
        # or
        uv pip install ragas[git]

    Args:
        experiment_name: Name for the experiment (used in branch name)
        commit_message: Custom commit message (defaults to "Experiment: {experiment_name}")
        repo_path: Path to git repository (defaults to current directory)
        create_branch: Whether to create a git branch for the experiment
        stage_all: Whether to stage untracked files (default: tracked files only)

    Returns:
        The commit hash of the versioned state
    """
    try:
        import git
    except ImportError as e:
        raise ImportError(
            "version_experiment() requires GitPython. Install it with:\n"
            "  pip install ragas[git]\n"
            "  # or\n"
            "  uv pip install ragas[git]\n\n"
            "Or install with full features:\n"
            "  pip install ragas[all]\n"
            "  # or\n"
            "  uv pip install ragas[all]"
        ) from e

    # Default to current directory if no repo path is provided
    if repo_path is None:
        repo_path = find_git_root()

    # Initialize git repo object
    repo = git.Repo(repo_path)

    # Check if there are any changes to the repo
    has_changes = False
    if stage_all and repo.is_dirty(untracked_files=True):
        print("Staging all changes")
        repo.git.add(".")
        has_changes = True
    elif repo.is_dirty(untracked_files=False):
        print("Staging changes to tracked files")
        repo.git.add("-u")
        has_changes = True

    # Check if there are uncommitted changes
    if has_changes:
        # Default commit message if none provided
        if commit_message is None:
            commit_message = f"Experiment: {experiment_name}"

        # Commit changes
        commit = repo.index.commit(commit_message)
        commit_hash = commit.hexsha
        print(f"Changes committed with hash: {commit_hash[:8]}")
    else:
        # No changes to commit, use current HEAD
        commit_hash = repo.head.commit.hexsha
        print("No changes detected, nothing to commit")

    # Format the branch/tag name
    version_name = f"ragas/{experiment_name}"

    # Create branch if requested
    if create_branch:
        repo.create_head(version_name, commit_hash)
        print(f"Created branch: {version_name}")

    return commit_hash


@t.runtime_checkable
class ExperimentProtocol(t.Protocol):
    async def __call__(self, *args, **kwargs) -> t.Any: ...
    async def arun(
        self,
        dataset: Dataset,
        name: t.Optional[str] = None,
        backend: t.Optional[t.Union[BaseBackend, str]] = None,
        *args,
        execution_config: t.Optional[ExperimentExecutionConfig] = None,
        **kwargs,
    ) -> "Experiment": ...


class ExperimentWrapper:
    """Wrapper class that implements ExperimentProtocol for decorated functions."""

    def __init__(
        self,
        func: t.Callable,
        experiment_model: t.Optional[t.Type[BaseModel]] = None,
        default_backend: t.Optional[t.Union[BaseBackend, str]] = None,
        name_prefix: str = "",
    ):
        self.func = func
        self.experiment_model = experiment_model
        self.default_backend = default_backend
        self.name_prefix = name_prefix
        # Preserve function metadata
        self.__name__ = getattr(func, "__name__", "experiment_function")
        self.__doc__ = getattr(func, "__doc__", None)

    async def __call__(self, *args, **kwargs) -> t.Any:
        """Call the original function."""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)

    async def arun(
        self,
        dataset: Dataset,
        name: t.Optional[str] = None,
        backend: t.Optional[t.Union[BaseBackend, str]] = None,
        *args,
        execution_config: t.Optional[ExperimentExecutionConfig] = None,
        **kwargs,
    ) -> "Experiment":
        """Run the experiment against a dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset whose rows are passed to the decorated experiment function.
        name : str, optional
            Name for the saved experiment. If omitted, a generated name is used.
        backend : BaseBackend or str, optional
            Backend used to save the experiment results. Defaults to the
            decorator backend or the dataset backend.
        *args
            Positional arguments forwarded to the decorated experiment function
            after the dataset row.
        execution_config : ExperimentExecutionConfig, optional
            Optional execution controls for concurrency, failure handling, and
            result ordering.
        **kwargs
            Keyword arguments forwarded to the decorated experiment function.

        Returns
        -------
        Experiment
            Saved experiment results. ``run_errors`` and ``run_summary`` are
            attached to the returned object for the current run only.

        Notes
        -----
        The default execution config preserves historical behavior: rows run
        with unbounded concurrency, failed rows are skipped, and successful
        rows are appended in completion order.
        """
        execution_config = execution_config or ExperimentExecutionConfig()

        # Generate name if not provided
        if name is None:
            name = memorable_names.generate_unique_name()
        if self.name_prefix:
            name = f"{self.name_prefix}-{name}"

        # Resolve backend
        experiment_backend = backend or self.default_backend
        if experiment_backend:
            resolved_backend = Experiment._resolve_backend(experiment_backend)
        else:
            resolved_backend = dataset.backend

        # Create experiment
        experiment_view = Experiment(
            name=name,
            data_model=self.experiment_model,
            backend=resolved_backend,
        )

        dataset_items = list(dataset)

        progress_bar = None
        try:
            progress_bar = tqdm(total=len(dataset_items), desc="Running experiment")

            async def run_row(index: int, item: t.Any):
                return await self(item, *args, **kwargs)

            def update_progress(index: int) -> None:
                if progress_bar is not None:
                    progress_bar.update(1)

            batch_result = await run_indexed_batch(
                dataset_items,
                run_row,
                max_concurrency=execution_config.max_concurrency,
                failure_policy=execution_config.failure_policy,
                on_complete=update_progress,
                error_context=f"experiment {name}",
            )
        finally:
            if progress_bar:
                progress_bar.close()

        experiment_view.run_errors = batch_result.errors
        for error in batch_result.errors:
            logger.warning(
                "Experiment row %s failed with %s: %s",
                error.index,
                error.error_type,
                error.message,
            )

        successful_results = batch_result.successes
        if execution_config.preserve_order:
            successful_results = sorted(successful_results, key=lambda item: item[0])

        appended_count = 0
        for _, result in successful_results:
            if result is not None:
                experiment_view.append(result)
                appended_count += 1

        skipped_count = (
            len(batch_result.errors) if execution_config.failure_policy == "skip" else 0
        )
        experiment_view.run_summary = ExperimentRunSummary(
            total_rows=len(dataset_items),
            succeeded=appended_count,
            failed=len(batch_result.errors),
            skipped=skipped_count,
            failure_policy=execution_config.failure_policy,
        )

        # Save experiment
        experiment_view.save()

        return experiment_view


def experiment(
    experiment_model: t.Optional[t.Type[BaseModel]] = None,
    backend: t.Optional[t.Union[BaseBackend, str]] = None,
    name_prefix: str = "",
) -> t.Callable[[t.Callable], ExperimentProtocol]:
    """Decorator for creating experiment functions.

    Args:
        experiment_model: The Pydantic model type to use for experiment results
        backend: Optional backend to use for storing experiment results
        name_prefix: Optional prefix for experiment names

    Returns:
        Decorator function that wraps experiment functions

    Example:
        @experiment(ExperimentDataRow)
        async def run_experiment(row: TestDataRow):
            # experiment logic here
            return ExperimentDataRow(...)
    """

    def decorator(func: t.Callable) -> ExperimentProtocol:
        wrapper = ExperimentWrapper(
            func=func,
            experiment_model=experiment_model,
            default_backend=backend,
            name_prefix=name_prefix,
        )
        return t.cast(ExperimentProtocol, wrapper)

    return decorator
