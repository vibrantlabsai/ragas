import asyncio
import importlib
import logging
import sys
from typing import Callable, List

import pytest
from numpy.random import Generator, default_rng

from ragas.run_config import RunConfig, add_async_retry, add_retry

# Use a simple type alias that works across Python versions
RandomComparison = Callable[[Generator, Generator], bool]


@pytest.fixture(scope="function")
def compare_rng() -> Callable[[Generator, Generator], bool]:
    """Pytest fixture wrapper to check :py:cls:`numpy.random.Generator` object equivalence."""

    def _compare_rng(rng_0: Generator, rng_1: Generator) -> bool:
        """Compare two :py:cls:`numpy.random.Generator`object.

        Args:
            rng_0 (numpy.random.Generator) : The first generator to compare with.
            rng_1 (numpy.random.Generator) : The second generator to compare with.

        Returns:
            bool: Whether the two generators are at the same state.

        """
        return rng_0.random() == rng_1.random()

    return _compare_rng


@pytest.mark.parametrize(
    "seed, expected_equivalence",
    (
        [42, True],
        [None, False],
    ),
)
def test_random_num_generator(
    seed, compare_rng: RandomComparison, expected_equivalence
):
    """Check :py:mod:`numpy.random` functionality and seed behaviour control."""
    rc = RunConfig(seed=seed)

    # Check type
    assert isinstance(rc.rng, Generator)

    # Check generated value
    rng = default_rng(seed=seed)
    assert compare_rng(rc.rng, rng) == expected_equivalence

    # Check generation consistency
    importlib.reload(sys.modules["numpy.random"])
    new_rc = RunConfig(seed=seed)
    new_rng = default_rng(seed=seed)

    # Put generator into the same state
    new_rc.rng.random()
    new_rng.random()

    # Check equivalence
    if expected_equivalence:
        assert all(list(map(compare_rng, [rc.rng, new_rc.rng], [new_rng, rng])))
    else:
        assert all(
            list(
                map(
                    lambda x, y: not compare_rng(x, y),
                    [rc.rng, new_rc.rng],
                    [new_rng, rng],
                )
            )
        )


class _RecordingHandler(logging.Handler):
    """Collect the messages tenacity writes to a logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _flaky_step() -> None:
    raise ValueError("boom")


async def _flaky_step_async() -> None:
    raise ValueError("boom")


def _collect_sync_retry_logs(run_config: RunConfig) -> List[str]:
    logger = logging.getLogger(f"ragas.retry.{_flaky_step.__name__}")
    handler = _RecordingHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    propagate = logger.propagate
    logger.propagate = False
    try:
        with pytest.raises(ValueError):
            add_retry(_flaky_step, run_config)()
    finally:
        logger.removeHandler(handler)
        logger.propagate = propagate
    return handler.messages


def _collect_async_retry_logs(run_config: RunConfig) -> List[str]:
    logger = logging.getLogger(f"TENACITYRetry[{_flaky_step_async.__name__}]")
    handler = _RecordingHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    propagate = logger.propagate
    logger.propagate = False
    try:
        with pytest.raises(ValueError):
            asyncio.run(add_async_retry(_flaky_step_async, run_config)())
    finally:
        logger.removeHandler(handler)
        logger.propagate = propagate
    return handler.messages


@pytest.mark.parametrize("log_tenacity", [False, True])
def test_log_tenacity_controls_retry_logging(log_tenacity: bool):
    """log_tenacity decides whether retry attempts are logged at all.

    ``log_tenacity=False`` is the documented default and must keep the retry
    loggers silent; ``log_tenacity=True`` opts into one record per attempt.
    """
    run_config = RunConfig(max_retries=3, max_wait=0, log_tenacity=log_tenacity)
    expected = run_config.max_retries if log_tenacity else 0

    assert len(_collect_sync_retry_logs(run_config)) == expected
    assert len(_collect_async_retry_logs(run_config)) == expected
