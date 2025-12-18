from __future__ import annotations

import itertools
import logging
import os
import random
import re
import string
import typing as t
import uuid
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import tiktoken
from datasets import Dataset
from rich.console import Console
from tqdm.auto import tqdm

if t.TYPE_CHECKING:
    from ragas.metrics.base import Metric
    from ragas.tokenizers import BaseTokenizer

DEBUG_ENV_VAR = "RAGAS_DEBUG"


@lru_cache(maxsize=1)
def get_cache_dir() -> str:
    "get cache location"
    DEFAULT_XDG_CACHE_HOME = "~/.cache"
    xdg_cache = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
    default_ragas_cache = os.path.join(xdg_cache, "ragas")
    return os.path.expanduser(os.getenv("RAGAS_CACHE_HOME", default_ragas_cache))


@lru_cache(maxsize=1)
def get_debug_mode() -> bool:
    if os.environ.get(DEBUG_ENV_VAR, str(False)).lower() == "true":
        return True
    else:
        return False


def safe_nanmean(arr: t.List[float]) -> float:
    if len(arr) == 0:
        return np.nan  # or some other value or behavior for empty arrays

    arr_numpy = np.asarray(arr)  # Ensure input is a numpy array

    if np.isnan(arr_numpy).all():
        return np.nan  # or some other value or behavior for all-NaN arrays

    return float(np.nanmean(arr_numpy))


def check_if_sum_is_close(
    values: t.List[float], close_to: float, num_places: int
) -> bool:
    multiplier = 10**num_places
    total = sum(int(round(v * multiplier)) for v in values)
    return total == int(round(close_to * multiplier))


def patch_logger(module: str, level: int):
    # enable debug logging
    patched_logger = logging.getLogger(module)
    patched_logger.setLevel(level=level)
    # Create a handler for the asyncio logger
    handler = logging.StreamHandler()  # or another type of Handler
    handler.setLevel(logging.DEBUG)
    # Optional: Set a formatter if you want a specific format for the logs
    formatter = logging.Formatter("[%(name)s.%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    # Add the handler to the asyncio logger
    patched_logger.addHandler(handler)
    # Set propagate to False if you don't want it to log to the root logger's handlers as well
    patched_logger.propagate = False


# Function to check if an element is NaN
def is_nan(x):
    try:
        return np.isnan(x)
    except TypeError:
        return False


def get_metric_language(metric: "Metric") -> str:
    from ragas.prompt import BasePrompt

    languags = [
        value.language
        for _, value in vars(metric).items()
        if isinstance(value, BasePrompt)
    ]
    return languags[0] if len(languags) > 0 else ""


class DeprecationHelper:
    """Helper class to handle deprecation warnings for exported classes."""

    def __init__(self, new_target: t.Type, deprecation_message: str):
        self.new_target = new_target
        self.deprecation_message = deprecation_message

    def _warn(self):
        warnings.warn(self.deprecation_message, DeprecationWarning, stacklevel=3)

    def __call__(self, *args, **kwargs):
        self._warn()
        return self.new_target(*args, **kwargs)

    def __getattr__(self, attr):
        self._warn()
        return getattr(self.new_target, attr)


def deprecated(
    since: str,
    *,
    removal: t.Optional[str] = None,
    alternative: t.Optional[str] = None,
    addendum: t.Optional[str] = None,
    pending: bool = False,
):
    """
    Decorator to mark functions or classes as deprecated.

    Args:
        since: str
             The release at which this API became deprecated.
        removal: str, optional
            The expected removal version. Cannot be used with pending=True.
            Must be specified with pending=False.
        alternative: str, optional
            The alternative API or function to be used instead
            of the deprecated function.
        addendum: str, optional
            Additional text appended directly to the final message.
        pending: bool
            Whether the deprecation version is already scheduled or not.
            Cannot be used with removal.


    Examples
    --------

        .. code-block:: python

            @deprecated("0.1", removal="0.2", alternative="some_new_function")
            def some_old_function():
                print("This is an old function.")

    """

    def deprecate(func: t.Callable):
        def emit_warning(*args, **kwargs):
            if pending and removal:
                raise ValueError(
                    "A pending deprecation cannot have a scheduled removal"
                )

            message = f"The function {func.__name__} was deprecated in {since},"

            if not pending:
                if removal:
                    message += f" and will be removed in the {removal} release."
                else:
                    raise ValueError(
                        "A non-pending deprecation must have a scheduled removal."
                    )
            else:
                message += " and will be removed in a future release."

            if alternative:
                message += f" Use {alternative} instead."

            if addendum:
                message += f" {addendum}"

            warnings.warn(message, stacklevel=2, category=DeprecationWarning)
            return func(*args, **kwargs)

        return emit_warning

    return deprecate


def get_or_init(
    dictionary: t.Dict[str, t.Any], key: str, default: t.Callable[[], t.Any]
) -> t.Any:
    _value = dictionary.get(key)
    value = _value if _value is not None else default()

    return value


def get_from_dict(data_dict: t.Dict, key: str, default=None) -> t.Any:
    keys = key.split(".")
    current = data_dict

    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default

    return current


REQUIRED_COLS_v1 = {
    "user_input": "question",
    "retrieved_contexts": "contexts",
    "response": "answer",
    "reference": "ground_truth",
}


def get_required_columns_v1(metric: Metric):
    required_cols = metric.required_columns.get("SINGLE_TURN", set())
    required_cols = [REQUIRED_COLS_v1.get(col) for col in required_cols]
    return [col for col in required_cols if col is not None]


def convert_row_v1_to_v2(row: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
    required_cols_v2 = {k: v for v, k in REQUIRED_COLS_v1.items()}
    return {required_cols_v2[k]: v for k, v in row.items() if k in required_cols_v2}


def convert_v1_to_v2_dataset(dataset: Dataset) -> Dataset:
    columns_map = {v: k for k, v in REQUIRED_COLS_v1.items() if v in dataset.features}
    return dataset.rename_columns(columns_map)


def convert_v2_to_v1_dataset(dataset: Dataset) -> Dataset:
    columns_map = {k: v for k, v in REQUIRED_COLS_v1.items() if k in dataset.features}
    return dataset.rename_columns(columns_map)


def camel_to_snake(name):
    """
    Convert a camelCase string to snake_case.
    eg: HaiThere -> hai_there
    """
    pattern = re.compile(r"(?<!^)(?=[A-Z])")
    return pattern.sub("_", name).lower()


def num_tokens_from_string(
    string: str,
    encoding_name: str = "cl100k_base",
    tokenizer: t.Optional["BaseTokenizer"] = None,
) -> int:
    """Returns the number of tokens in a text string.

    Parameters
    ----------
    string : str
        The text to count tokens for.
    encoding_name : str
        Tiktoken encoding name (ignored if tokenizer is provided).
    tokenizer : BaseTokenizer, optional
        A tokenizer instance. If provided, encoding_name is ignored.

    Returns
    -------
    int
        Number of tokens in the string.
    """
    if tokenizer is not None:
        return tokenizer.count_tokens(string)
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


def batched(iterable: t.Iterable, n: int) -> t.Iterator[t.Tuple]:
    """Batch data from the iterable into tuples of length n. The last batch may be shorter than n."""
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


class ProgressBarManager:
    """Manages progress bars for batch and non-batch execution."""

    def __init__(self, desc: str, show_progress: bool):
        self.desc = desc
        self.show_progress = show_progress

    def create_single_bar(self, total: int) -> tqdm:
        """Create a single progress bar for non-batch execution."""
        return tqdm(
            total=total,
            desc=self.desc,
            disable=not self.show_progress,
        )

    def create_nested_bars(self, total_jobs: int, batch_size: int):
        """Create nested progress bars for batch execution."""
        n_batches = (total_jobs + batch_size - 1) // batch_size

        overall_pbar = tqdm(
            total=total_jobs,
            desc=self.desc,
            disable=not self.show_progress,
            position=0,
            leave=True,
        )

        batch_pbar = tqdm(
            total=min(batch_size, total_jobs),
            desc=f"Batch 1/{n_batches}",
            disable=not self.show_progress,
            position=1,
            leave=False,
        )

        return overall_pbar, batch_pbar, n_batches

    def update_batch_bar(
        self, batch_pbar: tqdm, batch_num: int, n_batches: int, batch_size: int
    ):
        """Update batch progress bar for new batch."""
        batch_pbar.reset(total=batch_size)
        batch_pbar.set_description(f"Batch {batch_num}/{n_batches}")


_LOGGER_DATE_TIME = "%Y-%m-%d %H:%M:%S"


def set_logging_level(logger_name: str = __name__, level: int = logging.DEBUG):
    """
    Set the logging level for a logger. Useful for debugging.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    log_format = (
        "[%(local_time)s - (%(utc_time)s UTC)] "
        "[%(levelname)s] [%(name)s] "
        "[RagasID: %(ragas_id)s, App-Version: %(app_version)s] %(message)s"
    )

    # Create a formatter with the custom formatter
    formatter = _ContextualFormatter(log_format, datefmt=_LOGGER_DATE_TIME)

    # Create a console handler and set its level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Apply the formatter to the handler
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    return logger


class _ContextualFormatter(logging.Formatter):
    """
    Custom logging formatter that adds context to the log records.
    """

    def format(self, record):
        from ragas import __version__
        from ragas._analytics import get_userid

        # Add UTC time
        record.utc_time = self.format_time(record, _LOGGER_DATE_TIME)
        # Add local time
        record.local_time = self.format_time(record, _LOGGER_DATE_TIME, local_time=True)
        # Add additional context
        record.ragas_id = get_userid()
        record.app_version = __version__
        return super().format(record)

    def format_time(self, record, datefmt=None, local_time=False):
        dt = (
            self.utc_converter(record.created)
            if not local_time
            else datetime.fromtimestamp(record.created)
        )
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

    @staticmethod
    def utc_converter(timestamp):
        return datetime.utcfromtimestamp(timestamp)  # UTC time conversion


base_logger = set_logging_level()

# Rich console instance for CLI and other formatting needs
console = Console()


class MemorableNames:
    """Generator for memorable, unique names for experiments and datasets."""

    def __init__(self):
        # List of adjectives (similar to what Docker uses)
        self.adjectives = [
            "admiring",
            "adoring",
            "affectionate",
            "agitated",
            "amazing",
            "angry",
            "awesome",
            "blissful",
            "bold",
            "boring",
            "brave",
            "busy",
            "charming",
            "clever",
            "cool",
            "compassionate",
            "competent",
            "condescending",
            "confident",
            "cranky",
            "crazy",
            "dazzling",
            "determined",
            "distracted",
            "dreamy",
            "eager",
            "ecstatic",
            "elastic",
            "elated",
            "elegant",
            "eloquent",
            "epic",
            "fervent",
            "festive",
            "flamboyant",
            "focused",
            "friendly",
            "frosty",
            "gallant",
            "gifted",
            "goofy",
            "gracious",
            "happy",
            "hardcore",
            "heuristic",
            "hopeful",
            "hungry",
            "infallible",
            "inspiring",
            "jolly",
            "jovial",
            "keen",
            "kind",
            "laughing",
            "loving",
            "lucid",
            "magical",
            "mystifying",
            "modest",
            "musing",
            "naughty",
            "nervous",
            "nifty",
            "nostalgic",
            "objective",
            "optimistic",
            "peaceful",
            "pedantic",
            "pensive",
            "practical",
            "priceless",
            "quirky",
            "quizzical",
            "relaxed",
            "reverent",
            "romantic",
            "sad",
            "serene",
            "sharp",
            "silly",
            "sleepy",
            "stoic",
            "stupefied",
            "suspicious",
            "sweet",
            "tender",
            "thirsty",
            "trusting",
            "upbeat",
            "vibrant",
            "vigilant",
            "vigorous",
            "wizardly",
            "wonderful",
            "xenodochial",
            "youthful",
            "zealous",
            "zen",
        ]

        # List of influential computer scientists and tech entrepreneurs
        self.scientists = [
            "turing",
            "hopper",
            "knuth",
            "torvalds",
            "ritchie",
            "thompson",
            "dijkstra",
            "kay",
            "wozniak",
            "gates",
            "jobs",
            "musk",
            "bezos",
            "lovelace",
            "berners_lee",
            "cerf",
            "gosling",
            "kernighan",
            "lamport",
            "mccarthy",
            "minsky",
            "rossum",
            "backus",
            "engelbart",
            "hamilton",
            "chomsky",
            "shannon",
            "zuckerberg",
            "page",
            "brin",
            "matsumoto",
            "stallman",
            "stroustrup",
            "cook",
            "neumann",
            "babbage",
            "tanenbaum",
            "rivest",
            "shamir",
            "adleman",
            "carmack",
            "andreessen",
            "ullman",
            "postel",
            "huffman",
            "boole",
            "curry",
            "liskov",
            "wing",
            "goldwasser",
            "hoare",
            "milner",
            "perlis",
            "sutherland",
            "tarjan",
            "valiant",
            "yao",
            "hopcroft",
            "naur",
            "wilkes",
            "codd",
            "diffie",
            "hellman",
            "pearl",
            "thiel",
            "narayen",
            "nadella",
            "pichai",
            "dorsey",
        ]

        self.used_names = set()

    def generate_name(self):
        """Generate a single memorable name."""
        adjective = random.choice(self.adjectives)
        scientist = random.choice(self.scientists)
        return f"{adjective}_{scientist}"

    def generate_unique_name(self):
        """Generate a unique memorable name."""
        attempts = 0
        max_attempts = 100  # Prevent infinite loops

        while attempts < max_attempts:
            name = self.generate_name()
            if name not in self.used_names:
                self.used_names.add(name)
                return name
            attempts += 1

        # If we exhaust our combinations, add a random suffix
        base_name = self.generate_name()
        unique_name = f"{base_name}_{random.randint(1000, 9999)}"
        self.used_names.add(unique_name)
        return unique_name

    def generate_unique_names(self, count):
        """Generate multiple unique memorable names."""
        return [self.generate_unique_name() for _ in range(count)]


# Global instance for easy access
memorable_names = MemorableNames()


def find_git_root(start_path: t.Union[str, Path, None] = None) -> Path:
    """Find the root directory of a git repository by traversing up from the start path."""
    # Start from the current directory if no path is provided
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path).resolve()

    # Check if the current directory is a git repository
    current_path = start_path
    while current_path != current_path.parent:  # Stop at filesystem root
        if (current_path / ".git").exists() and (current_path / ".git").is_dir():
            return current_path

        # Move up to the parent directory
        current_path = current_path.parent

    # Final check for the root directory
    if (current_path / ".git").exists() and (current_path / ".git").is_dir():
        return current_path

    # No git repository found
    raise ValueError(f"No git repository found in or above {start_path}")


def create_nano_id(size: int = 12) -> str:
    """Generate a short unique identifier."""
    # Define characters to use (alphanumeric)
    alphabet = string.ascii_letters + string.digits

    # Generate UUID and convert to int
    uuid_int = t.cast(int, uuid.uuid4().int)

    # Convert to base62
    result = ""
    while uuid_int:
        uuid_int, remainder = divmod(uuid_int, len(alphabet))
        result = alphabet[remainder] + result

    # Pad if necessary and return desired length
    return result[:size]


def async_to_sync(async_func):
    """Convert an async function to a sync function"""
    import asyncio
    import functools

    @functools.wraps(async_func)
    def sync_wrapper(*args, **kwargs):
        try:
            # Check if we're already in an event loop
            asyncio.get_running_loop()
            # If we get here, we're in a running loop
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, async_func(*args, **kwargs))
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(async_func(*args, **kwargs))

    return sync_wrapper


def get_test_directory():
    """Create a test directory that will be cleaned up on process exit.

    Returns:
        str: Path to test directory
    """
    import tempfile

    # Create a directory in the system temp directory
    test_dir = os.path.join(tempfile.gettempdir(), f"ragas_test_{create_nano_id()}")
    os.makedirs(test_dir, exist_ok=True)

    return test_dir
