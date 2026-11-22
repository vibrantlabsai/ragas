"""Shared utilities for embedding implementations."""

import asyncio
import threading
import typing as t
from concurrent.futures import ThreadPoolExecutor


def run_async_in_current_loop(coro: t.Awaitable[t.Any]) -> t.Any:
    """Run an async coroutine in the current event loop if possible.

    This handles Jupyter environments correctly by using a separate thread
    when a running event loop is detected.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine

    Raises:
        Any exception raised by the coroutine
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()

        if loop.is_running():
            # If the loop is already running (like in Jupyter notebooks),
            # we run the coroutine in a separate thread with its own event loop
            result_container: t.Dict[str, t.Any] = {"result": None, "exception": None}

            def run_in_thread():
                # Create a new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    # Run the coroutine in this thread's event loop
                    result_container["result"] = new_loop.run_until_complete(coro)
                except Exception as e:
                    # Capture any exceptions to re-raise in the main thread
                    result_container["exception"] = e
                finally:
                    # Clean up the event loop
                    new_loop.close()

            # Start the thread and wait for it to complete
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()

            # Re-raise any exceptions that occurred in the thread
            if result_container["exception"]:
                raise result_container["exception"]

            return result_container["result"]
        else:
            # Standard case - event loop exists but isn't running
            return loop.run_until_complete(coro)

    except RuntimeError:
        # If we get a runtime error about no event loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # Clean up
            loop.close()
            asyncio.set_event_loop(None)


async def run_sync_in_async(func: t.Callable, *args, **kwargs) -> t.Any:
    """Run a sync function in an async context using ThreadPoolExecutor.

    Args:
        func: The sync function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function
    """
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))


def batch_texts(texts: t.List[str], batch_size: int) -> t.List[t.List[str]]:
    """Batch a list of texts into smaller chunks.

    Args:
        texts: List of texts to batch
        batch_size: Size of each batch

    Returns:
        List of batches, where each batch is a list of texts
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    batches = []
    for i in range(0, len(texts), batch_size):
        batches.append(texts[i : i + batch_size])
    return batches


def get_optimal_batch_size(provider: str, model: str) -> int:
    """Get optimal batch size for a provider/model combination.

    Args:
        provider: The embedding provider
        model: The model name

    Returns:
        Optimal batch size for the provider/model
    """
    provider_lower = provider.lower()

    # Provider-specific batch sizes
    if "openai" in provider_lower:
        return 100  # OpenAI supports large batches
    elif "cohere" in provider_lower:
        return 96  # Cohere's documented limit
    elif "google" in provider_lower or "vertex" in provider_lower:
        return 5  # Google/Vertex AI is more conservative
    elif "huggingface" in provider_lower:
        return 32  # HuggingFace default
    else:
        return 10  # Conservative default for unknown providers


def validate_texts(texts: t.Union[str, t.List[str]]) -> t.List[str]:
    """Validate and normalize text inputs.

    Args:
        texts: Single text or list of texts

    Returns:
        List of validated texts

    Raises:
        ValueError: If texts are invalid
    """
    if isinstance(texts, str):
        texts = [texts]

    if not isinstance(texts, list):
        raise ValueError("Texts must be a string or list of strings")

    if not texts:
        raise ValueError("Texts list cannot be empty")

    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"Text at index {i} must be a string, got {type(text)}")
        if not text.strip():
            raise ValueError(f"Text at index {i} cannot be empty or whitespace only")

    return texts


def safe_import(module_name: str, package_name: t.Optional[str] = None) -> t.Any:
    """Safely import a module with helpful error message.

    Args:
        module_name: Name of the module to import
        package_name: Optional package name for better error messages

    Returns:
        The imported module

    Raises:
        ImportError: If the module cannot be imported
    """
    try:
        return __import__(module_name, fromlist=[""])
    except ImportError as e:
        package_name = package_name or module_name
        raise ImportError(
            f"Failed to import {module_name}. "
            f"Please install the required package: pip install {package_name}"
        ) from e
