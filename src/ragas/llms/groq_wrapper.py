"""Groq LLM wrapper implementation for Ragas."""

import asyncio
import logging
import re
import threading
import typing as t

from langchain_core.callbacks import Callbacks
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue

from ragas._analytics import LLMUsageEvent, track
from ragas.cache import CacheInterface
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)


class GroqLLMWrapper(BaseRagasLLM):
    """
    Groq LLM wrapper for Ragas.

    This wrapper provides direct integration with Groq's API for fast inference
    using their optimized language models.
    """

    def __init__(
        self,
        groq_client: t.Any,
        model: str = "llama3-70b-8192",
        rpm_limit: int = 30,
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
    ) -> None:
        """
        Initialize Groq LLM wrapper.

        Args:
            groq_client: The Groq client instance for API calls
            model: The Groq model to use (default: "llama3-70b-8192")
            rpm_limit: Requests per minute limit for rate limiting (default: 30)
            run_config: Ragas run configuration
            cache: Optional cache backend for caching responses
        """
        # Initialize parent class first
        super().__init__(cache=cache)

        # Store parameters
        self.groq_client = groq_client
        self.model = model
        self.rpm_limit = rpm_limit

        # Initialize the semaphore for rate limiting
        self._semaphore = asyncio.Semaphore(rpm_limit)

        # Set run config
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

        # Track initialization
        track(
            LLMUsageEvent(
                provider="groq",
                model=model,
                llm_type="groq_wrapper",
                num_requests=1,
                is_async=False,
            )
        )

    def _extract_json(self, raw: str) -> str:
        """
        Extract JSON from a raw response string.

        Handles both fenced JSON blocks (```json ... ```) and naked JSON objects/arrays.
        If no JSON is found, returns the original string.

        Args:
            raw: The raw response string from the LLM

        Returns:
            Extracted JSON string or the original string if no JSON pattern matches
        """
        # Strip leading/trailing whitespace
        raw = raw.strip()

        # Try to extract from ```json ... ``` fenced block
        json_fence_pattern = r"```json\s*(.*?)\s*```"
        fence_match = re.search(json_fence_pattern, raw, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        # Try to find JSON array [...] or object {...}
        # Check for arrays first, then objects, to handle arrays of objects correctly
        json_arr_pattern = r"(\[[^\[\]]*(?:\{[^{}]*\}[^\[\]]*)*\])"
        arr_match = re.search(json_arr_pattern, raw, re.DOTALL)
        if arr_match:
            return arr_match.group(1).strip()

        # Match JSON objects
        json_obj_pattern = r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
        obj_match = re.search(json_obj_pattern, raw, re.DOTALL)
        if obj_match:
            return obj_match.group(1).strip()

        # No JSON pattern found, return original
        return raw

    async def _acall_once(self, prompt_text: str, temperature: float) -> str:
        """
        Make a single async call to Groq API with rate limiting.

        Args:
            prompt_text: The prompt text to send to the model
            temperature: Temperature parameter for generation

        Returns:
            Extracted JSON string from the response

        Raises:
            Exception: If the API call fails
        """
        async with self._semaphore:
            try:
                # Try to import Groq-specific exceptions
                try:
                    from groq import RateLimitError
                except ImportError:
                    RateLimitError = None  # type: ignore

                # Make the API call
                response = await self.groq_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=temperature,
                )

                # Extract content from the first choice
                content = response.choices[0].message.content

                # Extract and return JSON
                return self._extract_json(content)

            except Exception as e:
                # Check if it's a rate limit error
                if RateLimitError is not None and isinstance(e, RateLimitError):
                    logger.warning("Groq rate limit error", exc_info=e)
                    raise
                else:
                    logger.error("Groq API call failed", exc_info=e)
                    raise

    def _run_async_in_current_loop(self, coro: t.Awaitable[t.Any]) -> t.Any:
        """
        Run an async coroutine in the current event loop if possible.

        This handles Jupyter environments correctly by using a separate thread
        when a running event loop is detected.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()

            if loop.is_running():
                # If the loop is already running (like in Jupyter notebooks),
                # we run the coroutine in a separate thread with its own event loop
                result_container: t.Dict[str, t.Any] = {
                    "result": None,
                    "exception": None,
                }

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

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """
        Generate text using Groq API synchronously.

        Args:
            prompt: The prompt value to generate from
            n: Number of completions to generate (default: 1)
            temperature: Temperature for generation (default: 0.01)
            stop: Optional list of stop sequences
            callbacks: Optional callbacks for generation

        Returns:
            LLMResult containing the generated text
        """
        # Run async method in event loop
        if temperature is None:
            temperature = self.get_temperature(n)

        result = self._run_async_in_current_loop(
            self.agenerate_text(
                prompt=prompt,
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
        )

        # Track usage
        track(
            LLMUsageEvent(
                provider="groq",
                model=self.model,
                llm_type="groq_wrapper",
                num_requests=n,
                is_async=False,
            )
        )

        return result

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """
        Generate text using Groq API asynchronously.

        Args:
            prompt: The prompt value to generate from
            n: Number of completions to generate (default: 1)
            temperature: Temperature for generation (default: 0.01)
            stop: Optional list of stop sequences
            callbacks: Optional callbacks for generation

        Returns:
            LLMResult containing the generated text
        """
        # Determine effective temperature
        if temperature is None:
            temperature = self.get_temperature(n)

        # Log warning if n > 1 (only support n=1 for now)
        if n > 1:
            logger.warning(
                f"Groq wrapper currently only supports n=1, but n={n} was requested. "
                "Generating n completions sequentially."
            )

        # Generate prompt text
        prompt_text = prompt.to_string()

        # Generate n completions
        generations = []
        for _ in range(n):
            text = await self._acall_once(prompt_text, temperature)
            generation = Generation(text=text)
            generations.append(generation)

        # Track usage
        track(
            LLMUsageEvent(
                provider="groq",
                model=self.model,
                llm_type="groq_wrapper",
                num_requests=n,
                is_async=True,
            )
        )

        # Return as LLMResult with proper structure
        # Following the pattern from LangchainLLMWrapper: generations as [[g1, g2, ...]]
        return LLMResult(generations=[generations])

    def is_finished(self, response: LLMResult) -> bool:
        """
        Check if the LLM response is finished/complete.

        For Groq, we treat all responses as finished unless they are empty.

        Args:
            response: The LLM result to check

        Returns:
            True if the response is complete, False otherwise
        """
        return True

    def __repr__(self) -> str:
        """Return string representation of the wrapper."""
        return (
            f"{self.__class__.__name__}(model={self.model}, rpm_limit={self.rpm_limit})"
        )
