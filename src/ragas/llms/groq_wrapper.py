from __future__ import annotations

import asyncio
import logging
import re
import typing as t

from langchain_core.outputs import Generation, LLMResult

from ragas.llms.base import BaseRagasLLM

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from langchain_core.prompt_values import PromptValue

logger = logging.getLogger(__name__)

_DEFAULT_GROQ_RPM = 30


class GroqLLMWrapper(BaseRagasLLM):
    """Ragas LLM wrapper for Groq.

    Handles Groq's RPM rate limits and JSON responses occasionally
    wrapped in markdown fences.

    Args:
        groq_llm: A LangChain-compatible Groq LLM instance.
        requests_per_minute: Groq RPM cap. Controls the async semaphore.
    """

    def __init__(self, groq_llm: t.Any, requests_per_minute: int = _DEFAULT_GROQ_RPM):
        super().__init__()
        self.groq_llm = groq_llm
        self._semaphore = asyncio.Semaphore(requests_per_minute)

    def _clean_json_response(self, text: str) -> str:
        """Strip markdown fences that Groq occasionally wraps around JSON."""
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        return match.group(1).strip() if match else text

    def is_finished(self, response: LLMResult) -> bool:
        for gen_list in response.generations:
            for gen in gen_list:
                info = getattr(gen, "generation_info", None) or {}
                reason = info.get("finish_reason", "stop")
                if reason not in ("stop", "end_turn", "eos_token"):
                    return False
        return True

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        results = []
        for _ in range(n):
            response = self.groq_llm.generate_prompt(
                prompts=[prompt],
                stop=stop,
                callbacks=callbacks,
            )
            for gen in response.generations[0]:
                gen.text = self._clean_json_response(gen.text)
            results.extend(response.generations[0])
        return LLMResult(generations=[results])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        results = []
        for _ in range(n):
            result = await self._agenerate_single(prompt, stop, callbacks)
            results.extend(result)
        return LLMResult(generations=[results])

    async def _agenerate_single(
        self,
        prompt: PromptValue,
        stop: t.Optional[t.List[str]],
        callbacks: Callbacks,
        _retries: int = 3,
    ) -> t.List[Generation]:
        """Single async call with semaphore and rate-limit retry."""
        async with self._semaphore:
            for attempt in range(_retries):
                try:
                    response = await self.groq_llm.agenerate_prompt(
                        prompts=[prompt],
                        stop=stop,
                        callbacks=callbacks,
                    )
                    gens = response.generations[0]
                    for gen in gens:
                        gen.text = self._clean_json_response(gen.text)
                    return gens
                except Exception as exc:
                    is_rate_limit = "429" in str(exc) or "rate_limit" in str(exc).lower()
                    if is_rate_limit and attempt < _retries - 1:
                        wait = 60 * (attempt + 1)
                        logger.warning(
                            "Groq rate limit hit, waiting %ss (attempt %d/%d)",
                            wait,
                            attempt + 1,
                            _retries,
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise
        return []

    def __repr__(self) -> str:
        return f"GroqLLMWrapper(model={getattr(self.groq_llm, 'model_name', '?')})"
