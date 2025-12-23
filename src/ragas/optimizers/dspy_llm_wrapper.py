import typing as t

from ragas.llms.base import BaseRagasLLM


class RagasDSPyLM:
    """
    Wrapper to make Ragas LLM compatible with DSPy.

    DSPy expects LM objects to have specific methods for inference.
    This wrapper adapts Ragas LLM to work with DSPy's optimization framework.

    Parameters
    ----------
    ragas_llm : BaseRagasLLM
        The Ragas LLM instance to wrap.
    """

    def __init__(self, ragas_llm: BaseRagasLLM):
        self.ragas_llm = ragas_llm
        self.history: t.List[t.Dict[str, t.Any]] = []

    def __call__(
        self,
        prompt: t.Optional[str] = None,
        messages: t.Optional[t.List[t.Dict[str, str]]] = None,
        **kwargs: t.Any,
    ) -> t.List[str]:
        """
        Call the LLM with a prompt or messages.

        Parameters
        ----------
        prompt : str, optional
            Single prompt string.
        messages : List[Dict[str, str]], optional
            List of message dictionaries with 'role' and 'content'.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        List[str]
            List of completions.
        """
        import asyncio

        if prompt is not None:
            messages = [{"role": "user", "content": prompt}]
        elif messages is None:
            raise ValueError("Either prompt or messages must be provided")

        result = asyncio.run(self._generate(messages, **kwargs))
        return [result]

    async def _generate(
        self, messages: t.List[t.Dict[str, str]], **kwargs: t.Any
    ) -> str:
        """
        Generate completion using Ragas LLM.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of messages.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        str
            Generated completion.
        """
        from ragas.llms.prompt import PromptValue

        prompt_value = PromptValue(prompt_str="", messages=messages)

        result = await self.ragas_llm.generate(prompt_value)

        if hasattr(result, "generations") and result.generations:
            generation = result.generations[0][0]
            if hasattr(generation, "text"):
                return generation.text
            else:
                return str(generation)
        else:
            return str(result)

    def inspect_history(self, n: int = 1) -> t.List[t.Dict[str, t.Any]]:
        """
        Inspect recent history of LLM calls.

        Parameters
        ----------
        n : int
            Number of recent calls to return.

        Returns
        -------
        List[Dict[str, Any]]
            Recent call history.
        """
        return self.history[-n:]
