"""DSPy engine that sends DSPy's model calls through a ragas Anthropic LLM.

``RagasAnthropicEngine`` takes the ``InstructorLLM`` that
``ragas.llms.llm_factory(model, provider="anthropic", client=Anthropic())``
returns and reuses its model name, its provider-mapped parameters, and the
Anthropic client it wraps. Calls are made with the official Anthropic SDK
(``client.messages.create``), because the ragas ``generate`` method takes a
single prompt string and a response model, while DSPy needs multi-turn
few-shot messages and prompt-cache breakpoints.

Request layout (stable bytes first, volatile input last):

- ``system``: DSPy's system message (field structure + instruction), marked
  with ``cache_control`` (breakpoint 1).
- few-shot user/assistant turns; the last assistant turn is marked with
  ``cache_control`` (breakpoint 2).
- the live user turn (context + question), never cached.
"""

from __future__ import annotations

import logging
import typing as t

from ragas.optimizers.prompt_optimization.budget import BudgetTracker, Usage

logger = logging.getLogger(__name__)

CACHE_CONTROL = {"type": "ephemeral"}
#: Sampling params Claude 5.5 models reject unless they are the defaults.
_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")


def _has_messages_create(obj: t.Any) -> bool:
    messages = getattr(obj, "messages", None)
    return messages is not None and callable(getattr(messages, "create", None))


def unwrap_anthropic_client(client: t.Any) -> t.Any:
    """Return the raw Anthropic client behind an instructor-patched client.

    instructor's ``Instructor`` wrapper also exposes ``messages.create`` (a
    patched version that expects a response model), so unwrap ``.client``
    down to the innermost object that still has ``messages.create``.
    """
    candidate = client
    for _ in range(5):
        inner = getattr(candidate, "client", None)
        if inner is None or inner is candidate or not _has_messages_create(inner):
            break
        candidate = inner
    if _has_messages_create(candidate):
        return candidate
    raise TypeError(
        "Could not find an Anthropic client with messages.create(); pass a ragas LLM "
        "built with llm_factory(..., provider='anthropic', client=Anthropic()) or "
        "pass client= explicitly."
    )


def _text_of(parts: t.Iterable[t.Any]) -> str:
    return "".join(
        getattr(p, "text", "") for p in parts if getattr(p, "type", "") == "text"
    )


class RagasAnthropicEngine:
    """A DSPy ``Engine`` (``complete(Request) -> Response``) backed by Claude.

    Args:
        ragas_llm: An ``InstructorLLM`` for provider ``"anthropic"``. Its model,
            mapped params and client are reused.
        client: Optional raw ``anthropic.Anthropic`` client, overriding the one
            inside ``ragas_llm``.
        model: Optional model name, overriding ``ragas_llm.model``.
        budget: Spend tracker; checked before and charged after every call.
        effort: Claude effort level (``"low"``, ``"medium"``, ...), or None to
            use the model default.
        max_tokens: Output cap, used when DSPy does not set one.
        cache_breakpoints: Mark the instruction and the few-shot prefix for
            prompt caching.
    """

    def __init__(
        self,
        ragas_llm: t.Any = None,
        *,
        client: t.Any = None,
        model: t.Optional[str] = None,
        budget: t.Optional[BudgetTracker] = None,
        effort: t.Optional[str] = "low",
        max_tokens: int = 2048,
        cache_breakpoints: bool = True,
    ):
        base_params: t.Dict[str, t.Any] = {}
        if ragas_llm is not None:
            provider = str(getattr(ragas_llm, "provider", "anthropic")).lower()
            if provider != "anthropic":
                raise ValueError(
                    f"RagasAnthropicEngine needs an anthropic ragas LLM; got provider {provider!r}"
                )
            model = model or ragas_llm.model
            if client is None:
                client = unwrap_anthropic_client(ragas_llm.client)
            base_params = dict(ragas_llm._map_provider_params())
        if client is None or model is None:
            raise ValueError("pass ragas_llm, or both client and model")
        self.client = unwrap_anthropic_client(client)
        self.model = model
        self.budget = budget
        self.effort = effort
        self.cache_breakpoints = cache_breakpoints
        # Keep only sampling values the user set explicitly on the ragas LLM.
        self._sampling = {
            k: base_params[k] for k in _SAMPLING_PARAMS if k in base_params
        }
        self.max_tokens = max(int(base_params.get("max_tokens") or 0), max_tokens)
        self.calls = 0

    # -- request mapping -------------------------------------------------

    def build_params(self, request: t.Any) -> t.Dict[str, t.Any]:
        """Map a DSPy ``lm15.Request`` to Anthropic ``messages.create`` kwargs."""
        system = request.system
        if system is None:
            system_text = ""
        elif isinstance(system, str):
            system_text = system
        else:
            system_text = _text_of(system)

        messages: t.List[t.Dict[str, t.Any]] = []
        for message in request.messages:
            role = message.role
            if role == "developer":
                role = "user"
            if role not in ("user", "assistant"):
                raise ValueError(
                    f"unsupported message role for this engine: {message.role!r}"
                )
            text = _text_of(message.parts)
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

        if self.cache_breakpoints and len(messages) >= 2:
            # Breakpoint 2: the last assistant turn before the live user input,
            # i.e. the end of the few-shot demos.
            for i in range(len(messages) - 2, -1, -1):
                if messages[i]["role"] == "assistant":
                    messages[i]["content"][-1]["cache_control"] = dict(CACHE_CONTROL)
                    break

        params: t.Dict[str, t.Any] = {
            "model": self.model,
            "max_tokens": int(request.config.max_tokens or self.max_tokens),
            "messages": messages,
        }
        if system_text:
            block: t.Dict[str, t.Any] = {"type": "text", "text": system_text}
            if self.cache_breakpoints:
                block["cache_control"] = dict(CACHE_CONTROL)  # breakpoint 1
            params["system"] = [block]
        params.update(self._sampling)
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        return params

    # -- engine protocol -------------------------------------------------

    def complete(self, request: t.Any) -> t.Any:
        from dspy.clients.engines.base import validate_request
        from dspy.lm15 import Message, Response, TextPart, Usage as LMUsage

        validate_request(request)
        if self.budget is not None:
            self.budget.check()

        response = self.client.messages.create(**self.build_params(request))
        self.calls += 1

        raw_usage = getattr(response, "usage", None)
        usage = Usage(
            input_tokens=int(getattr(raw_usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(raw_usage, "output_tokens", 0) or 0),
            cache_read_input_tokens=int(
                getattr(raw_usage, "cache_read_input_tokens", 0) or 0
            ),
            cache_creation_input_tokens=int(
                getattr(raw_usage, "cache_creation_input_tokens", 0) or 0
            ),
        )
        if self.budget is not None:
            self.budget.record(self.model, usage)

        stop_reason = getattr(response, "stop_reason", None)
        text = _text_of(getattr(response, "content", []) or [])
        if stop_reason == "refusal":
            logger.warning("model %s refused a request", self.model)
            text, finish = "", "content_filter"
        elif stop_reason == "max_tokens":
            finish = "length"
        else:
            finish = "stop"

        return Response(
            id=getattr(response, "id", None) or None,
            model=getattr(response, "model", None) or self.model,
            message=Message.assistant([TextPart(text)]),
            finish_reason=finish,
            usage=LMUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_input_tokens,
                cache_write_tokens=usage.cache_creation_input_tokens,
            ),
        )

    def stream(self, request: t.Any) -> t.Any:
        from dspy.lm15 import response_to_events

        return response_to_events(self.complete(request))

    def close(self) -> None:
        pass


def make_dspy_lm(engine: t.Any, cache: bool = True, num_retries: int = 3) -> t.Any:
    """Wrap an engine as a ``dspy.LM``. DSPy's own disk cache makes reruns free."""
    import dspy

    return dspy.LM(
        f"anthropic/{engine.model}",
        engine=engine,
        cache=cache,
        num_retries=num_retries,
    )
