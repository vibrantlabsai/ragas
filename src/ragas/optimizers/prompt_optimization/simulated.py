"""Deterministic stand-ins for the Anthropic API, for ``--dry-run`` and tests.

``SimulatedAnthropicClient`` mimics ``anthropic.Anthropic().messages.create``
so the real :class:`RagasAnthropicEngine` (request mapping, cache
breakpoints, budget tracking) runs unchanged, without network or cost.

Two responders decide what the "model" says:

- :class:`SimulatedStudent` answers RAG questions. In ``"sensitive"`` mode it
  behaves better when the prompt asks for it (concise answers, refusing when
  the documents lack the answer, or golden demos that show both), so the
  optimizer has something real to find. ``"insensitive"`` ignores the prompt
  (a negative control). ``"recall_dropping"`` gets shorter but loses facts
  when asked to be concise (a floor control).
- :class:`SimulatedProposer` fills any DSPy signature's output fields, and
  writes instruction candidates when asked for ``proposed_instruction``.
"""

from __future__ import annotations

import itertools
import re
import threading
import typing as t
from types import SimpleNamespace

from ragas.optimizers.prompt_optimization.bundle import GoldenRow

REFUSAL = "I don't know based on the provided documents."
_FIELD_RE = re.compile(
    r"\[\[ ## (\w+) ## \]\]\s*(.*?)(?=\[\[ ## \w+ ## \]\]|\Z)", re.DOTALL
)
_OUTPUT_FIELDS_RE = re.compile(r"Your output fields are:\s*(.*?)(?:\n\n|\Z)", re.DOTALL)
_FIELD_NAME_RE = re.compile(r"`(\w+)`")

Responder = t.Callable[[t.Dict[str, t.Any]], str]

DEFAULT_INSTRUCTION_CANDIDATES = (
    "Answer the question using only the documents. Be concise: reply with the answer only.",
    "Answer directly and briefly from the documents only. If the documents do not "
    "contain the answer, reply exactly: I don't know based on the provided documents.",
    "You are a helpful HR assistant. Read the documents and answer the question.",
    "Use only the provided documents. If the answer is missing, say you don't know.",
)


def _text(blocks: t.Any) -> str:
    if isinstance(blocks, str):
        return blocks
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def _fields(text: str) -> t.Dict[str, str]:
    """Field values from a ChatAdapter message, without DSPy's trailing directions."""
    values = {}
    for name, value in _FIELD_RE.findall(text):
        # DSPy appends "Respond with the corresponding output fields..." to the
        # last input field of the live user turn.
        values[name] = value.split("\n\nRespond with")[0].strip()
    return values


def format_fields(values: t.Mapping[str, str]) -> str:
    body = "\n\n".join(f"[[ ## {k} ## ]]\n{v}" for k, v in values.items())
    return f"{body}\n\n[[ ## completed ## ]]"


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class SimulatedAnthropicClient:
    """Implements ``messages.create`` with a responder function; records requests."""

    def __init__(self, responder: Responder, model_name: str = "simulated"):
        self.responder = responder
        self.model_name = model_name
        self.requests: t.List[t.Dict[str, t.Any]] = []
        self._lock = threading.Lock()
        self._seen_prefixes: t.Set[str] = set()
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **params: t.Any) -> t.Any:
        with self._lock:
            self.requests.append(params)
        text = self.responder(params)
        system = _text(params.get("system", []))
        prompt_text = system + "".join(_text(m["content"]) for m in params["messages"])
        # Simulate caching: the system prefix is "cached" from its second use.
        with self._lock:
            cached = system in self._seen_prefixes
            self._seen_prefixes.add(system)
        system_tokens = _approx_tokens(system) if system else 0
        usage = SimpleNamespace(
            input_tokens=_approx_tokens(prompt_text) - (system_tokens if cached else 0),
            output_tokens=_approx_tokens(text),
            cache_read_input_tokens=system_tokens if cached else 0,
            cache_creation_input_tokens=0 if cached else system_tokens,
        )
        return SimpleNamespace(
            id=f"msg_sim_{len(self.requests)}",
            model=params.get("model", self.model_name),
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason="end_turn",
            usage=usage,
        )


class SimulatedStudent:
    """Answers golden-dataset questions with prompt-dependent quality."""

    VERBOSE_TEMPLATE = "Great question! {answer}. Let me know if there is anything else I can help with."
    HALLUCINATION = (
        "Yes, Orbitline offers this to all employees as part of its benefits package."
    )

    def __init__(self, rows: t.Sequence[GoldenRow], behavior: str = "sensitive"):
        if behavior not in ("sensitive", "insensitive", "recall_dropping"):
            raise ValueError(f"unknown behavior {behavior!r}")
        self.behavior = behavior
        self.by_question = {r.question.strip(): r for r in rows}

    def __call__(self, params: t.Dict[str, t.Any]) -> str:
        system = _text(params.get("system", [])).lower()
        messages = params["messages"]
        live = _fields(_text(messages[-1]["content"]))
        row = self.by_question.get(live.get("question", "").strip())
        if row is None:
            return format_fields({"answer": "I am not sure."})
        demo_answers = [
            _text(m["content"]) for m in messages[:-1] if m["role"] == "assistant"
        ]
        return format_fields({"answer": self.answer(row, system, demo_answers)})

    def answer(self, row: GoldenRow, system: str, demo_answers: t.Sequence[str]) -> str:
        if self.behavior == "insensitive":
            return (
                self.VERBOSE_TEMPLATE.format(answer=row.reference.rstrip("."))
                if row.answerable
                else self.HALLUCINATION
            )

        concise = any(
            k in system
            for k in ("concise", "briefly", "brief", "answer only", "directly")
        )
        refuses = (
            "don't know" in system
            or "do not know" in system
            or any("don't know" in d.lower() for d in demo_answers)
        )
        shown_style = bool(demo_answers)

        if not row.answerable:
            return REFUSAL if refuses else self.HALLUCINATION
        if self.behavior == "recall_dropping" and (concise or shown_style):
            # Shorter, but drops facts: high precision, low recall.
            words = row.reference.split()
            return " ".join(words[: max(1, len(words) // 3)])
        if concise or shown_style:
            return row.reference
        return self.VERBOSE_TEMPLATE.format(answer=row.reference.rstrip("."))


class SimulatedProposer:
    """Fills every output field of whatever DSPy signature it is asked about."""

    def __init__(self, instructions: t.Sequence[str] = DEFAULT_INSTRUCTION_CANDIDATES):
        self._cycle = itertools.cycle(list(instructions))
        self._lock = threading.Lock()

    def __call__(self, params: t.Dict[str, t.Any]) -> str:
        system = _text(params.get("system", []))
        match = _OUTPUT_FIELDS_RE.search(system)
        names = _FIELD_NAME_RE.findall(match.group(1)) if match else ["answer"]
        values = {}
        for name in names:
            if "instruction" in name:
                with self._lock:
                    values[name] = next(self._cycle)
            else:
                values[name] = (
                    "Short factual questions about company HR policies, answered from documents."
                )
        return format_fields(values)
