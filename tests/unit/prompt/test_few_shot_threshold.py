"""Regression tests for FewShotPydanticPrompt.threshold_for_examples forwarding.

Before the fix, line 120 in few_shot_pydantic_prompt.py called
  self.example_store.get_examples(data, self.top_k_for_examples)
without the threshold argument, so InMemoryExampleStore always applied its own
default (0.7) regardless of what was set on the prompt.  Issue #3056.
"""
from __future__ import annotations

import asyncio
import math

import pytest
from pydantic import BaseModel

from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.prompt.few_shot_pydantic_prompt import FewShotPydanticPrompt, InMemoryExampleStore
from ragas.prompt.pydantic_prompt import PydanticPrompt


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

class _QIn(BaseModel):
    question: str


class _QOut(BaseModel):
    answer: str


class _BaseQ(PydanticPrompt[_QIn, _QOut]):
    instruction = "answer the question"
    input_model = _QIn
    output_model = _QOut
    examples = [
        (_QIn(question="alpha"), _QOut(answer="A")),
        (_QIn(question="beta"), _QOut(answer="B")),
    ]


class _FakeEmb(BaseRagasEmbeddings):
    """Deterministic 2-D embeddings: cos(query, alpha)=0.5, cos(query, beta)=0.866."""

    def _emb(self, text: str):
        if "alpha" in text:
            return [1.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0]
        if "query" in text:
            return [0.5, math.sqrt(3) / 2]
        return [1.0, 0.0]

    def embed_query(self, text: str):
        return self._emb(text)

    def embed_documents(self, texts):
        return [self._emb(t) for t in texts]

    async def aembed_query(self, text: str):
        return self._emb(text)

    async def aembed_documents(self, texts):
        return [self._emb(t) for t in texts]


class _StubLLM:
    async def generate(self, *a, **k):
        raise RuntimeError("STOP-BEFORE-LLM")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prompt_with_spy():
    """Return (fs_prompt, calls_list) where calls_list records get_examples args."""
    calls = []
    _orig = InMemoryExampleStore.get_examples

    def spy(self, data, top_k=5, threshold=0.7):
        calls.append({"top_k": top_k, "threshold": threshold})
        return _orig(self, data, top_k, threshold)

    InMemoryExampleStore.get_examples = spy  # type: ignore[method-assign]
    fs = FewShotPydanticPrompt.from_pydantic_prompt(_BaseQ(), _FakeEmb())
    return fs, calls, _orig


def _restore(orig):
    InMemoryExampleStore.get_examples = orig  # type: ignore[method-assign]


async def _run(fs, threshold):
    fs.threshold_for_examples = threshold
    try:
        await fs.generate_multiple(_StubLLM(), _QIn(question="query"))  # type: ignore[arg-type]
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_threshold_forwarded_to_store():
    """generate_multiple must pass threshold_for_examples to get_examples (fixes #3056)."""
    fs, calls, orig = _make_prompt_with_spy()
    try:
        asyncio.run(_run(fs, 0.4))
    finally:
        _restore(orig)
    assert calls, "get_examples was never called"
    assert calls[0]["threshold"] == pytest.approx(0.4), (
        f"Expected threshold=0.4 forwarded to store, got {calls[0]['threshold']}"
    )


def test_low_threshold_includes_more_examples():
    """With threshold=0.4, both examples (sim 0.5 and 0.866) should be selected."""
    fs, calls, orig = _make_prompt_with_spy()
    try:
        asyncio.run(_run(fs, 0.4))
    finally:
        _restore(orig)
    selected = [q.question for q, _ in fs.examples]
    assert "alpha" in selected, (
        "alpha (sim=0.5) should be included when threshold=0.4"
    )
    assert "beta" in selected, (
        "beta (sim=0.866) should be included when threshold=0.4"
    )


def test_high_threshold_excludes_low_similarity_examples():
    """With threshold=0.8, only the high-similarity example (beta, 0.866) is selected."""
    fs, calls, orig = _make_prompt_with_spy()
    try:
        asyncio.run(_run(fs, 0.8))
    finally:
        _restore(orig)
    selected = [q.question for q, _ in fs.examples]
    assert "alpha" not in selected, (
        "alpha (sim=0.5) should be excluded when threshold=0.8"
    )
    assert "beta" in selected, (
        "beta (sim=0.866) should be included when threshold=0.8"
    )


def test_default_threshold_matches_store_default():
    """The default threshold_for_examples=0.7 should match the store's own default."""
    fs = FewShotPydanticPrompt.from_pydantic_prompt(_BaseQ(), _FakeEmb())
    assert fs.threshold_for_examples == pytest.approx(0.7)
