"""Fixtures for the prompt_optimization tests. No network is used."""

from __future__ import annotations

import os
import sys
import typing as t
from pathlib import Path

import pytest

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
sys.path.insert(0, str(Path(__file__).parent))

from prompt_opt_testdata import StubAnswerer, make_rows  # noqa: E402

from ragas.optimizers.prompt_optimization.bundle import GoldenRow, PromptBundle  # noqa: E402


@pytest.fixture
def golden_rows() -> t.List[GoldenRow]:
    return make_rows()


@pytest.fixture
def splits(golden_rows: t.List[GoldenRow]) -> t.Dict[str, t.List[GoldenRow]]:
    from ragas.optimizers.prompt_optimization.dataset import split_rows

    return split_rows(golden_rows)


@pytest.fixture
def baseline_bundle() -> PromptBundle:
    return PromptBundle(
        instruction="You are a friendly assistant. Be thorough and add extra details.",
        source="baseline",
    )


@pytest.fixture
def stub_answerer() -> StubAnswerer:
    return StubAnswerer()
