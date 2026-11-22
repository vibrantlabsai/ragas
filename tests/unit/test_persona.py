"""Persona lookup tests — regression for issue #2047 (an LLM-reformatted persona
name raised KeyError and aborted test set generation)."""

from __future__ import annotations

import typing as t

import pytest

from ragas.testset.graph import Node
from ragas.testset.persona import Persona, PersonaList
from ragas.testset.synthesizers.multi_hop.base import MultiHopQuerySynthesizer
from ragas.testset.synthesizers.single_hop.base import SingleHopQuerySynthesizer


def _sample_personas() -> t.List[Persona]:
    return [
        Persona(name="First time flight taker", role_description="First-time flyer."),
        Persona(name="Frequent flier", role_description="Travels regularly."),
    ]


def test_getitem_returns_persona_on_exact_match():
    personas = PersonaList(personas=_sample_personas())
    assert personas["Frequent flier"].name == "Frequent flier"


def test_getitem_raises_keyerror_when_missing():
    personas = PersonaList(personas=_sample_personas())
    with pytest.raises(KeyError):
        personas["Nonexistent persona"]


def test_get_returns_default_when_missing():
    personas = PersonaList(personas=_sample_personas())
    assert personas.get("Nonexistent persona") is None
    fallback = _sample_personas()[0]
    assert personas.get("Nonexistent persona", fallback) is fallback


def test_get_is_case_and_whitespace_insensitive():
    personas = PersonaList(personas=_sample_personas())
    assert personas.get("frequent flier").name == "Frequent flier"
    assert (
        personas.get("  First   Time  Flight Taker ").name
        == "First time flight taker"
    )


def test_single_hop_prepare_combinations_skips_unknown_persona():
    synthesizer = SingleHopQuerySynthesizer(llm=object())  # llm unused in this path
    node = Node(properties={"page_content": "content"})
    combinations = synthesizer.prepare_combinations(
        node,
        terms=["airports"],
        personas=_sample_personas(),
        persona_concepts={"Completely Different Persona": ["airports"]},
    )
    assert combinations[0]["personas"] == []  # previously raised KeyError


def test_single_hop_prepare_combinations_matches_reformatted_name():
    synthesizer = SingleHopQuerySynthesizer(llm=object())
    node = Node(properties={"page_content": "content"})
    combinations = synthesizer.prepare_combinations(
        node,
        terms=["airports"],
        personas=_sample_personas(),
        persona_concepts={"first time flight taker": ["airports"]},  # re-cased key
    )
    matched = combinations[0]["personas"]
    assert len(matched) == 1
    assert matched[0].name == "First time flight taker"


def test_multi_hop_prepare_combinations_skips_unknown_persona():
    synthesizer = MultiHopQuerySynthesizer(llm=object())
    node = Node(properties={"entities": ["paris"]})
    combinations = synthesizer.prepare_combinations(
        nodes=[node],
        combinations=[["paris"]],
        personas=_sample_personas(),
        persona_item_mapping={"Completely Different Persona": ["paris"]},
        property_name="entities",
    )
    assert all(entry["personas"] == [] for entry in combinations)
