import typing as t

import pytest

from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.prompts import PersonaThemesMapping, ThemesPersonasInput
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)


class MockThemePersonaMatchingPrompt(PydanticPrompt):
    async def generate(self, data: ThemesPersonasInput, llm, callbacks=None):
        themes: t.List[str] = data.themes
        personas: t.List[Persona] = data.personas
        return PersonaThemesMapping(
            mapping={persona.name: themes for persona in personas}
        )


def test_extract_themes_from_items_with_strings(fake_llm):
    """Test _extract_themes_from_items with string input."""
    synthesizer = SingleHopSpecificQuerySynthesizer(llm=fake_llm)

    items = ["Theme1", "Theme2", "Theme3"]
    themes = synthesizer._extract_themes_from_items(items)

    assert set(themes) == {"Theme1", "Theme2", "Theme3"}


def test_extract_themes_from_items_with_tuples(fake_llm):
    """Test _extract_themes_from_items with tuple input (the bug fix)."""
    synthesizer = SingleHopSpecificQuerySynthesizer(llm=fake_llm)

    # This is the format that was causing the ValidationError in issue #2368
    items = [("Entity1", "Entity1"), ("Entity2", "Entity2")]
    themes = synthesizer._extract_themes_from_items(items)

    assert set(themes) == {"Entity1", "Entity2"}


def test_extract_themes_from_items_with_mixed_formats(fake_llm):
    """Test _extract_themes_from_items with mixed formats."""
    synthesizer = SingleHopSpecificQuerySynthesizer(llm=fake_llm)

    items = ["Theme1", ("Entity2", "Entity2"), ["Entity3", "Entity3"]]
    themes = synthesizer._extract_themes_from_items(items)

    assert set(themes) == {"Theme1", "Entity2", "Entity3"}


def test_extract_themes_from_items_with_dict(fake_llm):
    """Test _extract_themes_from_items with dict input."""
    synthesizer = SingleHopSpecificQuerySynthesizer(llm=fake_llm)

    items = {"Theme1": "value1", "Theme2": "value2"}
    themes = synthesizer._extract_themes_from_items(items)

    assert set(themes) == {"Theme1", "Theme2"}


def test_extract_themes_from_items_empty_input(fake_llm):
    """Test _extract_themes_from_items with empty input."""
    synthesizer = SingleHopSpecificQuerySynthesizer(llm=fake_llm)

    assert synthesizer._extract_themes_from_items([]) == []
    assert synthesizer._extract_themes_from_items(None) == []
    assert synthesizer._extract_themes_from_items("invalid") == []


def test_extract_themes_from_items_with_nested_empty_tuples(fake_llm):
    """Test _extract_themes_from_items skips non-string elements."""
    synthesizer = SingleHopSpecificQuerySynthesizer(llm=fake_llm)

    items = [("Theme1", 123), (456, "Theme2"), ("Theme3", "Theme3")]
    themes = synthesizer._extract_themes_from_items(items)

    # Only string elements should be extracted
    assert set(themes) == {"Theme1", "Theme2", "Theme3"}


@pytest.mark.asyncio
async def test_generate_scenarios_with_tuple_entities(fake_llm):
    """Test that _generate_scenarios handles tuple-formatted entities correctly.

    This test validates the fix for issue #2368 where entities property
    containing tuples would cause ValidationError.
    """
    # Create a node with tuple-formatted entities (the problematic case)
    node = Node(type=NodeType.CHUNK)
    node.add_property("entities", [("Entity1", "Entity1"), ("Entity2", "Entity2")])

    kg = KnowledgeGraph(nodes=[node])

    personas = [
        Persona(
            name="Researcher",
            role_description="A researcher interested in entities.",
        ),
    ]

    synthesizer = SingleHopSpecificQuerySynthesizer(llm=fake_llm)
    synthesizer.theme_persona_matching_prompt = MockThemePersonaMatchingPrompt()

    # This should not raise ValidationError
    scenarios = await synthesizer._generate_scenarios(
        n=2,
        knowledge_graph=kg,
        persona_list=personas,
        callbacks=None,
    )

    # Should generate scenarios successfully
    assert len(scenarios) > 0


@pytest.mark.asyncio
async def test_generate_sample_includes_metadata(fake_llm):
    node = Node(type=NodeType.CHUNK)
    node.add_property("page_content", "Context about microservices and patterns.")
    persona = Persona(name="Engineer", role_description="Builds systems")

    synthesizer = SingleHopSpecificQuerySynthesizer(llm=fake_llm)

    # Stub the prompt to avoid LLM dependency and return deterministic values
    class StubPrompt(PydanticPrompt):
        async def generate(self, data, llm, callbacks=None):  # type: ignore[override]
            class R:
                query = "What is microservices?"
                answer = "Microservices are loosely coupled services."

            return R()

    synthesizer.generate_query_reference_prompt = StubPrompt()

    # Build a minimal scenario
    from ragas.testset.synthesizers.base import QueryLength, QueryStyle
    from ragas.testset.synthesizers.single_hop.base import SingleHopScenario

    scenario = SingleHopScenario(
        nodes=[node],
        persona=persona,
        style=QueryStyle.PERFECT_GRAMMAR,
        length=QueryLength.MEDIUM,
        term="microservices",
    )

    sample = await synthesizer._generate_sample(scenario, callbacks=None)  # type: ignore[arg-type]

    assert sample.user_input == "What is microservices?"
    assert sample.reference == "Microservices are loosely coupled services."
    assert sample.reference_contexts == ["Context about microservices and patterns."]
    # New metadata fields
    assert sample.persona_name == "Engineer"
    assert sample.query_style == "PERFECT_GRAMMAR"
    assert sample.query_length == "MEDIUM"


@pytest.mark.asyncio
async def test_generate_scenarios_with_string_entities(fake_llm):
    """Test that _generate_scenarios still works with string-formatted entities."""
    # Create a node with string-formatted entities (the normal case)
    node = Node(type=NodeType.CHUNK)
    node.add_property("entities", ["Entity1", "Entity2", "Entity3"])

    kg = KnowledgeGraph(nodes=[node])

    personas = [
        Persona(
            name="Researcher",
            role_description="A researcher interested in entities.",
        ),
    ]

    synthesizer = SingleHopSpecificQuerySynthesizer(llm=fake_llm)
    synthesizer.theme_persona_matching_prompt = MockThemePersonaMatchingPrompt()

    # This should work as before
    scenarios = await synthesizer._generate_scenarios(
        n=2,
        knowledge_graph=kg,
        persona_list=personas,
        callbacks=None,
    )

    # Should generate scenarios successfully
    assert len(scenarios) > 0
