import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.testset.synthesizers.prompts import PersonaThemesMapping
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.utils import calculate_split_values


class MockThemePersonaMatchingPrompt(PydanticPrompt):
    async def generate(self, data, llm, callbacks=None):
        return PersonaThemesMapping(
            mapping={persona.name: data.themes for persona in data.personas}
        )


class StubSingleHopSynthesizer(SingleHopSpecificQuerySynthesizer):
    async def _generate_sample(self, scenario, callbacks):
        return SingleTurnSample(user_input="question", reference="answer")


def make_synthesizer(llm, name):
    synthesizer = StubSingleHopSynthesizer(llm=llm, name=name)
    synthesizer.theme_persona_matching_prompt = MockThemePersonaMatchingPrompt()
    return synthesizer


def make_knowledge_graph(num_nodes=3):
    nodes = []
    for i in range(num_nodes):
        node = Node(type=NodeType.CHUNK)
        node.add_property("entities", [f"Entity{i}A", f"Entity{i}B", f"Entity{i}C"])
        nodes.append(node)
    return KnowledgeGraph(nodes=nodes)


PERSONAS = [Persona(name="Researcher", role_description="A curious researcher.")]


@pytest.mark.parametrize(
    "probs, n",
    [
        ([1 / 3, 1 / 3, 1 / 3], 10),
        ([1 / 3, 1 / 3, 1 / 3], 50),
        ([0.5, 0.25, 0.25], 10),
        ([0.5, 0.5], 3),
        ([1 / 3, 1 / 3, 1 / 3], 2),
    ],
)
def test_calculate_split_values_sums_to_n(probs, n):
    splits, split_values = calculate_split_values(probs, n)

    assert sum(splits) == n
    assert split_values[-1] == n
    for split, prob in zip(splits, probs):
        assert abs(split - n * prob) < 1


@pytest.mark.asyncio
async def test_generate_scenarios_returns_at_most_n(fake_llm):
    synthesizer = make_synthesizer(fake_llm, "single_hop")

    scenarios = await synthesizer.generate_scenarios(
        n=5, knowledge_graph=make_knowledge_graph(), persona_list=PERSONAS
    )

    assert len(scenarios) == 5


@pytest.mark.parametrize("testset_size", [2, 5, 10, 50])
def test_generate_returns_testset_size_samples(fake_llm, fake_embedding, testset_size):
    generator = TestsetGenerator(
        llm=fake_llm,
        embedding_model=fake_embedding,
        knowledge_graph=make_knowledge_graph(),
        persona_list=PERSONAS,
    )
    query_distribution = [
        (make_synthesizer(fake_llm, name), 1 / 3) for name in ("a", "b", "c")
    ]

    testset = generator.generate(
        testset_size=testset_size,
        query_distribution=query_distribution,
        num_personas=1,
    )

    assert len(testset.samples) == testset_size
