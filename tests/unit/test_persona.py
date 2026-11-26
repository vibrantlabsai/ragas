from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import (
    Persona,
    PersonaGenerationPrompt,
    generate_personas_from_kg,
)


class EchoPersonaPrompt(PersonaGenerationPrompt):
    """Returns a persona named after the summary it was given."""

    async def generate(self, llm, data, **kwargs):
        return Persona(name=data.text, role_description="test")


def _kg(summaries_and_embeddings):
    return KnowledgeGraph(
        nodes=[
            Node(
                type=NodeType.DOCUMENT,
                properties={"summary": summary, "summary_embedding": embedding},
            )
            for summary, embedding in summaries_and_embeddings
        ]
    )


def test_personas_dissimilar_unnormalized_embeddings_are_not_grouped(fake_llm):
    # cosine similarity is ~0.47, but the raw dot product is 2.0
    kg = _kg(
        [
            ("summary about cooking", [2.0, 0.5]),
            ("summary about astronomy", [0.5, 2.0]),
        ]
    )

    personas = generate_personas_from_kg(
        kg, fake_llm, persona_generation_prompt=EchoPersonaPrompt(), num_personas=2
    )

    assert {p.name for p in personas} == {
        "summary about cooking",
        "summary about astronomy",
    }


def test_personas_similar_small_norm_embeddings_are_grouped(fake_llm):
    # cosine similarity is ~1.0, but the raw dot product is only 0.25
    kg = _kg(
        [
            ("short summary", [0.5, 0.0]),
            ("a longer summary of the same topic", [0.5, 0.01]),
        ]
    )

    personas = generate_personas_from_kg(
        kg, fake_llm, persona_generation_prompt=EchoPersonaPrompt(), num_personas=2
    )

    assert [p.name for p in personas] == ["a longer summary of the same topic"] * 2
