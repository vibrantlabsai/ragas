"""Context Entity Recall prompt classes and models."""

from typing import List

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt


class ExtractEntitiesInput(BaseModel):
    """Input model for entity extraction."""

    text: str = Field(..., description="The text to extract entities from")


class EntitiesList(BaseModel):
    """Structured output for entity extraction."""

    entities: List[str] = Field(
        ..., description="List of unique entities extracted from the text"
    )


class ExtractEntitiesPrompt(BasePrompt[ExtractEntitiesInput, EntitiesList]):
    """Entity extraction prompt with structured input/output."""

    input_model = ExtractEntitiesInput
    output_model = EntitiesList

    instruction = """Given a text, extract unique entities without repetition.
Ensure you consider different forms or mentions of the same entity as a single entity.
Named entities include: persons, locations, organizations, dates, monetary amounts, and other proper nouns."""

    examples = [
        (
            ExtractEntitiesInput(
                text="The Eiffel Tower, located in Paris, France, is one of the most iconic landmarks globally. Millions of visitors are attracted to it each year for its breathtaking views of the city. Completed in 1889, it was constructed in time for the 1889 World's Fair."
            ),
            EntitiesList(
                entities=[
                    "Eiffel Tower",
                    "Paris",
                    "France",
                    "1889",
                    "World's Fair",
                ]
            ),
        ),
        (
            ExtractEntitiesInput(
                text="The Colosseum in Rome, also known as the Flavian Amphitheatre, stands as a monument to Roman architectural and engineering achievement. Construction began under Emperor Vespasian in AD 70 and was completed by his son Titus in AD 80. It could hold between 50,000 and 80,000 spectators who watched gladiatorial contests and public spectacles."
            ),
            EntitiesList(
                entities=[
                    "Colosseum",
                    "Rome",
                    "Flavian Amphitheatre",
                    "Vespasian",
                    "AD 70",
                    "Titus",
                    "AD 80",
                ]
            ),
        ),
        (
            ExtractEntitiesInput(
                text="The Great Wall of China, stretching over 21,196 kilometers from east to west, is a marvel of ancient defensive architecture. Built to protect against invasions from the north, its construction started as early as the 7th century BC. Today, it is a UNESCO World Heritage Site and a major tourist attraction."
            ),
            EntitiesList(
                entities=[
                    "Great Wall of China",
                    "21,196 kilometers",
                    "7th century BC",
                    "UNESCO World Heritage Site",
                ]
            ),
        ),
    ]
