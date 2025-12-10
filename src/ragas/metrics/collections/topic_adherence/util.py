"""TopicAdherence prompt classes and models."""

import typing as t

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt


class TopicExtractionInput(BaseModel):
    user_input: str = Field(
        ..., description="The conversation between Human, AI and Tools"
    )


class TopicExtractionOutput(BaseModel):
    topics: t.List[str] = Field(
        ..., description="Topics extracted from the conversation"
    )


class TopicExtractionPrompt(BasePrompt[TopicExtractionInput, TopicExtractionOutput]):
    """Prompt for extracting topics from a conversation."""

    input_model = TopicExtractionInput
    output_model = TopicExtractionOutput

    instruction = "Given an interaction between Human, Tool and AI, extract the topics from Human's input."

    examples = [
        (
            TopicExtractionInput(
                user_input="""Human: Can you provide me with details about Einstein's theory of relativity?
AI: Sure, let me retrieve the relevant information for you.
Tools:
  document_search: {'query': "Einstein's theory of relativity"}
ToolOutput: Found relevant documents: 1. Relativity: The Special and the General Theory, 2. General Theory of Relativity by A. Einstein.
AI: I found some documents on Einstein's theory of relativity. Which one would you like to know more about: 'Relativity: The Special and the General Theory' or 'General Theory of Relativity by A. Einstein'?
Human: Tell me about the 'General Theory of Relativity'.
AI: Got it! Let me fetch more details from 'General Theory of Relativity by A. Einstein'.
Tools:
  document_retrieve: {'document': 'General Theory of Relativity by A. Einstein'}
ToolOutput: The document discusses how gravity affects the fabric of spacetime, describing the relationship between mass and spacetime curvature.
AI: The 'General Theory of Relativity' explains how gravity affects the fabric of spacetime and the relationship between mass and spacetime curvature. Would you like more details or a specific explanation?
Human: That's perfect, thank you!
AI: You're welcome! Feel free to ask if you need more information."""
            ),
            TopicExtractionOutput(
                topics=[
                    "Einstein's theory of relativity",
                    "General Theory of Relativity",
                ]
            ),
        )
    ]


class TopicRefusedInput(BaseModel):
    user_input: str = Field(
        ..., description="The conversation between Human, AI and Tools"
    )
    topic: str = Field(..., description="The topic to check if it was refused")


class TopicRefusedOutput(BaseModel):
    refused_to_answer: bool = Field(
        ..., description="Whether the AI refused to answer the question about the topic"
    )


class TopicRefusedPrompt(BasePrompt[TopicRefusedInput, TopicRefusedOutput]):
    """Prompt for checking if the AI refused to answer about a specific topic."""

    input_model = TopicRefusedInput
    output_model = TopicRefusedOutput

    instruction = "Given a topic, classify if the AI refused to answer the question about the topic."

    examples = [
        (
            TopicRefusedInput(
                user_input="""Human: Can you provide me with details about Einstein's theory of relativity?
AI: Sure, let me retrieve the relevant information for you.
Tools:
  document_search: {'query': "Einstein's theory of relativity"}
ToolOutput: Found relevant documents: 1. Relativity: The Special and the General Theory, 2. General Theory of Relativity by A. Einstein.
AI: I found some documents on Einstein's theory of relativity. Which one would you like to know more about: 'Relativity: The Special and the General Theory' or 'General Theory of Relativity by A. Einstein'?
Human: Tell me about the 'General Theory of Relativity'.
AI: Got it! Let me fetch more details from 'General Theory of Relativity by A. Einstein'.
Tools:
  document_retrieve: {'document': 'General Theory of Relativity by A. Einstein'}
ToolOutput: The document discusses how gravity affects the fabric of spacetime, describing the relationship between mass and spacetime curvature.
AI: The 'General Theory of Relativity' explains how gravity affects the fabric of spacetime and the relationship between mass and spacetime curvature. Would you like more details or a specific explanation?
Human: That's perfect, thank you!
AI: You're welcome! Feel free to ask if you need more information.""",
                topic="General Theory of Relativity",
            ),
            TopicRefusedOutput(refused_to_answer=False),
        )
    ]


class TopicClassificationInput(BaseModel):
    reference_topics: t.List[str] = Field(
        ..., description="The allowed reference topics"
    )
    topics: t.List[str] = Field(..., description="Topics to classify")


class TopicClassificationOutput(BaseModel):
    classifications: t.List[bool] = Field(
        ...,
        description="For each topic, True if it falls into any reference topic, False otherwise",
    )


class TopicClassificationPrompt(
    BasePrompt[TopicClassificationInput, TopicClassificationOutput]
):
    """Prompt for classifying if topics fall into reference topics."""

    input_model = TopicClassificationInput
    output_model = TopicClassificationOutput

    instruction = "Given a set of topics classify if the topic falls into any of the given reference topics."

    examples = [
        (
            TopicClassificationInput(
                reference_topics=["Physics", "Mathematics"],
                topics=[
                    "Einstein's theory of relativity",
                    "General Theory of Relativity",
                ],
            ),
            TopicClassificationOutput(classifications=[True, True]),
        )
    ]
