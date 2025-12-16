"""Utility functions and prompt classes for MultiModalRelevance metric."""

import typing as t

from pydantic import BaseModel, Field

from ragas.metrics.collections.multi_modal_faithfulness.util import (
    is_image_path_or_url,
    process_image_to_base64,
)


class MultiModalRelevanceInput(BaseModel):
    """Input model for multimodal relevance evaluation."""

    user_input: str = Field(..., description="The user's question or input")
    response: str = Field(..., description="The response to evaluate for relevance")
    retrieved_contexts: t.List[str] = Field(
        ...,
        description="List of retrieved contexts (text or image paths/URLs)",
    )


class MultiModalRelevanceOutput(BaseModel):
    """Output model for multimodal relevance evaluation."""

    relevant: bool = Field(
        ...,
        description="True if the response is relevant to the question and contexts, False otherwise",
    )
    reason: str = Field(
        default="",
        description="Explanation for the relevance verdict",
    )


def build_multimodal_relevance_message_content(
    instruction: str,
    user_input: str,
    response: str,
    retrieved_contexts: t.List[str],
) -> t.List[t.Dict[str, t.Any]]:
    """
    Build multimodal message content for relevance evaluation.

    Args:
        instruction: The evaluation instruction
        user_input: The user's question or input
        response: The response to evaluate
        retrieved_contexts: List of contexts (text or image references)

    Returns:
        List of content blocks for the message
    """
    content: t.List[t.Dict[str, t.Any]] = []

    # Add instruction, question, and response
    prompt_text = f"""{instruction}

Question: {user_input}

Response to evaluate: {response}

Retrieved contexts:
"""
    content.append({"type": "text", "text": prompt_text})

    # Process each context
    for i, ctx in enumerate(retrieved_contexts):
        # Try to process as image
        image_data = process_image_to_base64(ctx)

        if image_data:
            # Add as image
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_data['mime_type']};base64,{image_data['encoded_data']}"
                    },
                }
            )
            content.append({"type": "text", "text": f"[Image context {i + 1}]"})
        else:
            # Add as text
            content.append({"type": "text", "text": f"Context {i + 1}: {ctx}"})

    # Add closing instruction
    content.append(
        {
            "type": "text",
            "text": "\n\nBased on the above contexts (both visual and textual), determine if the response is relevant. A response is relevant if it appropriately addresses the question using information from the provided contexts.",
        }
    )

    return content


# Instruction for the prompt
MULTIMODAL_RELEVANCE_INSTRUCTION = """You are evaluating whether a response for a given question is relevant and in line with the provided context information.

A response is considered RELEVANT if:
- It appropriately addresses the user's question
- It is consistent with the visual and/or textual context provided
- The information in the response can be supported by the context

A response is considered NOT RELEVANT if:
- It does not address the user's question
- It contradicts or is not in line with the context information
- It provides information that is unrelated to both the question and context

You must evaluate relevance based on BOTH visual (images) and textual context if provided."""


__all__ = [
    "MultiModalRelevanceInput",
    "MultiModalRelevanceOutput",
    "build_multimodal_relevance_message_content",
    "is_image_path_or_url",
    "process_image_to_base64",
    "MULTIMODAL_RELEVANCE_INSTRUCTION",
]
