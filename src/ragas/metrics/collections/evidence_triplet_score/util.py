"""Evidence triplet score prompt classes and models."""

import typing as t

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt


class EvidenceTripletInput(BaseModel):
    """Input model for evidence triplet scoring."""

    user_input: str = Field(..., description="The user question or task")
    response: str = Field(..., description="The generated response to evaluate")
    retrieved_contexts: t.List[str] = Field(
        ..., description="Retrieved context passages used as evidence"
    )


class EvidenceTripletOutput(BaseModel):
    """Structured output for evidence triplet scoring."""

    truth: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Degree to which retrieved evidence supports the response",
    )
    indeterminacy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Degree to which retrieved evidence is missing, ambiguous, incomplete, "
            "or insufficient"
        ),
    )
    falsity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Degree to which retrieved evidence contradicts the response or "
            "exposes unsupported claims"
        ),
    )
    reason: str = Field(..., description="Short explanation of the evidence state")


class EvidenceTripletPrompt(BasePrompt[EvidenceTripletInput, EvidenceTripletOutput]):
    """Prompt for scoring RAG evidence as truth, indeterminacy, and falsity."""

    input_model = EvidenceTripletInput
    output_model = EvidenceTripletOutput

    instruction = """You are evaluating a generated RAG response against retrieved contexts.
Return a neutrosophic evidence triplet with three independent scores in [0.0, 1.0].

Definitions:
- truth: retrieved contexts support the generated response.
- indeterminacy: retrieved contexts are missing, incomplete, ambiguous, pending, stale, or insufficient for judging the response.
- falsity: retrieved contexts contradict the response or show that important response claims are unsupported.

Important:
- The three scores are independent. Do not force them to sum to 1.
- Use high indeterminacy when the contexts do not provide enough evidence.
- Use high falsity when the contexts contradict the response.
- Use high truth only when the response is supported by the contexts.
- Base the evaluation only on the provided contexts.
Return JSON with truth, indeterminacy, falsity, and reason."""

    examples = [
        (
            EvidenceTripletInput(
                user_input="Where was Albert Einstein born?",
                response="Albert Einstein was born in Ulm, Germany.",
                retrieved_contexts=[
                    "Albert Einstein was born in Ulm, Germany on March 14, 1879."
                ],
            ),
            EvidenceTripletOutput(
                truth=0.95,
                indeterminacy=0.05,
                falsity=0.0,
                reason="The context directly supports the birthplace claim.",
            ),
        ),
        (
            EvidenceTripletInput(
                user_input="What caused the outage?",
                response="The outage was caused by a database failover.",
                retrieved_contexts=[
                    "Network routing errors were detected during the outage window.",
                    "Root cause analysis is still pending.",
                ],
            ),
            EvidenceTripletOutput(
                truth=0.15,
                indeterminacy=0.65,
                falsity=0.55,
                reason=(
                    "The contexts do not confirm the database cause and provide "
                    "some contrary network evidence while root cause remains pending."
                ),
            ),
        ),
    ]
