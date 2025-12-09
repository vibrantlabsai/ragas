"""AgentGoalAccuracy prompt classes and models."""

import typing as t

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt


class WorkflowInput(BaseModel):
    workflow: str = Field(
        ..., description="The agentic workflow comprised of Human, AI and Tools"
    )


class WorkflowOutput(BaseModel):
    user_goal: str = Field(
        ..., description="The task or objective the user wants to achieve"
    )
    end_state: str = Field(
        ..., description="The final outcome or result of the workflow"
    )


class InferGoalOutcomePrompt(BasePrompt[WorkflowInput, WorkflowOutput]):
    """Prompt for inferring user goal and end state from a workflow."""

    input_model = WorkflowInput
    output_model = WorkflowOutput

    instruction = "Given an agentic workflow comprised of Human, AI and Tools, identify the user_goal (the task or objective the user wants to achieve) and the end_state (the final outcome or result of the workflow)."

    examples = [
        (
            WorkflowInput(
                workflow="""Human: Hey, book a table at the nearest best Chinese restaurant for 8:00pm
AI: Sure, let me find the best options for you.
Tools:
  restaurant_search: {'cuisine': 'Chinese', 'time': '8:00pm'}
ToolOutput: Found a few options: 1. Golden Dragon, 2. Jade Palace
AI: I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?
Human: Let's go with Golden Dragon.
AI: Great choice! I'll book a table for 8:00pm at Golden Dragon.
Tools:
  restaurant_book: {'name': 'Golden Dragon', 'time': '8:00pm'}
ToolOutput: Table booked at Golden Dragon for 8:00pm.
AI: Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!
Human: thanks"""
            ),
            WorkflowOutput(
                user_goal="Book a table at the nearest best Chinese restaurant for 8:00pm.",
                end_state="A table is successfully booked at Golden Dragon (Chinese restaurant) for 8:00pm.",
            ),
        )
    ]


class CompareOutcomeInput(BaseModel):
    desired_outcome: str = Field(
        ..., description="The desired outcome or result of the workflow"
    )
    arrived_outcome: str = Field(
        ..., description="The actual outcome or result of the workflow"
    )


class CompareOutcomeOutput(BaseModel):
    reason: str = Field(
        ..., description="Explanation for why the outcomes match or differ"
    )
    verdict: t.Literal["0", "1"] = Field(
        ..., description="1 if outcomes match, 0 if they differ"
    )


class CompareOutcomePrompt(BasePrompt[CompareOutcomeInput, CompareOutcomeOutput]):
    """Prompt for comparing desired outcome with achieved outcome."""

    input_model = CompareOutcomeInput
    output_model = CompareOutcomeOutput

    instruction = "Given user goal, desired outcome and achieved outcome compare them and identify if they are the same (1) or different (0)."

    examples = [
        (
            CompareOutcomeInput(
                desired_outcome="A table is successfully booked at any Chinese restaurant for 8:00pm.",
                arrived_outcome="A table is successfully booked at Jade Palace (Chinese restaurant) for 8:00pm.",
            ),
            CompareOutcomeOutput(
                reason="The arrived outcome is same as the desired outcome and aligns with the user goal.",
                verdict="1",
            ),
        )
    ]
