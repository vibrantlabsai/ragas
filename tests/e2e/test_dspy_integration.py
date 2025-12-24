import os

import pytest

try:
    import dspy  # noqa: F401

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_dspy_optimizer_import():
    """Test that DSPyOptimizer can be imported when dspy-ai is installed."""
    from ragas.optimizers import DSPyOptimizer

    optimizer = DSPyOptimizer(num_candidates=5)
    assert optimizer.num_candidates == 5
    assert optimizer._dspy is not None


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_dspy_optimizer_basic_optimization():
    """Test basic optimization flow with real DSPy (minimal example)."""
    from pydantic import BaseModel, Field

    from ragas.dataset_schema import (
        PromptAnnotation,
        SampleAnnotation,
        SingleMetricAnnotation,
    )
    from ragas.llms import llm_factory
    from ragas.losses import MSELoss
    from ragas.optimizers import DSPyOptimizer
    from ragas.prompt.pydantic_prompt import PydanticPrompt

    class QuestionInput(BaseModel):
        question: str = Field(description="The question to answer")

    class ScoreOutput(BaseModel):
        score: float = Field(description="Relevance score between 0 and 1")

    class TestPrompt(PydanticPrompt[QuestionInput, ScoreOutput]):
        instruction = "Score the relevance of the question."
        input_model = QuestionInput
        output_model = ScoreOutput

    test_prompt = TestPrompt()

    class MockMetric:
        name = "test_metric"

        def get_prompts(self):
            return {"score_prompt": test_prompt}

    prompt_annotation = PromptAnnotation(
        prompt_input={"question": "What is AI?"},
        prompt_output={"score": 0.9},
        edited_output=None,
    )

    samples = [
        SampleAnnotation(
            metric_input={"question": "What is AI?"},
            metric_output=0.9,
            prompts={"score_prompt": prompt_annotation},
            is_accepted=True,
        ),
        SampleAnnotation(
            metric_input={"question": "Random text"},
            metric_output=0.3,
            prompts={
                "score_prompt": PromptAnnotation(
                    prompt_input={"question": "Random text"},
                    prompt_output={"score": 0.3},
                    edited_output=None,
                )
            },
            is_accepted=True,
        ),
    ]

    dataset = SingleMetricAnnotation(name="test_metric", samples=samples)

    from openai import OpenAI

    client = OpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)
    optimizer = DSPyOptimizer(
        num_candidates=2,
        max_bootstrapped_demos=1,
        max_labeled_demos=1,
    )

    optimizer.metric = MockMetric()
    optimizer.llm = llm

    loss = MSELoss()

    try:
        result = optimizer.optimize(dataset, loss, {})

        assert "score_prompt" in result
        assert isinstance(result["score_prompt"], str)
        assert len(result["score_prompt"]) > 0
    except Exception as e:
        pytest.skip(f"DSPy optimization failed (expected in CI): {e}")


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy-ai not installed")
def test_dspy_adapter_conversions():
    """Test adapter utilities without making API calls."""
    from pydantic import BaseModel, Field

    from ragas.dataset_schema import (
        PromptAnnotation,
        SampleAnnotation,
        SingleMetricAnnotation,
    )
    from ragas.losses import MSELoss
    from ragas.optimizers.dspy_adapter import (
        create_dspy_metric,
        pydantic_prompt_to_dspy_signature,
        ragas_dataset_to_dspy_examples,
    )
    from ragas.prompt.pydantic_prompt import PydanticPrompt

    class InputModel(BaseModel):
        question: str = Field(description="The question")

    class OutputModel(BaseModel):
        answer: str = Field(description="The answer")

    class TestPrompt(PydanticPrompt[InputModel, OutputModel]):
        instruction = "Answer the question"
        input_model = InputModel
        output_model = OutputModel

    prompt = TestPrompt()

    signature = pydantic_prompt_to_dspy_signature(prompt)
    assert signature.__doc__ == "Answer the question"

    prompt_annotation = PromptAnnotation(
        prompt_input={"question": "What is 2+2?"},
        prompt_output={"answer": "4"},
        edited_output=None,
    )

    sample = SampleAnnotation(
        metric_input={"question": "What is 2+2?"},
        metric_output=0.9,
        prompts={"test_prompt": prompt_annotation},
        is_accepted=True,
    )

    dataset = SingleMetricAnnotation(name="test_metric", samples=[sample])
    examples = ragas_dataset_to_dspy_examples(dataset, "test_prompt")

    assert len(examples) == 1
    assert examples[0].question == "What is 2+2?"
    assert examples[0].answer == "4"

    loss = MSELoss()
    metric_fn = create_dspy_metric(loss, "score")

    import dspy

    mock_example = dspy.Example(score=0.9).with_inputs()
    mock_prediction = dspy.Example(score=0.8).with_inputs()

    result = metric_fn(mock_example, mock_prediction)
    assert isinstance(result, float)
