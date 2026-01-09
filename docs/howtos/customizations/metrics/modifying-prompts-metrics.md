# Modifying prompts in metrics

Every metric in Ragas that uses an LLM also uses one or more prompts to generate intermediate results that are used to formulate scores. Prompts can be treated like hyperparameters when using LLM-based metrics. An optimized prompt that suits your domain and use-case can increase the accuracy of your LLM-based metrics by 10-20%. Since optimal prompts depend on the LLM being used, you may want to tune the prompts that power each metric.

**Quick start**: If you need a simple custom metric, consider using [`DiscreteMetric`][ragas.metrics.discrete.DiscreteMetric] or [`NumericMetric`][ragas.metrics.numeric.NumericMetric] which accept custom prompts directly. See [Discrete Metrics](../../../concepts/metrics/overview/index.md#1-discrete-metrics) for examples.

This guide covers modifying prompts in **existing collection metrics** (like Faithfulness, FactualCorrectness) which use the [`BasePrompt`][ragas.prompt.BasePrompt] class. Make sure you have an understanding of the [Prompt Object documentation](../../../concepts/components/prompt.md) before going further.

## Understand the prompts of your metric

For metrics that support prompt customization, Ragas provides access to the underlying prompt objects through the metric instance. Let's look at how to access prompts in the `Faithfulness` metric:

```python
from ragas.metrics.collections import Faithfulness
from openai import AsyncOpenAI
from ragas.llms import llm_factory

# Setup dependencies
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric instance
scorer = Faithfulness(llm=llm)

# Faithfulness has two prompts:
# 1. statement_generator_prompt - breaks response into atomic statements
# 2. nli_statement_prompt - evaluates each statement against context
print(scorer.statement_generator_prompt)
print(scorer.nli_statement_prompt)
```

## Generating and viewing the prompt string

Let's view the prompt that will be sent to the LLM:

```python
from ragas.metrics.collections.faithfulness.util import StatementGeneratorInput

# Create sample input
sample_input = StatementGeneratorInput(
    question="What is the Eiffel Tower?",
    answer="The Eiffel Tower is located in Paris."
)

# Generate the prompt string
prompt_string = scorer.statement_generator_prompt.to_string(sample_input)
print(prompt_string)
```

## Modifying prompts

Modern metrics in Ragas use modular BasePrompt classes. To customize a prompt:

1. **Access the prompt**: The prompt is available as an attribute on metric instances
2. **Modify the prompt class**: Extend or subclass the prompt to customize instruction or examples
3. **Update the metric**: Assign your custom prompt to the metric's attribute

### Example: Customizing FactualCorrectness prompt

FactualCorrectness uses two prompts internally:
- `prompt` - ClaimDecompositionPrompt for breaking text into claims
- `nli_prompt` - NLIStatementPrompt for verifying claims against context

You can customize either or both:

```python
from ragas.metrics.collections import FactualCorrectness
from ragas.metrics.collections.factual_correctness.util import (
    ClaimDecompositionPrompt,
    NLIStatementPrompt,
)

# Create a custom claim decomposition prompt by subclassing
class CustomClaimDecompositionPrompt(ClaimDecompositionPrompt):
    instruction = """You are an expert at breaking down complex statements into atomic claims.
Break down the input text into clear, verifiable claims.
Only output valid JSON with a "claims" array."""

# Optionally customize the NLI prompt too
class CustomNLIPrompt(NLIStatementPrompt):
    instruction = """Carefully evaluate if each statement is supported by the context.
Be strict in your verification - only mark as supported if directly stated."""

# Create metric instance and replace prompts
scorer = FactualCorrectness(llm=llm)
scorer.prompt = CustomClaimDecompositionPrompt()
scorer.nli_prompt = CustomNLIPrompt()

# Now the metric will use the custom prompts
result = await scorer.ascore(
    response="The Eiffel Tower is in Paris and was built in 1889.",
    reference="The Eiffel Tower is located in Paris. It was completed in 1889."
)
```

### Example: Customizing Faithfulness examples

Few-shot examples can greatly influence LLM outputs. Here's how to modify them:

```python
from ragas.metrics.collections import Faithfulness
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementInput,
    NLIStatementOutput,
    NLIStatementPrompt,
    StatementFaithfulnessAnswer,
)

# Create custom prompt with domain-specific examples
class DomainSpecificNLIPrompt(NLIStatementPrompt):
    examples = [
        (
            NLIStatementInput(
                context="Machine learning is a field within artificial intelligence that enables systems to learn from data.",
                statements=[
                    "Machine learning is a subset of AI.",
                    "Machine learning uses statistical techniques.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Machine learning is a subset of AI.",
                        reason="The context states ML is 'a field within artificial intelligence', supporting this claim.",
                        verdict=1
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Machine learning uses statistical techniques.",
                        reason="The context doesn't mention statistical techniques.",
                        verdict=0
                    ),
                ]
            ),
        ),
    ]

# Update the metric with custom prompt
scorer = Faithfulness(llm=llm)
scorer.nli_statement_prompt = DomainSpecificNLIPrompt()

# Now evaluate with domain-specific prompts
result = await scorer.ascore(
    user_input="How do neural networks work?",
    response="Neural networks are inspired by biological neurons.",
    retrieved_contexts=["Artificial neural networks are computing systems loosely inspired by biological neural networks."]
)
```

## Adapting prompts to different languages

You can adapt prompts to different languages using the `adapt` method:

```python
from ragas.metrics.collections import Faithfulness

scorer = Faithfulness(llm=llm)

# Adapt the statement generator prompt to Spanish
adapted_prompt = await scorer.statement_generator_prompt.adapt(
    target_language="spanish",
    llm=llm,
    adapt_instruction=False  # Keep instruction in English, only translate examples
)

# Replace the prompt with the adapted version
scorer.statement_generator_prompt = adapted_prompt

# Now use the metric with Spanish examples
result = await scorer.ascore(
    user_input="¿Dónde nació Einstein?",
    response="Einstein nació en Alemania.",
    retrieved_contexts=["Albert Einstein nació en Alemania..."]
)
```

## Verifying your customizations

Here's how to verify your prompt customizations work:

```python
from ragas.metrics.collections.faithfulness.util import NLIStatementInput

# Create sample input to test the prompt
sample_input = NLIStatementInput(
    context="Paris is the capital and most populous city of France.",
    statements=["The capital of France is Paris.", "Paris is in Germany."]
)

# Generate and view the full prompt string
full_prompt = scorer.nli_statement_prompt.to_string(sample_input)
print("Full Prompt:")
print(full_prompt)
```

