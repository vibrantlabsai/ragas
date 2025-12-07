# Modifying prompts in metrics

Every metric in Ragas that uses an LLM also uses one or more prompts to generate intermediate results that are used to formulate scores. Prompts can be treated like hyperparameters when using LLM-based metrics. An optimized prompt that suits your domain and use-case can increase the accuracy of your LLM-based metrics by 10-20%. Since optimal prompts depend on the LLM being used, you may want to tune the prompts that power each metric.

Each prompt in Ragas is written using the [BasePrompt][ragas.prompt.metrics.base_prompt.BasePrompt] or [PydanticPrompt][ragas.prompt.pydantic_prompt.PydanticPrompt] classes. Make sure you have an understanding of the [Prompt Object documentation](../../../concepts/components/prompt.md) before going further.

### Understand the prompts of your Metric

For metrics that support prompt customization, Ragas provides access to the underlying prompt objects through the metric instance. Let's look at how to access and modify prompts in the `Faithfulness` metric:

```python
from ragas.metrics.collections import Faithfulness
from openai import AsyncOpenAI
from ragas.llms import llm_factory

# Setup dependencies
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric instance
scorer = Faithfulness(llm=llm)

# Access the internal prompt (implementation specific to each metric)
# Most modern metrics have prompts initialized in __init__
print(scorer.prompt)
```
Output
```
<ragas.metrics.collections.faithfulness.util.FaithfulnessPrompt at 0x7f8c41410970>
```



### Generating and viewing the prompt string

Let's view the prompt that will be sent to the LLM:

```python
# For metrics with BasePrompt, you can generate the prompt string
# Here's an example with Faithfulness metric
from ragas.metrics.collections.faithfulness.util import FaithfulnessInput

# Create sample input
sample_input = FaithfulnessInput(
    response="The Eiffel Tower is located in Paris.",
    context="The Eiffel Tower is an iconic iron lattice tower located in Paris, France."
)

# Generate the prompt string
prompt_string = scorer.prompt.to_string(sample_input)
print(prompt_string)
```
Output
```
Your task is to judge the faithfulness of a series of statements based on a given context.
For each statement you must return verdict as 1 if the statement can be directly inferred
based on the context or 0 if the statement can not be directly inferred based on the context.

[Example statements and context shown here...]
```

### Modifying prompts in modern metrics

Modern metrics in Ragas use modular BasePrompt classes that are part of each metric's `util.py` module. To customize a prompt:

1. **Access the prompt**: The prompt is available as an attribute (usually `self.prompt`) on metric instances
2. **Modify the prompt class**: Extend or subclass the prompt to customize instruction or examples
3. **Update the metric**: Pass your custom prompt during metric initialization or modify it afterward

Here's an example with the `FactualCorrectness` metric:

```python
from ragas.metrics.collections import FactualCorrectness
from ragas.metrics.collections.factual_correctness.util import (
    ClaimDecompositionInput,
    ClaimDecompositionOutput,
    ClaimDecompositionPrompt,
)

# Create a custom prompt by subclassing
class CustomClaimDecompositionPrompt(ClaimDecompositionPrompt):
    @property
    def instruction(self):
        return """You are an expert at breaking down complex statements into atomic claims.
Break down the input text into clear, verifiable claims.
Only output valid JSON with a "claims" array."""

# Create metric instance and replace prompt
scorer = FactualCorrectness(llm=llm)
scorer.prompt = CustomClaimDecompositionPrompt()

# Now the metric will use the custom prompt
result = await scorer.ascore(
    response="The Eiffel Tower is in Paris and was built in 1889.",
    reference="The Eiffel Tower is located in Paris. It was completed in 1889."
)
```

### Modifying examples in default prompt

Few-shot examples can greatly influence LLM outputs. The examples in default prompts may not reflect your specific domain or use-case. Here's how to modify them:

```python
from ragas.metrics.collections.faithfulness.util import (
    FaithfulnessInput,
    FaithfulnessOutput,
    FaithfulnessPrompt,
    StatementFaithfulnessAnswer,
)

# Create custom prompt with domain-specific examples
class DomainSpecificFaithfulnessPrompt(FaithfulnessPrompt):
    examples = [
        (
            FaithfulnessInput(
                response="Machine learning is a subset of AI that uses statistical techniques.",
                context="Machine learning is a field within artificial intelligence that enables systems to learn from data without being explicitly programmed.",
            ),
            FaithfulnessOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Machine learning is a subset of AI.",
                        reason="This statement is supported by the context which mentions ML as a field within AI.",
                        verdict=1
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Machine learning uses statistical techniques.",
                        reason="While related, the context doesn't explicitly mention statistical techniques.",
                        verdict=0
                    ),
                ]
            ),
        ),
        # Add more examples for your specific domain
    ]

# Update the metric with custom prompt
scorer = Faithfulness(llm=llm)
scorer.prompt = DomainSpecificFaithfulnessPrompt()

# Now evaluate with domain-specific prompts
result = await scorer.ascore(
    response="Neural networks are inspired by biological neurons.",
    context="Artificial neural networks are computing systems loosely inspired by biological neural networks found in animal brains."
)
```

This approach ensures that the LLM has examples that better reflect your domain and evaluation criteria.

### Saving and loading custom prompts (BasePrompt)

Modern metrics in Ragas use `BasePrompt` instances stored as attributes on the metric. You can save and load these prompts individually for reuse across evaluation runs. This is useful when you've customized prompts for your domain or adapted them to different languages.

**Why save prompts?**
- Avoid recreating customizations every time
- Reuse adapted prompts without re-running expensive LLM calls
- Share optimized prompts with your team
- Version control prompt modifications

#### Example: Saving and loading individual prompts

```python
import os
from ragas.metrics.collections import Faithfulness
from ragas.metrics.collections.faithfulness.util import StatementGeneratorPrompt
from openai import AsyncOpenAI
from ragas.llms import llm_factory

# Setup
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric instance
scorer = Faithfulness(llm=llm)

# Access the prompt you want to customize
# The metric has two prompts stored as instance attributes:
# - scorer.statement_generator_prompt
# - scorer.nli_statement_prompt

# Customize the statement generator prompt
custom_prompt = StatementGeneratorPrompt()
custom_prompt.instruction = "Custom instruction for your domain"

# Replace the prompt in the metric
scorer.statement_generator_prompt = custom_prompt

# Save the customized prompt to a file
os.makedirs("custom_prompts", exist_ok=True)
custom_prompt.save("custom_prompts/statement_generator_english.json")

# Later, in a different script or session, load the saved prompt
loaded_prompt = StatementGeneratorPrompt.load("custom_prompts/statement_generator_english.json")

# Use the loaded prompt in your metric
scorer.statement_generator_prompt = loaded_prompt

# Now evaluate with your custom prompt
result = await scorer.ascore(
    user_input="Where was Einstein born?",
    response="Einstein was born in Germany.",
    retrieved_contexts=["Albert Einstein was born in Germany..."]
)
```

**What's happening here:**
1. We access the specific prompt attribute on the metric (`scorer.statement_generator_prompt`)
2. We customize the prompt by modifying its properties
3. We call `prompt.save(file_path)` to save it to a JSON file (stores instruction, examples, language)
4. Later, we call `PromptClass.load(file_path)` to recreate the prompt from the saved file
5. We assign the loaded prompt back to the metric's attribute

#### Example: Adapting and saving prompts for different languages

```python
from ragas.metrics.collections import Faithfulness
from ragas.metrics.collections.faithfulness.util import StatementGeneratorPrompt
from openai import AsyncOpenAI
from ragas.llms import llm_factory
import os

# Setup with a good LLM for translation
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

scorer = Faithfulness(llm=llm)

# Adapt the prompt to Spanish (this uses the LLM to translate examples)
adapted_prompt = await scorer.statement_generator_prompt.adapt(
    target_language="spanish",
    llm=llm,
    adapt_instruction=False  # Keep instruction in English, only translate examples
)

# Save the adapted prompt for reuse
os.makedirs("adapted_prompts", exist_ok=True)
adapted_prompt.save("adapted_prompts/statement_generator_spanish.json")

# Replace the prompt in the metric with the adapted version
scorer.statement_generator_prompt = adapted_prompt

# Later, load the Spanish prompt without re-adapting
scorer.statement_generator_prompt = StatementGeneratorPrompt.load(
    "adapted_prompts/statement_generator_spanish.json"
)

# Now use the metric with Spanish prompts
result = await scorer.ascore(
    user_input="¿Dónde nació Einstein?",
    response="Einstein nació en Alemania.",
    retrieved_contexts=["Albert Einstein nació en Alemania..."]
)
```

**What's happening here:**
1. We call `prompt.adapt()` which uses the LLM to translate few-shot examples to the target language
2. This returns a new prompt instance with translated examples
3. We save this adapted prompt to avoid repeating the expensive adaptation step
4. In future runs, we load the saved Spanish prompt directly
5. The metric now evaluates using prompts with Spanish examples

#### Example: Creating custom prompt subclasses

```python
from ragas.metrics.collections import Faithfulness
from ragas.metrics.collections.faithfulness.util import (
    StatementGeneratorPrompt,
    StatementGeneratorInput,
    StatementGeneratorOutput,
)
from openai import AsyncOpenAI
from ragas.llms import llm_factory

# Create a custom version of the prompt for your specific domain
class MedicalStatementPrompt(StatementGeneratorPrompt):
    instruction = """Break down medical diagnoses into atomic, verifiable statements.
    Ensure each statement contains only one medical fact and uses precise medical terminology."""

    examples = [
        (
            StatementGeneratorInput(
                question="What is the patient's diagnosis?",
                answer="The patient has Type 2 diabetes with poor glycemic control and early signs of diabetic neuropathy.",
            ),
            StatementGeneratorOutput(
                statements=[
                    "The patient has Type 2 diabetes.",
                    "The patient has poor glycemic control.",
                    "The patient shows early signs of diabetic neuropathy.",
                ]
            ),
        ),
    ]

# Setup
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
scorer = Faithfulness(llm=llm)

# Use your custom prompt
custom_prompt = MedicalStatementPrompt()
scorer.statement_generator_prompt = custom_prompt

# Save for reuse
custom_prompt.save("domain_prompts/medical_statement_generator.json")

# Later, load your custom prompt
loaded = MedicalStatementPrompt.load("domain_prompts/medical_statement_generator.json")
scorer.statement_generator_prompt = loaded
```

**What's happening here:**
1. We create a subclass of the original prompt with domain-specific instruction and examples
2. We instantiate our custom prompt and assign it to the metric
3. We save the custom prompt with all its modifications
4. When loading, we use our custom class (`MedicalStatementPrompt.load()`) to ensure proper type
5. The metric now uses domain-specific prompts tailored to medical text evaluation

**Key points about BasePrompt save/load:**
- Each prompt is saved/loaded individually (not all prompts at once)
- You have full control over where and when to save prompts
- The saved JSON file contains: instruction, examples, language, and version info
- Use `prompt.save(file_path)` (instance method) to save
- Use `PromptClass.load(file_path)` (class method) to load
- Modern metrics may have multiple prompt attributes - save each one separately

### Full prompt customization example

Here's a complete example showing how to verify your customizations:

```python
# Create sample input to test the prompt
sample_input = FaithfulnessInput(
    response="The capital of France is Paris.",
    context="Paris is the capital and most populous city of France."
)

# Generate and view the full prompt string
full_prompt = scorer.prompt.to_string(sample_input)
print("Full Prompt:")
print(full_prompt)
print("\n" + "="*80 + "\n")

# Now use it for evaluation
result = await scorer.ascore(
    response="The capital of France is Paris.",
    context="Paris is the capital and most populous city of France."
)
print(f"Faithfulness Score: {result.value}")
```
