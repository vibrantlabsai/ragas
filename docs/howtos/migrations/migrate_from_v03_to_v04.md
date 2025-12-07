# Migration from v0.3 to v0.4

Ragas v0.4 introduces a fundamental shift towards an **experiment-based architecture**. This represents the most significant change since v0.2, moving from isolated metric evaluations to a cohesive experimentation framework where evaluation, analysis, and iteration are tightly integrated.

This architectural change led to several concrete improvements:

1. **Collections-Based Metrics System** - A standardized approach to metrics that work seamlessly within experiments
2. **Unified LLM Factory System** - Simplified LLM initialization with universal provider support
3. **Modern Prompt System** - Function-based prompts that are more composable and reusable

This guide will walk you through the key changes and provide step-by-step migration instructions.

## Overview of Major Changes

The shift to experiment-based architecture focuses on three core improvements:

1. **Experiment-Centric Design** - Move from one-off metric runs to structured experimentation workflows with integrated analysis
2. **Collections-Based Metrics** - Metrics designed to work within experiments, returning structured results for better analysis and tracking
3. **Enhanced LLM & Prompt System** - Universal provider support and modern prompt patterns enabling better experimentation

### Key Statistics

- **Metrics Migrated**: 20+ core metrics to the new collections system
- **Breaking Changes**: 7+ major API changes
- **Deprecations**: Legacy wrapper classes and old prompt definitions
- **New Features**: GPT-5/o-series support, automatic constraint handling, universal provider support

## Understanding the Experiment-Based Architecture

Before migrating, it helps to understand the shift in thinking:

**v0.3 (Metric-Centric):**
```
Data → Individual Metric → Score → Analysis
```

Each metric run was relatively isolated. You'd run a metric, get a float score, and handle tracking/analysis externally.

**v0.4 (Experiment-Centric):**
```
Data → Experiment → [Metrics Collection] → Structured Results → Integrated Analysis
```

Metrics now work within an experimentation context where evaluation, analysis, and iteration are integrated. This enables:

- Better tracking of metric results with explanations
- Easier comparison across experiment runs
- Built-in support for analyzing metric behavior
- Cleaner workflows for iterating on your system


## Migration Path

We recommend migrating in this order:

1. **Update evaluation approach** (Section: [Evaluation to Experiment](#evaluation-to-experiment)) - Switch from `evaluate()` to `experiment()`
2. **Update your LLM setup** (Section: [LLM Initialization](#llm-initialization))
3. **Migrate metrics** (Section: [Metrics Migration](#metrics-migration))
4. **Migrate embeddings** (Section: [Embeddings Migration](#embeddings-migration))
5. **Update prompts** (Section: [Prompt System Migration](#prompt-system-migration)) - If you're customizing prompts
6. **Update data schemas** (Section: [Data Schema Changes](#data-schema-changes))
7. **Refactor custom metrics** (Section: [Custom Metrics](#custom-metrics))

---

## Evaluation to Experiment

v0.4 replaces the `evaluate()` function with an `experiment()`-based approach to better support iterative evaluation workflows and structured result tracking.

### What Changed

The key shift: move from a **simple evaluation function** (`evaluate()`) that returns scores to an **experiment decorator** (`@experiment()`) that supports structured workflows with built-in tracking and versioning.

### Before (v0.3)

```python
from ragas import evaluate
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

# Setup
dataset = ...  # Your dataset
metrics = [Faithfulness(llm=llm), AnswerRelevancy(llm=llm)]

# Simple evaluation
result = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm,
    embeddings=embeddings
)

print(result)  # Returns EvaluationResult with scores
```

### After (v0.4)

```python
from ragas import experiment
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from pydantic import BaseModel

# Define experiment result structure
class ExperimentResult(BaseModel):
    faithfulness: float
    answer_relevancy: float

# Create experiment function
@experiment(ExperimentResult)
async def run_evaluation(row):
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm)

    faith_result = await faithfulness.ascore(
        response=row.response,
        retrieved_contexts=row.contexts
    )

    relevancy_result = await answer_relevancy.ascore(
        user_input=row.user_input,
        response=row.response
    )

    return ExperimentResult(
        faithfulness=faith_result.value,
        answer_relevancy=relevancy_result.value
    )

# Run experiment
exp_results = await run_evaluation(dataset)
```

### Benefits of Using `experiment()`

1. **Structured Results** - Define exactly what you want to track
2. **Per-Row Control** - Customize evaluation per sample if needed
3. **Version Tracking** - Optional git integration via `version_experiment()`
4. **Iterative Workflows** - Easy to modify and re-run experiments
5. **Better Integration** - Works seamlessly with modern metrics and datasets

---

## LLM Initialization

### What Changed

The v0.3 system required different factory functions depending on your use case:

- `instructor_llm_factory()` for metrics requiring instructor
- `llm_factory()` for general LLM operations
- Various wrapper classes for LangChain and LlamaIndex

v0.4 consolidates everything into a **single unified factory**:

```python
from ragas.llms import llm_factory
```

This factory:

- Returns `InstructorBaseRagasLLM` with guaranteed structured outputs
- Automatically detects and configures provider-specific constraints
- Supports GPT-5 and o-series models with automatic `temperature` and `top_p` constraints
- Works with all major providers: OpenAI, Anthropic, Cohere, Google, Azure, Bedrock, etc.

### Before (v0.3)

```python
from ragas.llms import instructor_llm_factory, llm_factory
from openai import AsyncOpenAI

# For metrics that need instructor
llm = instructor_llm_factory("openai", model="gpt-4o-mini", client=AsyncOpenAI(api_key="..."))

# Or, the old way (not recommended, still supported in 0.3)
client = AsyncOpenAI(api_key="sk-...")
llm = llm_factory("openai", model="gpt-4o-mini", client=client)
```

### After (v0.4)

```python
from ragas.llms import llm_factory
from openai import AsyncOpenAI

# Single unified approach - works everywhere
client = AsyncOpenAI(api_key="sk-...")
llm = llm_factory("gpt-4o-mini", client=client)
```

**Key differences:**

| Aspect | v0.3 | v0.4 |
|--------|------|------|
| **Factory function** | `instructor_llm_factory()` or `llm_factory()` | `llm_factory()` |
| **Provider detection** | Manual via provider string | Automatic from model name |
| **Return type** | `BaseRagasLLM` (various) | `InstructorBaseRagasLLM` |
| **Constraint handling** | Manual configuration | Automatic for GPT-5/o-series |
| **Async client required** | Yes | Yes |

### Migration Steps

1. **Update imports**:

    ```python
    # Remove this
    from ragas.llms import instructor_llm_factory

    # Use this instead
    from ragas.llms import llm_factory
    ```

2. **Replace factory calls**:

    ```python
    # Old - v0.3
    llm = instructor_llm_factory("openai", model="gpt-4o", client=client)

    # New - v0.4
    llm = llm_factory("gpt-4o", client=client)
    ```

3. **Update with other providers** (model name detection works automatically):

    ```python
    # OpenAI
    llm = llm_factory("gpt-4o-mini", client=AsyncOpenAI(api_key="..."))

    # Anthropic
    llm = llm_factory("claude-3-sonnet-20240229", client=AsyncAnthropic(api_key="..."))

    # Google
    llm = llm_factory("gemini-2.0-flash", client=...)
    ```

### LLM Wrapper Classes (Deprecated)

If you were using wrapper classes, they are now deprecated and will be removed in the future:

```python
# Deprecated - will be removed
from ragas.llms import LangchainLLMWrapper, LlamaIndexLLMWrapper
```

```python
# Recommended - use llm_factory directly
from ragas.llms import llm_factory
```

**Migration**: Replace wrapper initialization with direct `llm_factory()` calls. The factory now handles provider detection automatically.

---

## Metrics Migration

### Why Metrics Changed

The shift to experiment-based architecture required metrics to integrate better with the experimentation workflow:

- **Structured Results**: Metrics now return `MetricResult` objects (with score + reasoning) instead of raw floats, enabling richer analysis and tracking within experiments
- **Keyword Arguments**: Moving from sample objects to direct keyword arguments makes metrics easier to compose and integrate with experimental pipelines
- **Standardized Input/Output**: Collections-based metrics follow a consistent pattern, making it easier to build meta-analysis and experimentation features on top

### Architectural Changes

The metrics system has been completely redesigned to support experiment workflows. Here are the core differences:

#### Base Class Changes

| Aspect | v0.3 | v0.4 |
|--------|------|------|
| **Import** | `from ragas.metrics import Metric` | `from ragas.metrics.collections import Metric` |
| **Base Class** | `MetricWithLLM`, `SingleTurnMetric` | `BaseMetric` (from collections) |
| **Scoring Method** | `async def single_turn_ascore(sample: SingleTurnSample)` | `async def ascore(**kwargs)` |
| **Input Type** | `SingleTurnSample` objects | Individual keyword arguments |
| **Output Type** | `float` score | `MetricResult` (with `.value` and optional `.reason`) |
| **LLM Parameter** | Required at initialization | Required at initialization |

#### Scoring Workflow

**v0.3 Approach:**
```python
# 1. Create a sample object containing all data
sample = SingleTurnSample(
    user_input="What is AI?",
    response="AI is artificial intelligence...",
    retrieved_contexts=["Context 1", "Context 2"],
    ground_truths=["AI definition"]
)

# 2. Call metric with the sample
metric = Faithfulness(llm=llm)
score = await metric.single_turn_ascore(sample)  # Returns: 0.85
```

**v0.4 Approach:**
```python
# 1. Call metric with individual arguments
metric = Faithfulness(llm=llm)
result = await metric.ascore(
    user_input="What is AI?",
    response="AI is artificial intelligence...",
    retrieved_contexts=["Context 1", "Context 2"]
)

# 2. Access result properties
print(result.value)      # Score: 0.85 (float)
print(result.reason)     # Optional explanation
```

### Available Metrics in v0.4

The following metrics have been successfully migrated to the collections system in v0.4:

#### RAG Evaluation Metrics
- **Faithfulness** - Is the response grounded in retrieved context? (v0.3.9+)
- **AnswerRelevancy** - Is the response relevant to the user query? (v0.3.9+)
- **AnswerCorrectness** - Does the response match the reference answer? (v0.3.9+)
- **AnswerAccuracy** - Is the answer factually accurate?
- **ContextPrecision** - Are retrieved contexts ranked by relevance? (v0.3.9+)
  - With reference: `ContextPrecisionWithReference`
  - Without reference: `ContextPrecisionWithoutReference`
  - Legacy name: `ContextUtilization` (now a wrapper for ContextPrecisionWithoutReference)
- **ContextRecall** - Are all relevant contexts successfully retrieved? (v0.3.9+)
- **ContextRelevance** - What percentage of retrieved context is relevant? (v0.3.9+)
- **ContextEntityRecall** - Are important entities from reference in context? (v0.3.9+)
- **NoiseSensitivity** - How robust is the metric to irrelevant context? (v0.3.9+)
- **ResponseGroundedness** - Are all claims grounded in retrieved context?

#### Text Comparison Metrics
- **SemanticSimilarity** - Do two texts have similar semantic meaning? (v0.3.9+)
- **FactualCorrectness** - Are factual claims verified correctly? (v0.3.9+)
- **BleuScore** - Bilingual evaluation understudy score (v0.3.9+)
- **RougeScore** - Recall-oriented understudy for gisting evaluation (v0.3.9+)

#### String-Based Metrics (Non-LLM)
- **ExactMatch** - Exact string matching
- **StringPresence** - Substring presence checking
- **LevenshteinDistance** - Edit distance similarity
- **MatchingSubstrings** - Count of matching substrings
- **NonLLMStringSimilarity** - Various string similarity algorithms

#### Summary Metrics
- **SummaryScore** - Overall summary quality assessment (v0.3.9+)

#### Removed Metrics (No Longer Available)
- **AspectCritic** - Use `@discrete_metric()` decorator instead
- **SimpleCriteria** - Use `@discrete_metric()` decorator instead
- **AnswerSimilarity** - Use `SemanticSimilarity` instead

#### Agent & Tool Metrics (Not Yet Migrated)
- **ToolCallAccuracy** - Still on old architecture (Pending migration)
- **ToolCallF1** - Still on old architecture (Pending migration)
- **TopicAdherence** - Still on old architecture (Pending migration)
- **AgentGoalAccuracy** - Still on old architecture (Pending migration)

#### SQL Metrics (Not Yet Migrated)
- **DataCompy Score** - Still on old architecture (Pending migration)
- **SQL Query Equivalence** - Still on old architecture (Pending migration)

#### General Purpose & Rubric Metrics (Not Yet Migrated)
- **Domain-Specific Rubrics** - Still on old architecture (Pending migration)
- **Instance-Specific Rubrics** - Still on old architecture (Pending migration)

#### Specialized Metrics (Not Yet Migrated)
- **Multi-Modal Faithfulness** - Still on old architecture (Pending migration)
- **Multi-Modal Relevance** - Still on old architecture (Pending migration)
- **CHRF Score** - Still on old architecture (Pending migration)
- **Quoted Spans** - Still on old architecture (Pending migration)

!!! note "Migration Status"

    Approximately **43%** of core metrics have been migrated to the collections system (16 out of ~37 metrics).

    These remaining metrics will be migrated in **v0.4.x** releases. You can still use legacy metrics with the old API, though they will show deprecation warnings.

### Step-by-Step Migration

#### Step 1: Update Imports

```python
# v0.3
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
```

```python
# v0.4
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
```

#### Step 2: Initialize Metrics (No Change Required)

```python
# v0.3
metric = Faithfulness(llm=llm)
```

```python
# v0.4 - Same initialization
metric = Faithfulness(llm=llm)
```

#### Step 3: Update Metric Scoring Calls

Replace `single_turn_ascore(sample)` with `ascore(**kwargs)`:

```python
# v0.3
sample = SingleTurnSample(
    user_input="What is AI?",
    response="AI is artificial intelligence.",
    retrieved_contexts=["AI is a technology..."],
    ground_truths=["AI definition"]
)

score = await metric.single_turn_ascore(sample)
print(score)  # Output: 0.85
```

```python
# v0.4
result = await metric.ascore(
    user_input="What is AI?",
    response="AI is artificial intelligence.",
    retrieved_contexts=["AI is a technology..."]
)

print(result.value)   # Output: 0.85
print(result.reason)  # Optional: "Response is faithful to context"
```

#### Step 4: Handle MetricResult Objects

In v0.4, metrics return `MetricResult` objects instead of raw floats:

```python
from ragas.metrics.collections.base import MetricResult

result = await metric.ascore(...)

# Access the score
score_value = result.value  # float between 0 and 1

# Access the explanation (if available)
if result.reason:
    print(f"Reason: {result.reason}")

# Convert to float for compatibility
score_float = float(result.value)
```

### Metric-Specific Migrations

#### Faithfulness

**Before (v0.3):**
```python
sample = SingleTurnSample(
    user_input="What is machine learning?",
    response="ML is a subset of AI.",
    retrieved_contexts=["ML involves algorithms..."]
)
score = await metric.single_turn_ascore(sample)
```

**After (v0.4):**
```python
result = await metric.ascore(
    user_input="What is machine learning?",
    response="ML is a subset of AI.",
    retrieved_contexts=["ML involves algorithms..."]
)
score = result.value
```

#### AnswerRelevancy

**Before (v0.3):**
```python
sample = SingleTurnSample(
    user_input="What is Python?",
    response="Python is a programming language..."
)
score = await metric.single_turn_ascore(sample)
```

**After (v0.4):**
```python
result = await metric.ascore(
    user_input="What is Python?",
    response="Python is a programming language..."
)
score = result.value
```

#### AnswerCorrectness

Note: This metric now uses `reference` instead of `ground_truths`:

**Before (v0.3):**
```python
sample = SingleTurnSample(
    user_input="What is AI?",
    response="AI is artificial intelligence.",
    ground_truths=["AI is artificial intelligence and machine learning."]
)
score = await metric.single_turn_ascore(sample)
```

**After (v0.4):**
```python
result = await metric.ascore(
    user_input="What is AI?",
    response="AI is artificial intelligence.",
    reference="AI is artificial intelligence and machine learning."
)
score = result.value
```

#### ContextPrecision

**Before (v0.3):**
```python
sample = SingleTurnSample(
    user_input="What is RAG?",
    response="RAG improves LLM accuracy.",
    retrieved_contexts=["RAG = Retrieval Augmented Generation...", "..."],
    ground_truths=["RAG definition"]
)
score = await metric.single_turn_ascore(sample)
```

**After (v0.4):**
```python
result = await metric.ascore(
    user_input="What is RAG?",
    response="RAG improves LLM accuracy.",
    retrieved_contexts=["RAG = Retrieval Augmented Generation...", "..."],
    reference="RAG definition"
)
score = result.value
```

---

## Prompt System Migration

### Why Prompts Changed

The shift to a modular architecture means prompts are now **first-class components** that can be:

- **Customized per metric** - Each metric has a well-defined prompt interface
- **Type-safe** - Input/Output models define exact structure expected
- **Reusable** - Prompt classes follow a consistent pattern across metrics
- **Testable** - Prompts can be generated and inspected independently

v0.3 used simple string-based or dataclass prompts scattered throughout metrics. v0.4 consolidates them into a unified `BasePrompt` architecture with dedicated input/output models.

### Architectural Changes

#### Base Prompt System

| Aspect | v0.3 | v0.4 |
|--------|------|------|
| **Prompt Definition** | `PydanticPrompt` dataclasses or strings | `BasePrompt` classes with `to_string()` method |
| **Input/Output Types** | Generic Pydantic models | Metric-specific Input/Output models |
| **Access Method** | Scatter across metric code | Centralized in metric's `util.py` module |
| **Customization** | Difficult, requires deep changes | Simple subclassing with `instruction` and `examples` properties |
| **Organization** | Mixed in metric files | Organized in separate `util.py` files |

### Available Metric Prompts in v0.4

The following metrics now have well-defined, customizable prompts:

- **Faithfulness** - `FaithfulnessPrompt`, `FaithfulnessInput`, `FaithfulnessOutput`
- **Context Recall** - `ContextRecallPrompt`, `ContextRecallInput`, `ContextRecallOutput`
- **Context Precision** - `ContextPrecisionPrompt`, `ContextPrecisionInput`, `ContextPrecisionOutput`
- **Answer Relevancy** - `AnswerRelevancyPrompt`, `AnswerRelevancyInput`, `AnswerRelevancyOutput`
- **Answer Correctness** - `AnswerCorrectnessPrompt`, `AnswerCorrectnessInput`, `AnswerCorrectnessOutput`
- **Response Groundedness** - `ResponseGroundednessPrompt`, `ResponseGroundednessInput`, `ResponseGroundednessOutput`
- **Answer Accuracy** - `AnswerAccuracyPrompt`, `AnswerAccuracyInput`, `AnswerAccuracyOutput`
- **Context Relevance** - `ContextRelevancePrompt`, `ContextRelevanceInput`, `ContextRelevanceOutput`
- **Context Entity Recall** - `ContextEntityRecallPrompt`, `ContextEntityRecallInput`, `ContextEntityRecallOutput`
- **Factual Correctness** - `ClaimDecompositionPrompt`, `VerificationPrompt`, with associated Input/Output models
- **Noise Sensitivity** - `NoiseAugmentationPrompt` with associated models
- **Summary Score** - `SummaryScorePrompt`, `SummaryScoreInput`, `SummaryScoreOutput`

### Step-by-Step Migration

#### Step 1: Access Prompts in Your Metrics

```python
from ragas.metrics.collections import Faithfulness
from ragas.llms import llm_factory

# Create metric instance
metric = Faithfulness(llm=llm)

# Access the prompt object
print(metric.prompt)  # <ragas.metrics.collections.faithfulness.util.FaithfulnessPrompt>
```

#### Step 2: View Prompt Strings

```python
from ragas.metrics.collections.faithfulness.util import FaithfulnessInput

# Create sample input
sample_input = FaithfulnessInput(
    response="The Eiffel Tower is in Paris.",
    context="The Eiffel Tower is located in Paris, France."
)

# Generate prompt string
prompt_string = metric.prompt.to_string(sample_input)
print(prompt_string)
```

#### Step 3: Customize Prompts (If Needed)

**Option A: Subclass the default prompt**

```python
from ragas.metrics.collections import Faithfulness
from ragas.metrics.collections.faithfulness.util import FaithfulnessPrompt

# Create custom prompt by subclassing
class CustomFaithfulnessPrompt(FaithfulnessPrompt):
    @property
    def instruction(self):
        return """Your custom instruction here."""

# Apply to metric
metric = Faithfulness(llm=llm)
metric.prompt = CustomFaithfulnessPrompt()
```

**Option B: Customize examples for domain-specific evaluation**

```python
from ragas.metrics.collections.faithfulness.util import (
    FaithfulnessInput,
    FaithfulnessOutput,
    FaithfulnessPrompt,
    StatementFaithfulnessAnswer,
)

class DomainSpecificPrompt(FaithfulnessPrompt):
    examples = [
        (
            FaithfulnessInput(
                response="ML uses statistical techniques.",
                context="Machine learning is a field that uses algorithms to learn from data.",
            ),
            FaithfulnessOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="ML uses statistical techniques.",
                        reason="Related to learning from data, but context doesn't explicitly mention statistical techniques.",
                        verdict=0
                    ),
                ]
            ),
        ),
    ]

# Apply custom prompt
metric = Faithfulness(llm=llm)
metric.prompt = DomainSpecificPrompt()
```

### Common Prompt Customizations

#### Changing Instructions

Most metrics allow overriding the instruction property:

```python
class StrictFaithfulnessPrompt(FaithfulnessPrompt):
    @property
    def instruction(self):
        return """Be very strict when judging faithfulness.
Only mark statements as faithful (verdict=1) if they are directly stated or strongly implied."""
```

#### Adding Domain Examples

Domain-specific examples significantly improve metric accuracy (10-20% improvement):

```python
class MedicalFaithfulnessPrompt(FaithfulnessPrompt):
    examples = [
        # Medical domain examples here
    ]
```

#### Changing Output Format

For advanced customization, subclass the prompt and override the `to_string()` method:

```python
class CustomPrompt(FaithfulnessPrompt):
    def to_string(self, input: FaithfulnessInput) -> str:
        # Custom prompt generation logic
        return "..."
```

### Verifying Custom Prompts

Always verify your custom prompts before using them:

```python
# Test prompt generation
sample_input = FaithfulnessInput(
    response="Test response.",
    context="Test context."
)

custom_metric = Faithfulness(llm=llm)
custom_metric.prompt = MyCustomPrompt()

# View the generated prompt
prompt_string = custom_metric.prompt.to_string(sample_input)
print(prompt_string)

# Then use it for evaluation
result = await custom_metric.ascore(
    response="Test response.",
    context="Test context."
)
```

### Migration from v0.3 Custom Prompts

If you had custom prompts in v0.3 using `PydanticPrompt`:

**Before (v0.3) - Dataclass approach:**
```python
from ragas.prompt.pydantic_prompt import PydanticPrompt
from pydantic import BaseModel

class MyInput(BaseModel):
    response: str
    context: str

class MyOutput(BaseModel):
    is_faithful: bool

class MyPrompt(PydanticPrompt[MyInput, MyOutput]):
    instruction = "Check if response is faithful to context"
    input_model = MyInput
    output_model = MyOutput
    examples = [...]
```

**After (v0.4) - BasePrompt approach:**
```python
from ragas.metrics.collections.base import BasePrompt
from pydantic import BaseModel

class MyInput(BaseModel):
    response: str
    context: str

class MyOutput(BaseModel):
    is_faithful: bool

class MyPrompt(BasePrompt):
    @property
    def instruction(self):
        return "Check if response is faithful to context"

    @property
    def input_model(self):
        return MyInput

    @property
    def output_model(self):
        return MyOutput

    @property
    def examples(self):
        return [...]

    def to_string(self, input: MyInput) -> str:
        # Generate prompt string from input
        return f"Check if this is faithful: {input.response}"
```

### Language Adaptation with BasePrompt.adapt()

v0.4 introduces the `adapt()` method on `BasePrompt` instances for language translation, replacing the deprecated `PromptMixin.adapt_prompts()` approach.

#### Before (v0.3) - PromptMixin Approach

```python
from ragas.prompt.mixin import PromptMixin
from ragas.metrics import Faithfulness

# Metrics inherited from PromptMixin to use adapt_prompts
class MyFaithfulness(Faithfulness, PromptMixin):
    pass

metric = MyFaithfulness(llm=llm)

# Adapt ALL prompts to another language
adapted_prompts = await metric.adapt_prompts(
    language="spanish",
    llm=llm,
    adapt_instruction=True
)

# Apply all adapted prompts
metric.set_prompts(**adapted_prompts)
```

**Issues with v0.3 approach:**
- Required mixin inheritance (tightly coupled)
- All prompts adapted together (inflexible)
- Mixin methods scattered across codebase

#### After (v0.4) - BasePrompt.adapt() Method

```python
from ragas.metrics.collections import Faithfulness

# Create metric with default prompt
metric = Faithfulness(llm=llm)

# Adapt individual prompt to another language
adapted_prompt = await metric.prompt.adapt(
    target_language="spanish",
    llm=llm,
    adapt_instruction=True
)

# Apply adapted prompt
metric.prompt = adapted_prompt

# Use metric with adapted language
result = await metric.ascore(
    response="...",
    retrieved_contexts=[...]
)
```

!!! note ""
    Save and load prompts will be available in a future version of v0.4.x using BasePrompt. Currently, PromptMixin only has it.

#### Language Adaptation Examples

**Adapt without instruction text (lightweight):**
```python
from ragas.metrics.collections import AnswerRelevancy

metric = AnswerRelevancy(llm=llm)

# Only update language field, keep instruction in English
adapted_prompt = await metric.prompt.adapt(
    target_language="french",
    llm=llm,
    adapt_instruction=False  # Default - just updates language
)

metric.prompt = adapted_prompt
print(metric.prompt.language)  # "french"
```

**Adapt with instruction translation (full translation):**
```python
# Translate both instruction and examples
adapted_prompt = await metric.prompt.adapt(
    target_language="german",
    llm=llm,
    adapt_instruction=True  # Translate instruction text too
)

metric.prompt = adapted_prompt

# Examples are also automatically translated
# Both instruction and examples in German now
```

**Adapt custom prompts:**
```python
from ragas.metrics.collections.faithfulness.util import FaithfulnessPrompt

class CustomFaithfulnessPrompt(FaithfulnessPrompt):
    @property
    def instruction(self):
        return "Custom instruction in English"

prompt = CustomFaithfulnessPrompt(language="english")

# Adapt to Italian
adapted = await prompt.adapt(
    target_language="italian",
    llm=llm,
    adapt_instruction=True
)

# Check language was updated
assert adapted.language == "italian"
```

#### Migration from v0.3 to v0.4

**Step 1: Remove PromptMixin inheritance**

```python
# v0.3
from ragas.prompt.mixin import PromptMixin
from ragas.metrics import Faithfulness

class MyMetric(Faithfulness, PromptMixin):  # ← Remove PromptMixin
    pass

# v0.4
from ragas.metrics.collections import Faithfulness

# No mixin needed - just use the metric directly
metric = Faithfulness(llm=llm)
```

**Step 2: Replace adapt_prompts() with adapt()**

```python
# v0.3
adapted_prompts = await metric.adapt_prompts(
    language="spanish",
    llm=llm,
    adapt_instruction=True
)
metric.set_prompts(**adapted_prompts)

# v0.4
adapted_prompt = await metric.prompt.adapt(
    target_language="spanish",
    llm=llm,
    adapt_instruction=True
)
metric.prompt = adapted_prompt
```

#### Complete Migration Example

**Before (v0.3):**
```python
from ragas.prompt.mixin import PromptMixin
from ragas.metrics import Faithfulness, AnswerRelevancy

class MyMetrics(Faithfulness, AnswerRelevancy, PromptMixin):
    pass

# Setup
metrics = MyMetrics(llm=llm)

# Adapt multiple metrics to Spanish
adapted = await metrics.adapt_prompts(
    language="spanish",
    llm=best_llm,
    adapt_instruction=True
)

metrics.set_prompts(**adapted)
metrics.save_prompts("./spanish_prompts")
```

**After (v0.4):**
```python
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

# Setup individual metrics
faith_metric = Faithfulness(llm=llm)
answer_metric = AnswerRelevancy(llm=llm)

# Adapt each metric's prompt independently
faith_adapted = await faith_metric.prompt.adapt(
    target_language="spanish",
    llm=best_llm,
    adapt_instruction=True
)
faith_metric.prompt = faith_adapted

answer_adapted = await answer_metric.prompt.adapt(
    target_language="spanish",
    llm=best_llm,
    adapt_instruction=True
)
answer_metric.prompt = answer_adapted

# Use metrics with adapted prompts
faith_result = await faith_metric.ascore(...)
answer_result = await answer_metric.ascore(...)
```

---

## Data Schema Changes

### SingleTurnSample Updates

The `SingleTurnSample` schema has been updated with breaking changes:

#### `ground_truths` → `reference`

The `ground_truths` parameter has been renamed to `reference` across the board:

**Before (v0.3):**
```python
sample = SingleTurnSample(
    user_input="...",
    response="...",
    ground_truths=["correct answer"]  # List of strings
)
```

**After (v0.4):**
```python
sample = SingleTurnSample(
    user_input="...",
    response="...",
    reference="correct answer"  # Single string
)
```

!!! tip ""

    - v0.3 used `ground_truths` as a **list**
    - v0.4 uses `reference` as a **single string**
    - For multiple references, use separate evaluation runs

#### Updated Schema

```python
from ragas import SingleTurnSample

# v0.4 complete sample
sample = SingleTurnSample(
    user_input="What is AI?",                      # Required
    response="AI is artificial intelligence.",     # Required
    retrieved_contexts=["Context 1", "Context 2"], # Optional
    reference="Correct definition of AI"           # Optional (was ground_truths)
)
```

### EvaluationDataset Updates

If you're using `EvaluationDataset`, update your data loading:

**Before (v0.3):**
```python
dataset = EvaluationDataset(
    samples=[
        SingleTurnSample(
            user_input="Q1",
            response="A1",
            ground_truths=["correct"]
        )
    ]
)
```

**After (v0.4):**
```python
dataset = EvaluationDataset(
    samples=[
        SingleTurnSample(
            user_input="Q1",
            response="A1",
            reference="correct"
        )
    ]
)
```

If loading from CSV/JSON, update your data files:

**Before (v0.3) CSV format:**
```csv
user_input,response,retrieved_contexts,ground_truths
"Q1","A1","[""ctx1""]","[""correct""]"
```

**After (v0.4) CSV format:**
```csv
user_input,response,retrieved_contexts,reference
"Q1","A1","[""ctx1""]","correct"
```

---

## Custom Metrics

### For Metrics Using Collections-Based Architecture

If you've already written custom metrics extending `BaseMetric` from collections, minimal changes are needed:

```python
from ragas.metrics.collections.base import BaseMetric, MetricResult
from pydantic import BaseModel

class MyCustomMetric(BaseMetric):
    name: str = "my_metric"
    dimensions: list[str] = ["my_dimension"]

    async def ascore(self, **kwargs) -> MetricResult:
        # Your metric logic
        score = 0.85
        reason = "Explanation of the score"
        return MetricResult(value=score, reason=reason)
```

**Key considerations:**

- Extend `BaseMetric`, not old `MetricWithLLM`
- Implement `async def ascore(**kwargs)` instead of `single_turn_ascore(sample)`
- Return `MetricResult` objects, not raw floats
- Use keyword arguments instead of `SingleTurnSample`

### For Metrics Using Legacy Architecture

If you have custom metrics extending `SingleTurnMetric` or `MetricWithLLM`:

```python
# v0.3 - Legacy approach
from ragas.metrics.base import MetricWithLLM

class MyMetric(MetricWithLLM):
    async def single_turn_ascore(self, sample: SingleTurnSample) -> float:
        # Extract values from sample
        user_input = sample.user_input
        response = sample.response
        contexts = sample.retrieved_contexts or []

        # Your logic
        return 0.85
```

**Migration path:**

1. Extend `BaseMetric` from collections instead
2. Change method signature to use keyword arguments
3. Return `MetricResult` instead of float
4. Add `dimensions` property if not present

```python
# v0.4 - Collections approach
from ragas.metrics.collections.base import BaseMetric, MetricResult

class MyMetric(BaseMetric):
    name: str = "my_metric"
    dimensions: list[str] = ["quality"]

    async def ascore(self,
                    user_input: str,
                    response: str,
                    retrieved_contexts: list[str] | None = None,
                    **kwargs) -> MetricResult:
        # Use keyword arguments directly
        contexts = retrieved_contexts or []

        # Your logic
        score = 0.85
        return MetricResult(value=score, reason="Optional explanation")
```

### Prompt System Updates

#### v0.3 - Dataclass-Based Prompts

```python
from ragas.prompt.pydantic_prompt import PydanticPrompt
from pydantic import BaseModel

class Input(BaseModel):
    query: str
    document: str

class Output(BaseModel):
    is_relevant: bool

class RelevancePrompt(PydanticPrompt[Input, Output]):
    instruction = "Is the document relevant to the query?"
    input_model = Input
    output_model = Output
    examples = [...]
```

#### v0.4 - Function-Based Prompts

The new approach uses simple functions:

```python
def relevance_prompt(query: str, document: str) -> str:
    return f"""Determine if the document is relevant to the query.

Query: {query}
Document: {document}

Respond with YES or NO."""
```

**Benefits:**

- Simpler and more composable
- No boilerplate class definitions
- Easier to test and modify
- Native Python type hints

**Migration:**

- Identify where you define prompts in custom metrics
- Convert dataclass definitions to functions
- Update metric to use the function directly

---

## Removed Features

The following features have been completely removed from v0.4 and will cause errors if used:

### Functions

**`instructor_llm_factory()`** - Removed entirely

- **Merged into**: `llm_factory()` function
- **Migration**: Replace all calls to `instructor_llm_factory()` with `llm_factory()`
- **Impact**: Direct breaking change, no fallback

**Before (v0.3) - No longer works:**

```python
llm = instructor_llm_factory("openai", model="gpt-4o", client=client)
```

**After (v0.4) - Use this instead:**
```python
llm = llm_factory("gpt-4o", client=client)
```

### Metrics

Three metrics have been completely removed from the collections API. They are no longer available and have no direct replacement:

**1. AspectCritic** - Removed

- **Reason**: Replaced by more flexible discrete metric pattern
- **Alternative**: Use `@discrete_metric()` decorator for custom aspect evaluation
- **Usage**:
  ```python
  # Instead of AspectCritic, use:
  from ragas.metrics import discrete_metric

  @discrete_metric(name="aspect_critic", allowed_values=["positive", "negative", "neutral"])
  def evaluate_aspect(response: str, aspect: str) -> str:
      # Your evaluation logic
      return "positive"
  ```

**2. SimpleCriteria** - Removed

- **Reason**: Replaced by more flexible discrete metric pattern
- **Alternative**: Use `@discrete_metric()` decorator for custom criteria
- **Usage**:
  ```python
  from ragas.metrics import discrete_metric

  @discrete_metric(name="custom_criteria", allowed_values=["pass", "fail"])
  def evaluate_criteria(response: str, criteria: str) -> str:
      return "pass" if criteria in response else "fail"
  ```

**3. AnswerSimilarity** - Removed (Redundant)

- **Reason**: Functionality fully covered by `SemanticSimilarity`
- **Direct replacement**: `SemanticSimilarity`
- **Usage**:
  ```python
  # v0.3 - No longer available
  from ragas.metrics import AnswerSimilarity  # ERROR

  # v0.4 - Use this instead
  from ragas.metrics.collections import SemanticSimilarity
  metric = SemanticSimilarity(llm=llm)
  result = await metric.ascore(
      reference="Expected answer",
      response="Actual answer"
  )
  ```

### Deprecated Methods (Removed in v0.4)

**`Metric.ascore()` and `Metric.score()`** - Removed

- **When removed**: Marked for removal in v0.3, removed in v0.4
- **Why**: Replaced by collections-based `ascore(**kwargs)` pattern
- **Migration**: Use collections metrics instead

**Legacy sample-based methods** - Removed

- **`single_turn_ascore(sample: SingleTurnSample)`** - Only on legacy metrics
- **Replace with**: Collections metrics using `ascore(**kwargs)`

---

## Deprecated Features

These features still work but show deprecation warnings. They will be removed in a **future release**.

### evaluate() Function - Deprecated

- **Status**: Still works but discouraged
- **Reason**: Replaced by `@experiment()` decorator for better structured workflows
- **Migration**: See [Evaluation to Experiment](#evaluation-to-experiment) section

**Before (v0.3) - Deprecated:**
```python
from ragas import evaluate

result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)
```

**After (v0.4) - Recommended:**
```python
from ragas import experiment
from pydantic import BaseModel

class Results(BaseModel):
    score: float

@experiment(Results)
async def run(row):
    result = await metric.ascore(**row.dict())
    return Results(score=result.value)

result = await run(dataset)
```

### LLM Wrapper Classes

#### LangchainLLMWrapper - Deprecated

- **Status**: Still works but discouraged
- **Deprecation warning**:
  ```
  Direct usage of LangChain LLMs with Ragas prompts is deprecated and will be
  removed in a future version. Use Ragas LLM interfaces instead
  ```
- **Migration**: Use `llm_factory()` with native client instead

**Before (v0.3) - Deprecated:**
```python
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

langchain_llm = ChatOpenAI(model="gpt-4o")
ragas_llm = LangchainLLMWrapper(langchain_llm)
```

**After (v0.4) - Recommended:**
```python
from ragas.llms import llm_factory
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="...")
ragas_llm = llm_factory("gpt-4o", client=client)
```

#### LlamaIndexLLMWrapper - Deprecated

- **Status**: Still works but discouraged
- **Similar warning** as LangchainLLMWrapper
- **Migration**: Use `llm_factory()` with native client

**Before (v0.3) - Deprecated:**
```python
from ragas.llms import LlamaIndexLLMWrapper
from llama_index.llms.openai import OpenAI

llamaindex_llm = OpenAI(model="gpt-4o")
ragas_llm = LlamaIndexLLMWrapper(llamaindex_llm)
```

**After (v0.4) - Recommended:**
```python
from ragas.llms import llm_factory
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="...")
ragas_llm = llm_factory("gpt-4o", client=client)
```

### Embeddings Migration

#### LangchainEmbeddingsWrapper & LlamaIndexEmbeddingsWrapper - Deprecated

- **Status**: Still work but show deprecation warnings
- **Reason**: Replaced by native embedding providers with direct client integration
- **Migration**: See [Embeddings Migration](#embeddings-migration) section

v0.4 replaces wrapper classes with **native embedding providers** that integrate directly with client libraries instead of using LangChain wrappers.

### What Changed

| Aspect | v0.3 | v0.4 |
|--------|------|------|
| **Class** | `LangchainEmbeddingsWrapper`, `LlamaIndexEmbeddingsWrapper` | `OpenAIEmbeddings`, `GoogleEmbeddings`, `HuggingFaceEmbeddings` |
| **Client** | LangChain/LlamaIndex wrapper | Native client (OpenAI, Google, etc.) |
| **Methods** | `embed_query()`, `embed_documents()` | `embed_text()`, `embed_texts()` |
| **Setup** | Wrap existing LangChain object | Pass native client directly |

#### OpenAI Migration

**Before (v0.3):**
```python
from langchain_openai import OpenAIEmbeddings as LangChainEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

embeddings = LangchainEmbeddingsWrapper(
    LangChainEmbeddings(api_key="sk-...")
)
embedding = embeddings.embed_query("text")
```

**After (v0.4):**
```python
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    client=AsyncOpenAI(api_key="sk-..."),
    model="text-embedding-3-small"
)
embedding = embeddings.embed_text("text")  # Different method name
```

#### Google Embeddings Migration

**Before (v0.3):**
```python
from langchain_community.embeddings import VertexAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

embeddings = LangchainEmbeddingsWrapper(
    VertexAIEmbeddings(model_name="textembedding-gecko@001", project="my-project")
)
```

**After (v0.4):**
```python
from ragas.embeddings import GoogleEmbeddings

embeddings = GoogleEmbeddings(
    model="text-embedding-004",
    use_vertex=True,
    project_id="my-project"
)
```

#### HuggingFace Migration

**Before (v0.3):**
```python
from ragas.embeddings import HuggingfaceEmbeddings

embeddings = HuggingfaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

**After (v0.4):**
```python
from ragas.embeddings import HuggingFaceEmbeddings  # Capitalization changed

embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"  # Optional GPU acceleration
)
```

### Using embedding_factory()

**Before (v0.3):**
```python
from ragas.embeddings import embedding_factory

embeddings = embedding_factory()  # Defaults to OpenAI
```

**After (v0.4):**
```python
from ragas.embeddings import embedding_factory
from openai import AsyncOpenAI

embeddings = embedding_factory(
    provider="openai",
    model="text-embedding-3-small",
    client=AsyncOpenAI(api_key="sk-...")
)
```

### Prompt System

#### Dataclass-based prompts (PydanticPrompt) - Deprecated

- **Status**: Legacy prompts still work but discouraged
- **Deprecation**: Modular BasePrompt architecture is now preferred
- **Migration**: See [Prompt System Migration](#prompt-system-migration) section

**Before (v0.3) - Deprecated approach:**
```python
from ragas.prompt.pydantic_prompt import PydanticPrompt
from pydantic import BaseModel

class Input(BaseModel):
    query: str

class Output(BaseModel):
    is_relevant: bool

class RelevancePrompt(PydanticPrompt[Input, Output]):
    instruction = "Is this relevant?"
    input_model = Input
    output_model = Output
```

**After (v0.4) - Recommended approach:**
```python
# Use BasePrompt classes instead - see Prompt System Migration section
from ragas.metrics.collections.faithfulness.util import FaithfulnessPrompt

class CustomPrompt(FaithfulnessPrompt):
    @property
    def instruction(self):
        return "Your custom instruction here"
```

### Legacy Metric Methods

#### `single_turn_ascore(sample)` - Deprecated

- **Status**: Only on legacy (non-collections) metrics
- **Deprecation**: Use collections metrics with `ascore()` instead
- **Timeline**: Will be removed in future releases when all metrics migrate

**Before (v0.3) - Deprecated:**
```python
sample = SingleTurnSample(user_input="...", response="...", ...)
score = await metric.single_turn_ascore(sample)
```

**After (v0.4) - Recommended:**
```python
result = await metric.ascore(user_input="...", response="...")
score = result.value
```

#### ContextUtilization

`ContextUtilization` is now a wrapper around `ContextPrecisionWithoutReference` for backward compatibility:

**Before (v0.3):**
```python
from ragas.metrics import ContextUtilization
metric = ContextUtilization(llm=llm)
score = await metric.single_turn_ascore(sample)
```

**After (v0.4):**
```python
from ragas.metrics.collections import ContextUtilization
# or use the modern name directly:
from ragas.metrics.collections import ContextPrecisionWithoutReference

metric = ContextUtilization(llm=llm)  # Still works (wrapper)
# or
metric = ContextPrecisionWithoutReference(llm=llm)  # Preferred

result = await metric.ascore(
    user_input="...",
    response="...",
    retrieved_contexts=[...]
)
score = result.value
```

---

## Breaking Changes Summary

Here's a complete list of breaking changes between v0.3 and v0.4:

| Change | v0.3 | v0.4 | Migration |
|--------|------|------|-----------|
| **Evaluation approach** | `evaluate()` function | `@experiment()` decorator | See [Evaluation to Experiment](#evaluation-to-experiment) |
| **Metrics location** | `ragas.metrics` | `ragas.metrics.collections` | Update import paths |
| **Scoring method** | `single_turn_ascore(sample)` | `ascore(**kwargs)` | Change method calls |
| **Score return type** | `float` | `MetricResult` | Use `.value` property |
| **LLM factory** | `instructor_llm_factory()` | `llm_factory()` | Use unified factory |
| **Embeddings approach** | Wrapper classes (LangChain) | Native providers | See [Embeddings Migration](#embeddings-migration) |
| **Embedding methods** | `embed_query()`, `embed_documents()` | `embed_text()`, `embed_texts()` | Update method calls |
| **ground_truths param** | `ground_truths: list[str]` | `reference: str` | Rename, change type |
| **Sample type** | `SingleTurnSample` | `SingleTurnSample` (updated) | Update sample creation |
| **Prompt system** | Dataclass-based | Function-based | Refactor custom prompts |

---

## Deprecations and Removals

### Removed in v0.4

These features have been completely removed and will cause errors:

- **`instructor_llm_factory()`** - Use `llm_factory()` instead
- **AspectCritic** from collections - No direct replacement
- **SimpleCriteriaScore** from collections - No direct replacement
- **AnswerSimilarity** - Use `SemanticSimilarity` instead

### Deprecated (Will be removed in future releases)

These features still work but show deprecation warnings:

- **`LangchainLLMWrapper`** - Use `llm_factory()` directly
- **`LlamaIndexLLMWrapper`** - Use `llm_factory()` directly
- **Legacy prompt classes** - Migrate to function-based prompts
- **`single_turn_ascore()`** on legacy metrics - Use collections metrics with `ascore()`

---

## New Features in v0.4 (Reference)

v0.4 introduces several new capabilities beyond the migration requirements. While not necessary for migrating from v0.3, these features may be useful for your upgrade:

- **GPT-5 and o-Series Support** - Automatic constraint handling for latest OpenAI models
- **Universal Provider Support** - Single `llm_factory()` works with all major providers (Anthropic, Google, Azure, etc.)
- **Function-Based Prompts** - More flexible and composable prompt definitions
- **Metric Decorators** - Simplified custom metric creation with `@discrete_metric`, `@numeric_metric`, `@ranking_metric`
- **MetricResult with Reasoning** - Structured results with optional explanations
- **Enhanced Metric Save/Load** - Easy serialization of metric configurations
- **Better Embeddings Support** - Both sync and async embedding operations

For detailed information on new features, see the [v0.4 Release Notes](../../releases/v0.4.0.md).

---

## Custom Metrics Migration

If you were using removed metrics like `AspectCritic` or `SimpleCriteria`, v0.4 provides decorator-based alternatives to replace them. You can also use the new simplified metric system for other custom metrics:

### Discrete Metrics (Categorical Outputs)

**Before (v0.3) - AspectCritic:**
```python
from ragas.metrics import AspectCritic
metric = AspectCritic(name="clarity", allowed_values=["clear", "unclear"])
result = await metric.single_turn_ascore(sample)
```

**After (v0.4) - @discrete_metric decorator:**
```python
from ragas.metrics import discrete_metric

@discrete_metric(name="clarity", allowed_values=["clear", "unclear"])
def clarity(response: str) -> str:
    return "clear" if len(response) > 50 else "unclear"

metric = clarity()
result = await metric.ascore(response="...")
print(result.value)  # "clear" or "unclear"
```

Use discrete metrics for any categorical classification. All removed metrics (AspectCritic, SimpleCriteria) can be replaced this way.

### Numeric Metrics (Continuous Values)

Use `@numeric_metric` for any scoring on a numerical scale:

```python
from ragas.metrics import numeric_metric

@numeric_metric(name="length_score", allowed_values=(0.0, 1.0))
def length_score(response: str) -> float:
    return min(len(response) / 500, 1.0)

# Custom range
@numeric_metric(name="quality_score", allowed_values=(0.0, 10.0))
def quality_score(response: str) -> float:
    return 7.5

metric = length_score()
result = await metric.ascore(response="...")
print(result.value)  # float between 0 and 1
```

### Ranking Metrics (Ordered Lists)

Use `@ranking_metric` to rank or order multiple items:

```python
from ragas.metrics import ranking_metric

@ranking_metric(name="context_rank", allowed_values=5)
def context_ranking(question: str, contexts: list[str]) -> list[str]:
    """Rank contexts by relevance."""
    scored = [(len(set(question.split()) & set(c.split())), c) for c in contexts]
    return [c for _, c in sorted(scored, reverse=True)]

metric = context_ranking()
result = await metric.ascore(question="...", contexts=[...])
print(result.value)  # Ranked list
```

### Summary

These decorators provide automatic validation, type safety, error handling, and result wrapping - reducing custom metric code from 50+ lines in v0.3 to just 5-10 lines in v0.4.

---

## Common Issues and Solutions

### Issue: ImportError for `instructor_llm_factory`

**Error:**
```
ImportError: cannot import name 'instructor_llm_factory' from 'ragas.llms'
```

**Solution:**
```python
# Instead of this
from ragas.llms import instructor_llm_factory

# Use this
from ragas.llms import llm_factory
```

### Issue: Metric Returns `MetricResult` Instead of Float

**Error:**
```python
score = await metric.ascore(...)
print(score)  # Prints: MetricResult(value=0.85, reason=None)
```

**Solution:**
```python
result = await metric.ascore(...)
score = result.value  # Access the float value
print(score)  # Prints: 0.85
```

### Issue: `SingleTurnSample` Missing `ground_truths`

**Error:**
```
TypeError: ground_truths is not a valid keyword
```

**Solution:**
```python
# Change from
sample = SingleTurnSample(..., ground_truths=["correct"])

# To
sample = SingleTurnSample(..., reference="correct")
```

## Getting Help

If you encounter issues during migration:

1. **Check the Documentation**
    - [Metrics Documentation](../../concepts/metrics/available_metrics/index.md)
    - [Collections API](../../concepts/metrics/overview/index.md)
    - [LLM Configuration](../../concepts/llms/index.md)

2. **GitHub Issues**
    - Search [existing issues](https://github.com/explodinggradients/ragas/issues)
    - Create a new issue with migration-specific details

3. **Community Support**
    - [Join our Discord community](https://discord.gg/5djav8GGNZ)
    - [Schedule a call](https://cal.com/shahul-ragas/30min) with the maintainers

---

## Summary

v0.4 represents a fundamental shift towards experiment-based architecture, enabling better integration of evaluation, analysis, and iteration workflows. While there are breaking changes, they all serve the goal of making Ragas a better experimentation platform.

The migration path is straightforward:

1. Update LLM initialization to use `llm_factory()`
2. Import metrics from `ragas.metrics.collections`
3. Replace `single_turn_ascore()` with `ascore()`
4. Rename `ground_truths` to `reference`
5. Handle `MetricResult` objects instead of floats

These technical changes enable:

- **Better Experimentation** - Structured metric results with reasoning for deeper analysis
- **Cleaner API** - Keyword arguments instead of sample objects make composition easier
- **Integrated Workflows** - Metrics designed to work seamlessly within experiment pipelines
- **Enhanced Functionality** - Universal provider support and automatic constraints
- **Future-proof** - Built on industry standards (instructor library, standardized patterns)

The experiment-based architecture will continue to improve in future releases, with more features for managing, analyzing, and iterating on your evaluations.

Good luck with your migration! We're here to help if you get stuck. 🎉
