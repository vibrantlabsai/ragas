# AG-UI Integration
Ragas can run experiments on agents that stream events via the [AG-UI protocol](https://docs.ag-ui.com/). This notebook shows how to build experiment datasets, configure metrics, and score AG-UI endpoints using the modern `@experiment` decorator pattern.

## Prerequisites
- Install dependencies: `pip install "ragas[ag-ui]" python-dotenv nest_asyncio`
- Start an AG-UI compatible agent locally (Google ADK, PydanticAI, CrewAI, etc.)
- Create an `.env` file with your evaluator LLM credentials (e.g. `OPENAI_API_KEY`, `GOOGLE_API_KEY`, etc.)
- If you run this notebook, call `nest_asyncio.apply()` (shown below) so you can `await` coroutines in-place.


```python
# !pip install "ragas[ag-ui]" python-dotenv nest_asyncio
```

## Imports and environment setup
Load environment variables and import the classes used throughout the walkthrough.


```python
import json

import nest_asyncio
import pandas as pd
from dotenv import load_dotenv
from IPython.display import display

from ragas.dataset import Dataset
from ragas.messages import HumanMessage

load_dotenv()
# Patch the existing notebook loop so we can await coroutines safely
nest_asyncio.apply()
```

## Build single-turn experiment data
Create dataset entries with `user_input` and `reference` using `Dataset.from_pandas()` when you only need to grade the final answer text.


```python
scientist_questions = Dataset.from_pandas(
    pd.DataFrame(
        [
            {
                "user_input": "Who originated the theory of relativity?",
                "reference": "Albert Einstein originated the theory of relativity.",
            },
            {
                "user_input": "Who discovered penicillin and when?",
                "reference": "Alexander Fleming discovered penicillin in 1928.",
            },
        ]
    ),
    name="scientist_questions",
    backend="inmemory",
)

scientist_questions
```

## Build multi-turn conversations

For tool-usage and goal accuracy metrics, provide:
- `reference_tool_calls`: Expected tool calls as JSON for `ToolCallF1`
- `reference`: Expected outcome description for `AgentGoalAccuracyWithReference`


```python
weather_queries = Dataset.from_pandas(
    pd.DataFrame(
        [
            {
                "user_input": [HumanMessage(content="What's the weather in Paris?")],
                "reference_tool_calls": json.dumps(
                    [{"name": "get_weather", "args": {"location": "Paris"}}]
                ),
                # Expected outcome - phrased to match what LLM extracts as end_state
                "reference": "The AI provided the current weather conditions for Paris.",
            },
            {
                "user_input": [
                    HumanMessage(content="Is it raining in London right now?")
                ],
                "reference_tool_calls": json.dumps(
                    [{"name": "get_weather", "args": {"location": "London"}}]
                ),
                "reference": "The AI provided the current weather conditions for London.",
            },
        ]
    ),
    name="weather_queries",
    backend="inmemory",
)

weather_queries
```

## Configure metrics and the evaluator LLM

For single-turn Q&A experiments, we use:
- `FactualCorrectness`: Compares response facts against reference
- `AnswerRelevancy`: Measures how relevant the response is to the question
- `DiscreteMetric`: Custom metric for conciseness

For multi-turn agent experiments, we use:
- `ToolCallF1`: Rule-based metric comparing actual vs expected tool calls
- `AgentGoalAccuracyWithReference`: LLM-based metric evaluating whether the agent achieved the user's goal


```python
from openai import AsyncOpenAI

from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import (
    AgentGoalAccuracyWithReference,
    AnswerRelevancy,
    FactualCorrectness,
    ToolCallF1,
)

# Async client for evaluator prompts
async_llm_client = AsyncOpenAI()
evaluator_llm = llm_factory("gpt-4o-mini", client=async_llm_client)

embedding_client = AsyncOpenAI()
evaluator_embeddings = embedding_factory(
    "openai",
    model="text-embedding-3-small",
    client=embedding_client,
    interface="modern",
)

conciseness_metric = DiscreteMetric(
    name="conciseness",
    allowed_values=["verbose", "concise"],
    prompt=(
        "Is the response concise and efficiently conveys information?\n\n"
        "Response: {response}\n\n"
        "Answer with only 'verbose' or 'concise'."
    ),
)

# Metrics for single-turn Q&A experiments
qa_metrics = [
    FactualCorrectness(
        llm=evaluator_llm,
        mode="f1",
        atomicity="high",
        coverage="high",
    ),
    AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        strictness=2,
    ),
    conciseness_metric,
]

# Metrics for multi-turn agent experiments
# - ToolCallF1: Rule-based metric for tool call accuracy
# - AgentGoalAccuracyWithReference: LLM-based metric for goal achievement
tool_metrics = [
    ToolCallF1(),
    AgentGoalAccuracyWithReference(llm=evaluator_llm),
]
```

## Run experiments against a live AG-UI endpoint
Set the endpoint URL exposed by your agent. The `run_ag_ui_row()` function calls your endpoint and returns enriched row data. Combine this with the `@experiment` decorator for evaluation pipelines.

Toggle the flags when you are ready to run the experiments. In Jupyter/IPython you can `await` the experiment directly once `nest_asyncio.apply()` has been called.


```python
AG_UI_ENDPOINT = "http://localhost:8000"  # Update to match your agent

RUN_FACTUAL_EXPERIMENT = True
RUN_TOOL_EXPERIMENT = True
```


```python
from ragas import experiment
from ragas.integrations.ag_ui import run_ag_ui_row


@experiment()
async def factual_experiment(row):
    """Single-turn Q&A experiment with factual correctness scoring."""
    # Call AG-UI endpoint and get enriched row
    enriched = await run_ag_ui_row(row, AG_UI_ENDPOINT, metadata=True)

    # Score with factual correctness metric
    fc_result = await qa_metrics[0].ascore(
        response=enriched["response"],
        reference=row["reference"],
    )

    # Score with answer relevancy metric
    ar_result = await qa_metrics[1].ascore(
        user_input=row["user_input"],
        response=enriched["response"],
    )

    # Score with conciseness metric
    concise_result = await conciseness_metric.ascore(
        response=enriched["response"],
        llm=evaluator_llm,
    )

    return {
        **enriched,
        "factual_correctness": fc_result.value,
        "answer_relevancy": ar_result.value,
        "conciseness": concise_result.value,
    }


if RUN_FACTUAL_EXPERIMENT:
    # Run the experiment against the dataset
    factual_result = await factual_experiment.arun(
        scientist_questions, name="scientist_qa_experiment"
    )
    display(factual_result.to_pandas())
```


```python
from ragas.messages import ToolCall


@experiment()
async def tool_experiment(row):
    """Multi-turn experiment with tool call and goal accuracy scoring."""
    # Call AG-UI endpoint and get enriched row
    enriched = await run_ag_ui_row(row, AG_UI_ENDPOINT)

    # Parse reference_tool_calls from JSON string (e.g., from CSV)
    ref_tool_calls_raw = row.get("reference_tool_calls")
    if isinstance(ref_tool_calls_raw, str):
        ref_tool_calls = [ToolCall(**tc) for tc in json.loads(ref_tool_calls_raw)]
    else:
        ref_tool_calls = ref_tool_calls_raw or []

    # Score with tool metrics using the modern collections API
    f1_result = await tool_metrics[0].ascore(
        user_input=enriched["messages"],
        reference_tool_calls=ref_tool_calls,
    )
    goal_result = await tool_metrics[1].ascore(
        user_input=enriched["messages"],
        reference=row.get("reference", ""),
    )

    return {
        **enriched,
        "tool_call_f1": f1_result.value,
        "agent_goal_accuracy": goal_result.value,
    }


if RUN_TOOL_EXPERIMENT:
    # Run the experiment against the dataset
    tool_result = await tool_experiment.arun(
        weather_queries, name="weather_tool_experiment"
    )
    display(tool_result.to_pandas())
```

## Advanced: Lower-Level Control

The `run_ag_ui_row()` function is the recommended API, but sometimes you need more control. You can use the lower-level `call_ag_ui_endpoint()` function directly.

This approach lets you:
- Customize event handling
- Add per-row endpoint configuration  
- Implement custom message processing
- Add additional logging or debugging


```python
from ragas.integrations.ag_ui import (
    call_ag_ui_endpoint,
    convert_to_ragas_messages,
    extract_response,
)


@experiment()
async def custom_ag_ui_experiment(row):
    """
    Custom experiment function with full control over endpoint calls.
    """
    # Call the AG-UI endpoint directly (lower-level than run_ag_ui_row)
    events = await call_ag_ui_endpoint(
        endpoint_url=AG_UI_ENDPOINT,
        user_input=row["user_input"],
        timeout=60.0,
    )

    # Convert AG-UI events to Ragas messages
    messages = convert_to_ragas_messages(events, metadata=True)

    # Extract response using helper (or custom logic)
    response = extract_response(messages)

    # Score with a custom metric
    score_result = await conciseness_metric.ascore(
        response=response,
        llm=evaluator_llm,
    )

    # Return result with custom fields
    return {
        **row,
        "response": response or "[No response]",
        "message_count": len(messages),
        "conciseness": score_result.value,
    }
```

Run the custom experiment against a dataset. The `@experiment` decorator provides `.arun()` for parallel execution and automatic result collection:


```python
RUN_CUSTOM_EXPERIMENT = True

if RUN_CUSTOM_EXPERIMENT:
    # Run the custom experiment
    custom_result = await custom_ag_ui_experiment.arun(
        scientist_questions, name="custom_ag_ui_experiment"
    )
    display(custom_result.to_pandas())
```

### API Comparison

| API Level | Function | When to Use |
|-----------|----------|-------------|
| High-level | `run_ag_ui_row()` | Standard experiments - handles endpoint call, conversion, and extraction |
| Low-level | `call_ag_ui_endpoint()` + `convert_to_ragas_messages()` | Custom event handling, per-row endpoint config, advanced debugging |

Both approaches work with the `@experiment` decorator - choose based on how much control you need.
