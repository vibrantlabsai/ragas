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
from ragas.integrations.ag_ui import create_ag_ui_experiment
from ragas.messages import HumanMessage

load_dotenv()
# Patch the existing notebook loop so we can await coroutines safely
nest_asyncio.apply()
```

## Build single-turn experiment data
Create dataset entries with `user_input` and `reference` using `Dataset.from_pandas()` when you only need to grade the final answer text.


```python
scientist_questions = Dataset.from_pandas(
    pd.DataFrame([
        {
            "user_input": "Who originated the theory of relativity?",
            "reference": "Albert Einstein originated the theory of relativity.",
        },
        {
            "user_input": "Who discovered penicillin and when?",
            "reference": "Alexander Fleming discovered penicillin in 1928.",
        },
    ]),
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
    pd.DataFrame([
        {
            "user_input": [HumanMessage(content="What's the weather in Paris?")],
            "reference_tool_calls": json.dumps([
                {"name": "get_weather", "args": {"location": "Paris"}}
            ]),
            # Expected outcome - phrased to match what LLM extracts as end_state
            "reference": "The AI provided the current weather conditions for Paris.",
        },
        {
            "user_input": [HumanMessage(content="Is it raining in London right now?")],
            "reference_tool_calls": json.dumps([
                {"name": "get_weather", "args": {"location": "London"}}
            ]),
            "reference": "The AI provided the current weather conditions for London.",
        },
    ]),
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
from openai import AsyncOpenAI, OpenAI

from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics import AgentGoalAccuracyWithReference, DiscreteMetric, ToolCallF1
from ragas.metrics.collections import AnswerRelevancy, FactualCorrectness

# Async client for evaluator prompts
async_llm_client = AsyncOpenAI()
evaluator_llm = llm_factory("gpt-4o-mini", client=async_llm_client)

# Sync client for embeddings (AnswerRelevancy still makes blocking calls)
embedding_client = OpenAI()
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
Set the endpoint URL exposed by your agent. The `create_ag_ui_experiment` factory returns an experiment function that can be run against datasets using `.arun()`.

Toggle the flags when you are ready to run the experiments. In Jupyter/IPython you can `await` the experiment directly once `nest_asyncio.apply()` has been called.


```python
AG_UI_ENDPOINT = "http://localhost:8000"  # Update to match your agent

RUN_FACTUAL_EXPERIMENT = True
RUN_TOOL_EXPERIMENT = True
```


```python
if RUN_FACTUAL_EXPERIMENT:
    # Create experiment function configured for Q&A testing
    factual_experiment = create_ag_ui_experiment(
        endpoint_url=AG_UI_ENDPOINT,
        metrics=qa_metrics,
        evaluator_llm=evaluator_llm,
        metadata=True,
    )
    
    # Run the experiment against the dataset
    factual_result = await factual_experiment.arun(
        scientist_questions,
        name="scientist_qa_experiment"
    )
    display(factual_result.to_pandas())
```


```python
if RUN_TOOL_EXPERIMENT:
    # Create experiment function configured for tool usage testing
    tool_experiment = create_ag_ui_experiment(
        endpoint_url=AG_UI_ENDPOINT,
        metrics=tool_metrics,
        evaluator_llm=evaluator_llm,
    )
    
    # Run the experiment against the dataset
    tool_result = await tool_experiment.arun(
        weather_queries,
        name="weather_tool_experiment"
    )
    display(tool_result.to_pandas())
```


```python

```
