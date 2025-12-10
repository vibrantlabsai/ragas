# AG-UI

[AG-UI](https://docs.ag-ui.com/) is an event-based protocol for streaming agent updates to user interfaces. The protocol standardizes message, tool-call, and state events, which makes it easy to plug different agent runtimes into visual frontends. The `ragas.integrations.ag_ui` module helps you transform those event streams into Ragas message objects and run experiments against live AG-UI endpoints using the modern `@experiment` decorator pattern.

This guide assumes you already have an AG-UI compatible agent running (for example, one built with Google ADK, PydanticAI, or CrewAI) and that you are familiar with creating datasets in Ragas.

## Install the integration

The AG-UI helpers live behind an optional extra. Install it together with the dependencies required by your evaluator LLM. When running inside Jupyter or IPython, include `nest_asyncio` so you can reuse the notebook's event loop.

```bash
pip install "ragas[ag-ui]" python-dotenv nest_asyncio
```

Configure your evaluator LLM credentials. For example, if you are using OpenAI models:

```bash
# .env
OPENAI_API_KEY=sk-...
```

Load the environment variables inside Python before running the examples:

```python
from dotenv import load_dotenv
import nest_asyncio

load_dotenv()

# If you're inside Jupyter/IPython, patch the running event loop once.
nest_asyncio.apply()
```

## Build an experiment dataset

`Dataset` can contain single-turn or multi-turn samples. With AG-UI you can test either pattern—single questions with free-form responses, or longer conversations that include tool calls.

### Single-turn samples

Use `Dataset.from_pandas()` with `user_input` and `reference` columns when you only need to grade the final answer text.

```python
import pandas as pd
from ragas.dataset import Dataset

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
```

### Multi-turn samples with tool expectations

When you want to grade intermediate agent behavior—like whether it calls tools correctly and achieves the user's goal—use conversation lists as `user_input`. Provide expected tool calls as JSON and optionally a reference outcome for goal accuracy evaluation.

```python
import json
import pandas as pd
from ragas.dataset import Dataset
from ragas.messages import HumanMessage

weather_queries = Dataset.from_pandas(
    pd.DataFrame([
        {
            "user_input": [HumanMessage(content="What's the weather in Paris?")],
            "reference_tool_calls": json.dumps([
                {"name": "get_weather", "args": {"location": "Paris"}}
            ]),
            # Expected outcome for AgentGoalAccuracyWithReference
            "reference": "The user received the current weather conditions for Paris.",
        },
        {
            "user_input": [HumanMessage(content="Is it raining in London right now?")],
            "reference_tool_calls": json.dumps([
                {"name": "get_weather", "args": {"location": "London"}}
            ]),
            "reference": "The user received the current weather conditions for London.",
        },
    ]),
    name="weather_queries",
    backend="inmemory",
)
```

### Loading from CSV

For larger datasets, store your test cases in CSV files and load them with the Dataset API:

```python
from ragas.dataset import Dataset

dataset = Dataset.load(
    name="scientist_biographies",
    backend="local/csv",
    root_dir="./test_data",
)
```

## Choose metrics and evaluator model

The integration works with any Ragas metric. To unlock the modern collections portfolio (and mix in custom checks), build an Instructor-compatible LLM for the evaluator prompts and use a synchronous OpenAI client for embeddings.

```python
from openai import AsyncOpenAI, OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import (
    AgentGoalAccuracyWithReference,
    AnswerRelevancy,
    FactualCorrectness,
    ToolCallF1,
)

async_llm_client = AsyncOpenAI()
evaluator_llm = llm_factory("gpt-4o-mini", client=async_llm_client)

# AnswerRelevancy's embeddings still run synchronously, so pair it with a sync client.
embedding_client = OpenAI()
evaluator_embeddings = embedding_factory(
    "openai", model="text-embedding-3-small", client=embedding_client, interface="modern"
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

# Metrics for single-turn Q&A evaluation
qa_metrics = [
    FactualCorrectness(
        llm=evaluator_llm, mode="f1", atomicity="high", coverage="high"
    ),
    AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=2),
    conciseness_metric,
]

# Metrics for multi-turn agent evaluation
# - ToolCallF1: Rule-based metric for tool call accuracy
# - AgentGoalAccuracyWithReference: LLM-based metric for goal achievement
tool_metrics = [
    ToolCallF1(),
    AgentGoalAccuracyWithReference(llm=evaluator_llm),
]
```

## Run experiments with @experiment

The AG-UI integration provides `run_ag_ui_row()` to call your endpoint and enrich each row with the agent's response. Combine this with the `@experiment` decorator to build evaluation pipelines.

> ⚠️ The endpoint must expose the AG-UI SSE stream. Common paths include `/chat`, `/agent`, or `/agentic_chat`.

### Basic single-turn evaluation

In Jupyter or IPython, use top-level `await` (after `nest_asyncio.apply()`) instead of `asyncio.run` to avoid the "event loop is already running" error. For scripts you can keep `asyncio.run`.

```python
from ragas import experiment
from ragas.integrations.ag_ui import run_ag_ui_row
from ragas.metrics.collections import FactualCorrectness

@experiment()
async def factual_experiment(row):
    # Call AG-UI endpoint and get enriched row
    enriched = await run_ag_ui_row(row, "http://localhost:8000/chat")

    # Score with metrics
    score = await FactualCorrectness(llm=evaluator_llm).ascore(
        response=enriched["response"],
        reference=row["reference"],
    )

    return {**enriched, "factual_correctness": score.value}

# Run the experiment against the dataset
# In Jupyter/IPython (after calling nest_asyncio.apply())
factual_result = await factual_experiment.arun(
    scientist_questions,
    name="scientist_qa_eval"
)

# In a standalone script, use:
# factual_result = asyncio.run(factual_experiment.arun(scientist_questions, name="scientist_qa_eval"))

factual_result.to_pandas()
```

The resulting dataframe includes per-sample scores, raw agent responses, and any retrieved contexts (tool results). Results are automatically saved by the experiment framework, and you can export to CSV through pandas.

### Multi-turn tool evaluation

For multi-turn datasets and tool evaluation, pass the messages and reference tool calls directly to the metrics:

```python
import json
from ragas import experiment
from ragas.integrations.ag_ui import run_ag_ui_row
from ragas.messages import ToolCall
from ragas.metrics.collections import AgentGoalAccuracyWithReference, ToolCallF1

@experiment()
async def tool_experiment(row):
    # Call AG-UI endpoint and get enriched row
    enriched = await run_ag_ui_row(row, "http://localhost:8000/chat")

    # Parse reference_tool_calls from JSON string (e.g., from CSV)
    ref_tool_calls_raw = row.get("reference_tool_calls")
    if isinstance(ref_tool_calls_raw, str):
        ref_tool_calls = [ToolCall(**tc) for tc in json.loads(ref_tool_calls_raw)]
    else:
        ref_tool_calls = ref_tool_calls_raw or []

    # Score with tool metrics using the modern collections API
    f1_result = await ToolCallF1().ascore(
        user_input=enriched["messages"],
        reference_tool_calls=ref_tool_calls,
    )
    goal_result = await AgentGoalAccuracyWithReference(llm=evaluator_llm).ascore(
        user_input=enriched["messages"],
        reference=row.get("reference", ""),
    )

    return {
        **enriched,
        "tool_call_f1": f1_result.value,
        "agent_goal_accuracy": goal_result.value,
    }

# Run the experiment
# In Jupyter/IPython
tool_result = await tool_experiment.arun(
    weather_queries,
    name="weather_tool_eval"
)

# Or in a script
# tool_result = asyncio.run(tool_experiment.arun(weather_queries, name="weather_tool_eval"))

tool_result.to_pandas()
```

If a request fails, the experiment logs the error and returns placeholder values for that sample so the experiment can continue with remaining samples.

## Working directly with AG-UI events

Sometimes you may want to collect event logs separately—perhaps from a recorded run or a staging environment—and evaluate them offline. The conversion helpers expose the same parsing logic used by `run_ag_ui_row()`.

```python
from ragas.integrations.ag_ui import convert_to_ragas_messages
from ag_ui.core import TextMessageChunkEvent

events = [
    TextMessageChunkEvent(
        message_id="assistant-1",
        role="assistant",
        delta="Hello from AG-UI!",
        timestamp="2024-12-01T00:00:00Z",
    )
]

ragas_messages = convert_to_ragas_messages(events, metadata=True)
```

If you already have a `MessagesSnapshotEvent` you can skip streaming reconstruction and call `convert_messages_snapshot`.

```python
from ragas.integrations.ag_ui import convert_messages_snapshot
from ag_ui.core import MessagesSnapshotEvent, UserMessage, AssistantMessage

snapshot = MessagesSnapshotEvent(
    messages=[
        UserMessage(id="msg-1", content="Hello?"),
        AssistantMessage(id="msg-2", content="Hi! How can I help you today?"),
    ]
)

ragas_messages = convert_messages_snapshot(snapshot)
```

The converted messages can be used to build custom evaluation workflows or passed directly to metric scoring functions.

## Extraction helpers

The integration provides helper functions to extract specific data from messages:

```python
from ragas.integrations.ag_ui import (
    extract_response,    # Get concatenated AI response text
    extract_tool_calls,  # Get all tool calls from AI messages
    extract_contexts,    # Get tool results/contexts
)

messages = convert_to_ragas_messages(events)

response = extract_response(messages)      # "Hello! The weather is sunny."
tool_calls = extract_tool_calls(messages)  # [ToolCall(name="get_weather", args={"location": "SF"})]
contexts = extract_contexts(messages)      # ["Sunny, 72F in San Francisco"]
```

## Tips for production experiments

- **Custom headers**: pass authentication tokens or tenant IDs via `extra_headers` parameter to `run_ag_ui_row()`.
- **Timeouts**: tune the `timeout` parameter if your agent performs long-running tool calls.
- **Metadata debugging**: set `metadata=True` to keep AG-UI run, thread, and message IDs on every message for easier traceability.
- **Experiment naming**: use descriptive `name` arguments to `.arun()` for easy identification of results.

For a complete production example, see `examples/ragas_examples/ag_ui_agent_experiments/experiments.py` which provides:

- CLI arguments for endpoint configuration
- CSV-based test datasets
- Proper logging and error handling
- Timestamped result output

An interactive walkthrough notebook is also available at `howtos/integrations/ag_ui.ipynb`.

## API Reference

### Primary API

- **`run_ag_ui_row(row, endpoint_url, ...)`** - Run a single row against an AG-UI endpoint and return enriched data with response, messages, tool_calls, and contexts.

### Conversion Functions

- **`convert_to_ragas_messages(events, metadata=False)`** - Convert AG-UI event sequences to Ragas messages
- **`convert_messages_snapshot(snapshot, metadata=False)`** - Convert AG-UI message snapshots to Ragas messages
- **`convert_messages_to_ag_ui(messages)`** - Convert Ragas messages to AG-UI format

### Extraction Helpers

- **`extract_response(messages)`** - Extract concatenated AI response text
- **`extract_tool_calls(messages)`** - Extract all tool calls from AI messages
- **`extract_contexts(messages)`** - Extract tool results/contexts from messages

### Low-Level

- **`call_ag_ui_endpoint(endpoint_url, user_input, ...)`** - Call an AG-UI endpoint and collect streaming events
- **`AGUIEventCollector`** - Collect and reconstruct messages from streaming events
