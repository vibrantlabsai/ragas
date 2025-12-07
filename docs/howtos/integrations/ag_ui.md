# AG-UI

[AG-UI](https://docs.ag-ui.com/) is an event-based protocol for streaming agent updates to user interfaces. The protocol standardizes message, tool-call, and state events, which makes it easy to plug different agent runtimes into visual frontends. The `ragas.integrations.ag_ui` module helps you transform those event streams into Ragas message objects and evaluate live AG-UI endpoints with the same metrics used across the rest of the Ragas ecosystem.

This guide assumes you already have an AG-UI compatible agent running (for example, one built with Google ADK, PydanticAI, or CrewAI) and that you are familiar with creating evaluation datasets in Ragas.

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

## Build an evaluation dataset

`EvaluationDataset` can contain single-turn or multi-turn samples. With AG-UI you can evaluate either pattern—single questions with free-form responses, or longer conversations that can include tool calls.

### Single-turn samples

Use `SingleTurnSample` when you only need the final answer text.

```python
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

scientist_questions = EvaluationDataset(
    samples=[
        SingleTurnSample(
            user_input="Who originated the theory of relativity?",
            reference="Albert Einstein originated the theory of relativity."
        ),
        SingleTurnSample(
            user_input="Who discovered penicillin and when?",
            reference="Alexander Fleming discovered penicillin in 1928."
        ),
    ]
)
```

### Multi-turn samples with tool expectations

When you want to grade intermediate agent behavior—like whether it calls tools correctly and achieves the user's goal—switch to `MultiTurnSample`. Provide an initial conversation history, expected tool calls, and optionally a reference outcome for goal accuracy evaluation.

```python
from ragas.dataset_schema import EvaluationDataset, MultiTurnSample
from ragas.messages import HumanMessage, ToolCall

weather_queries = EvaluationDataset(
    samples=[
        MultiTurnSample(
            user_input=[HumanMessage(content="What's the weather in Paris?")],
            reference_tool_calls=[
                ToolCall(name="get_weather", args={"location": "Paris"})
            ],
            # Expected outcome for AgentGoalAccuracyWithReference
            # Use outcome-focused language that matches what the LLM extracts as end_state
            reference="The user received the current weather conditions for Paris.",
        ),
        MultiTurnSample(
            user_input=[HumanMessage(content="Is it raining in London right now?")],
            reference_tool_calls=[
                ToolCall(name="get_weather", args={"location": "London"})
            ],
            reference="The user received the current weather conditions for London.",
        ),
    ]
)
```

## Choose metrics and evaluator model

The integration works with any Ragas metric. To unlock the modern collections portfolio (and mix in custom checks), build an Instructor-compatible LLM for the evaluator prompts and use a synchronous OpenAI client for embeddings that still rely on blocking calls.

```python
from openai import AsyncOpenAI, OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics import AgentGoalAccuracyWithReference, DiscreteMetric, ToolCallF1
from ragas.metrics.collections import AnswerRelevancy, FactualCorrectness

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

## Evaluate a live AG-UI endpoint

`evaluate_ag_ui_agent` calls your FastAPI endpoint, captures the AG-UI Server-Sent Events (SSE) stream, converts those events into Ragas messages, and runs the metrics you selected.

> ⚠️ The endpoint must expose the AG-UI SSE stream. Common paths include `/chat`, `/agent`, or `/agentic_chat`.

### Evaluate factual responses

In Jupyter or IPython, use top-level `await` (after `nest_asyncio.apply()`) instead of `asyncio.run` to avoid the "event loop is already running" error. For scripts you can keep `asyncio.run`.

```python
import asyncio
from ragas.integrations.ag_ui import evaluate_ag_ui_agent

async def run_factual_eval():
    result = await evaluate_ag_ui_agent(
        endpoint_url="http://localhost:8000/agentic_chat",
        dataset=scientist_questions,
        metrics=qa_metrics,
        evaluator_llm=evaluator_llm,
        metadata=True,  # optional, keeps run/thread metadata on messages
    )
    return result

# In Jupyter/IPython (after calling nest_asyncio.apply())
factual_result = await run_factual_eval()

# In a standalone script, use:
# factual_result = asyncio.run(run_factual_eval())
factual_result.to_pandas()
```

The resulting dataframe includes per-sample scores, raw agent responses, and any retrieved contexts (if provided by the agent). You can save it with `result.save()` or export to CSV through pandas.

### Evaluate tool usage

The same function supports multi-turn datasets. Agent responses (AI messages and tool outputs) are appended to the existing conversation before scoring.

```python
async def run_tool_eval():
    result = await evaluate_ag_ui_agent(
        endpoint_url="http://localhost:8000/agentic_chat",
        dataset=weather_queries,
        metrics=tool_metrics,
        evaluator_llm=evaluator_llm,
    )
    return result

# In Jupyter/IPython
tool_result = await run_tool_eval()

# Or in a script
# tool_result = asyncio.run(run_tool_eval())
tool_result.to_pandas()
```

If a request fails, the executor logs the error and marks the corresponding sample with `NaN` scores so you can retry or inspect the endpoint logs.

## Working directly with AG-UI events

Sometimes you may want to collect event logs separately—perhaps from a recorded run or a staging environment—and evaluate them offline. The conversion helpers expose the same parsing logic used by `evaluate_ag_ui_agent`.

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

The converted messages can be plugged into `EvaluationDataset` objects or passed directly to lower-level Ragas evaluation APIs if you need custom workflows.

## Tips for production evaluations

- **Batch size**: use the `batch_size` argument to control parallel requests to your agent.
- **Custom headers**: pass authentication tokens or tenant IDs via `extra_headers`.
- **Timeouts**: tune the `timeout` parameter if your agent performs long-running tool calls.
- **Metadata debugging**: set `metadata=True` to keep AG-UI run, thread, and message IDs on every `RagasMessage` for easier traceability.

Once you are satisfied with your scoring setup, consider wrapping the snippets in a script or notebook. An example walkthrough notebook is available at `docs/howtos/integrations/ag_ui.ipynb`.
