# AG-UI Integration
Ragas can evaluate agents that stream events via the [AG-UI protocol](https://docs.ag-ui.com/). This notebook shows how to build evaluation datasets, configure metrics, and score AG-UI endpoints.


## Prerequisites
- Install optional dependencies with `pip install "ragas[ag-ui]" python-dotenv nest_asyncio`
- Start an AG-UI compatible agent locally (Google ADK, PydanticAI, CrewAI, etc.)
- Create an `.env` file with your evaluator LLM credentials (e.g. `OPENAI_API_KEY`, `GOOGLE_API_KEY`, etc.)
- If you run this notebook, call `nest_asyncio.apply()` (shown below) so you can `await` coroutines in-place.


```python
# !pip install "ragas[ag-ui]" python-dotenv nest_asyncio
```

## Imports and environment setup
Load environment variables and import the classes used throughout the walkthrough.



```python
import nest_asyncio
from dotenv import load_dotenv
from IPython.display import display

from ragas.dataset_schema import EvaluationDataset, MultiTurnSample, SingleTurnSample
from ragas.integrations.ag_ui import (
    convert_messages_snapshot,
    convert_to_ragas_messages,
    evaluate_ag_ui_agent,
)
from ragas.messages import HumanMessage, ToolCall

load_dotenv()
# Patch the existing notebook loop so we can await coroutines safely
nest_asyncio.apply()
```

## Build single-turn evaluation data
Create `SingleTurnSample` entries when you only need to grade the final answer text.



```python
scientist_questions = EvaluationDataset(
    samples=[
        SingleTurnSample(
            user_input="Who originated the theory of relativity?",
            reference="Albert Einstein originated the theory of relativity.",
        ),
        SingleTurnSample(
            user_input="Who discovered penicillin and when?",
            reference="Alexander Fleming discovered penicillin in 1928.",
        ),
    ]
)

scientist_questions
```




    EvaluationDataset(features=['user_input', 'reference'], len=2)



## Build multi-turn conversations

For tool-usage and goal accuracy metrics, use `MultiTurnSample` with:
- `reference_tool_calls`: Expected tool calls for `ToolCallF1`
- `reference`: Expected outcome description for `AgentGoalAccuracyWithReference`


```python
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

weather_queries
```




    EvaluationDataset(features=['user_input', 'reference', 'reference_tool_calls'], len=2)



## Configure metrics and the evaluator LLM

For single-turn Q&A evaluation, we use:
- `FactualCorrectness`: Compares response facts against reference
- `AnswerRelevancy`: Measures how relevant the response is to the question
- `DiscreteMetric`: Custom metric for conciseness

For multi-turn agent evaluation, we use:
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

# Metrics for single-turn Q&A evaluation
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

# Metrics for multi-turn agent evaluation
# - ToolCallF1: Rule-based metric for tool call accuracy
# - AgentGoalAccuracyWithReference: LLM-based metric for goal achievement
tool_metrics = [
    ToolCallF1(),
    AgentGoalAccuracyWithReference(llm=evaluator_llm),
]
```

## Evaluate a live AG-UI endpoint
Set the endpoint URL exposed by your agent. Toggle the flags when you are ready to run the evaluations.
In Jupyter/IPython you can `await` the helpers directly once `nest_asyncio.apply()` has been called.



```python
AG_UI_ENDPOINT = "http://localhost:8000"  # Update to match your agent

RUN_FACTUAL_EVAL = True
RUN_TOOL_EVAL = True
```


```python
async def evaluate_factual():
    return await evaluate_ag_ui_agent(
        endpoint_url=AG_UI_ENDPOINT,
        dataset=scientist_questions,
        metrics=qa_metrics,
        evaluator_llm=evaluator_llm,
        metadata=True,
    )


if RUN_FACTUAL_EVAL:
    factual_result = await evaluate_factual()
    factual_df = factual_result.to_pandas()
    display(factual_df)
```


    Calling AG-UI Agent:   0%|          | 0/2 [00:00<?, ?it/s]


    Query 0 - Agent returned no tool/context messages; using placeholder.
    Query 1 - Agent returned no tool/context messages; using placeholder.



    Evaluating:   0%|          | 0/6 [00:00<?, ?it/s]



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
      <th>reference</th>
      <th>factual_correctness</th>
      <th>answer_relevancy</th>
      <th>conciseness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Who originated the theory of relativity?</td>
      <td>[[no retrieved contexts provided by agent]]</td>
      <td>Albert Einstein originated the theory of relat...</td>
      <td>Albert Einstein originated the theory of relat...</td>
      <td>1.0</td>
      <td>0.999999</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Who discovered penicillin and when?</td>
      <td>[[no retrieved contexts provided by agent]]</td>
      <td>Hello, Penicillin was discovered in 1928 by Al...</td>
      <td>Alexander Fleming discovered penicillin in 1928.</td>
      <td>1.0</td>
      <td>0.986753</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
async def evaluate_tool_usage():
    return await evaluate_ag_ui_agent(
        endpoint_url=AG_UI_ENDPOINT,
        dataset=weather_queries,
        metrics=tool_metrics,
        evaluator_llm=evaluator_llm,
    )


if RUN_TOOL_EVAL:
    tool_result = await evaluate_tool_usage()
    tool_df = tool_result.to_pandas()
    display(tool_df)
```


    Calling AG-UI Agent:   0%|          | 0/2 [00:00<?, ?it/s]


    ToolCallResult received but no AIMessage found. Creating synthetic AIMessage.
    ToolCallResult received but no AIMessage found. Creating synthetic AIMessage.



    Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>reference</th>
      <th>reference_tool_calls</th>
      <th>tool_call_f1</th>
      <th>agent_goal_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'content': 'What's the weather in Paris?', '...</td>
      <td>The user received the current weather conditio...</td>
      <td>[{'name': 'get_weather', 'args': {'location': ...</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[{'content': 'Is it raining in London right no...</td>
      <td>The user received the current weather conditio...</td>
      <td>[{'name': 'get_weather', 'args': {'location': ...</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

