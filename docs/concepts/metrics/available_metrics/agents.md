# Agentic or Tool use

Agentic or tool use workflows can be evaluated in multiple dimensions. Here are some of the metrics that can be used to evaluate the performance of agents or tools in a given task.


## Topic Adherence

AI systems deployed in real-world applications are expected to adhere to domains of interest while interacting with users but LLMs sometimes may answer general queries by ignoring this limitation. The topic adherence metric evaluates the ability of the AI to stay on predefined domains during the interactions. This metric is particularly important in conversational AI systems, where the AI is expected to only provide assistance to queries related to predefined domains.

`TopicAdherence` requires a predefined set of topics that the AI system is expected to adhere to which is provided using `reference_topics` along with `user_input`. The metric can compute precision, recall, and F1 score for topic adherence, defined as

$$
\text{Precision } = {|\text{Queries that are answered and are adheres to any present reference topics}| \over |\text{Queries that are answered and are adheres to any present reference topics}| + |\text{Queries that are answered and do not adheres to any present reference topics}|}
$$

$$
\text{Recall } = {|\text{Queries that are answered and are adheres to any present reference topics}| \over |\text{Queries that are answered and are adheres to any present reference topics}| + |\text{Queries that were refused and should have been answered}|}
$$

$$
\text{F1 Score } = {2 \times \text{Precision} \times \text{Recall} \over \text{Precision} + \text{Recall}}
$$

### Example

```python
import asyncio
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import TopicAdherence
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall


async def evaluate_topic_adherence():
    # Setup LLM
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    user_input = [
        HumanMessage(
            content="Can you provide me with details about Einstein's theory of relativity?"
        ),
        AIMessage(
            content="Sure, let me retrieve the relevant information for you.",
            tool_calls=[
                ToolCall(
                    name="document_search",
                    args={"query": "Einstein's theory of relativity"},
                )
            ],
        ),
        ToolMessage(
            content="Found relevant documents: 1. Relativity: The Special and the General Theory, 2. General Theory of Relativity by A. Einstein."
        ),
        AIMessage(
            content="I found some documents on Einstein's theory of relativity. Which one would you like to know more about: 'Relativity: The Special and the General Theory' or 'General Theory of Relativity by A. Einstein'?"
        ),
        HumanMessage(content="Tell me about the 'General Theory of Relativity'."),
        AIMessage(
            content="Got it! Let me fetch more details from 'General Theory of Relativity by A. Einstein'.",
            tool_calls=[
                ToolCall(
                    name="document_retrieve",
                    args={"document": "General Theory of Relativity by A. Einstein"},
                )
            ],
        ),
        ToolMessage(
            content="The document discusses how gravity affects the fabric of spacetime, describing the relationship between mass and spacetime curvature."
        ),
        AIMessage(
            content="The 'General Theory of Relativity' explains how gravity affects the fabric of spacetime and the relationship between mass and spacetime curvature. Would you like more details or a specific explanation?"
        ),
        HumanMessage(
            content="No, that's perfect. By the way, do you know any good recipes for a chocolate cake?"
        ),
        AIMessage(
            content="Sure! Let me find a simple and delicious recipe for a chocolate cake.",
            tool_calls=[
                ToolCall(name="recipe_search", args={"query": "chocolate cake recipe"})
            ],
        ),
        ToolMessage(
            content="Here's a popular recipe for a chocolate cake: Ingredients include flour, sugar, cocoa powder, eggs, milk, and butter. Instructions: Mix dry ingredients, add wet ingredients, and bake at 350°F for 30-35 minutes."
        ),
        AIMessage(
            content="I found a great recipe for chocolate cake! Would you like the full details, or is that summary enough?"
        ),
    ]

    # Evaluate with precision mode
    metric = TopicAdherence(llm=llm, mode="precision")
    result = await metric.ascore(
        user_input=user_input,
        reference_topics=["science"],
    )
    print(f"Topic Adherence (precision): {result.value}")


if __name__ == "__main__":
    asyncio.run(evaluate_topic_adherence())
```
Output
```
Topic Adherence (precision): 0.6666666666444444
```


To change the mode to recall, set the `mode` parameter to `recall`.

```python
metric = TopicAdherence(llm=llm, mode="recall")
```
Output
```
0.99999999995
```

### Legacy API (Deprecated)

!!! warning "Deprecation Notice"
    The legacy `TopicAdherenceScore` from `ragas.metrics` is deprecated and will be removed in v1.0. Please migrate to `ragas.metrics.collections.TopicAdherence` which provides the same functionality with a modern API.

The legacy API can still be used but requires `MultiTurnSample`:

```python
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from ragas.metrics import TopicAdherenceScore  # Legacy import
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

sample = MultiTurnSample(
    user_input=[...],  # conversation messages
    reference_topics=["science"],
)
scorer = TopicAdherenceScore(llm=evaluator_llm, mode="precision")
score = await scorer.multi_turn_ascore(sample)
```



## Tool call Accuracy

`ToolCallAccuracy` measures how accurately an LLM agent invokes tools compared to expected tool calls. It evaluates both the sequence of tool calls and the accuracy of their arguments. This metric is particularly useful for validating that agents call the right tools with the right parameters in multi-step workflows.

The metric requires `user_input` (conversation messages) and `reference_tool_calls` (expected tool calls). It returns a score between 0 and 1, where higher values indicate better performance.

### Key Features

**Two Evaluation Modes:**

1. **Strict Order (default)**: Tool calls must match exactly in sequence
    - Use for: Sequential workflows where order matters
    - Example: Must search before filtering results

2. **Flexible Order**: Tool calls can be in any order
    - Use for: Parallel operations where order doesn't matter
    - Example: Fetching weather for multiple cities simultaneously

**Scoring:**

- Evaluates sequence alignment (correct tools in correct order)
- Evaluates argument accuracy (correct parameters for each tool)
- Final score = (argument accuracy) × (sequence aligned ? 1 : 0)

### Example: Basic Usage

```python
import asyncio
from ragas.metrics.collections import ToolCallAccuracy
from ragas.messages import AIMessage, HumanMessage, ToolCall

async def evaluate_tool_call_accuracy():
    # Define the conversation with tool calls
    user_input = [
        HumanMessage(content="What's the weather like in New York right now?"),
        AIMessage(
            content="The current temperature in New York is 75°F and it's partly cloudy.",
            tool_calls=[ToolCall(name="weather_check", args={"location": "New York"})],
        ),
        HumanMessage(content="Can you translate that to Celsius?"),
        AIMessage(
            content="Let me convert that to Celsius for you.",
            tool_calls=[
                ToolCall(
                    name="temperature_conversion", args={"temperature_fahrenheit": 75}
                )
            ],
        ),
    ]

    # Define expected tool calls
    reference_tool_calls = [
        ToolCall(name="weather_check", args={"location": "New York"}),
        ToolCall(name="temperature_conversion", args={"temperature_fahrenheit": 75}),
    ]

    # Evaluate
    metric = ToolCallAccuracy()
    result = await metric.ascore(
        user_input=user_input,
        reference_tool_calls=reference_tool_calls,
    )
    print(f"Tool Call Accuracy: {result.value}")

if __name__ == "__main__":
    asyncio.run(evaluate_tool_call_accuracy())
```
Output:
```
Tool Call Accuracy: 1.0
```

### Example: Flexible Order Mode

For scenarios where tool calls can happen in parallel:

```python
# Enable flexible order mode
metric = ToolCallAccuracy(strict_order=False)

user_input = [
    HumanMessage(content="Get weather for Paris and London"),
    AIMessage(
        content="Fetching weather data...",
        tool_calls=[
            ToolCall(name="weather_check", args={"location": "London"}),
            ToolCall(name="weather_check", args={"location": "Paris"}),
        ],
    ),
]

reference_tool_calls = [
    ToolCall(name="weather_check", args={"location": "Paris"}),
    ToolCall(name="weather_check", args={"location": "London"}),
]

result = await metric.ascore(
    user_input=user_input,
    reference_tool_calls=reference_tool_calls,
)
print(f"Score: {result.value}")  # 1.0 (order doesn't matter)
```

### Scoring Examples

**Perfect match:**
```python
# All tools called correctly with correct arguments
Expected: [weather_check(location="Paris"), translate(text="hello")]
Got:      [weather_check(location="Paris"), translate(text="hello")]
Score: 1.0
```

**Partial argument match:**
```python
# Some arguments incorrect
Expected: [search(query="python", limit=10, sort="date")]
Got:      [search(query="python", limit=10, sort="relevance")]
Score: 0.66 (2 out of 3 arguments match)
```

**Wrong order (strict mode):**
```python
# Correct tools but wrong sequence
Expected: [search(...), filter(...)]
Got:      [filter(...), search(...)]
Score: 0.0 (sequence not aligned)
```

### Use Cases

1. **Agent Validation**: Test if agents correctly use tools
2. **Regression Testing**: Ensure tool calling doesn't degrade after changes
3. **Multi-Step Workflows**: Validate complex sequential operations
4. **Tool Selection**: Verify agents pick the right tool from many options

### When to Use Different Metrics

| Metric | Use When |
|--------|----------|
| **ToolCallAccuracy** | You care about exact tool sequence and arguments |
| **ToolCallF1** | You want precision/recall metrics for tool calling |
| **AgentGoalAccuracy** | You care about outcome, not the specific tools used |

**Example:** For "Book me a flight to Paris", if you only care that the booking succeeds (not which intermediate tools were called), use `AgentGoalAccuracyWithReference` instead.

### Legacy API (Deprecated)

!!! warning "Deprecation Notice"
    The legacy `ToolCallAccuracy` from `ragas.metrics` is deprecated and will be removed in v1.0. Please migrate to `ragas.metrics.collections.ToolCallAccuracy` which provides the same functionality with a modern API.

The legacy API can still be used but requires `MultiTurnSample`:

```python
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import AIMessage, HumanMessage, ToolCall
from ragas.metrics import ToolCallAccuracy  # Legacy import

sample = MultiTurnSample(
    user_input=[
        HumanMessage(content="What's the weather in New York?"),
        AIMessage(
            content="Checking weather...",
            tool_calls=[ToolCall(name="weather_check", args={"location": "New York"})],
        ),
    ],
    reference_tool_calls=[
        ToolCall(name="weather_check", args={"location": "New York"}),
    ],
)

scorer = ToolCallAccuracy()
score = await scorer.multi_turn_ascore(sample)
```

The legacy version also supported custom argument comparison metrics:

```python
from ragas.metrics._string import NonLLMStringSimilarity
from ragas.metrics._tool_call_accuracy import ToolCallAccuracy

metric = ToolCallAccuracy()
metric.arg_comparison_metric = NonLLMStringSimilarity()
```

## Tool Call F1

`ToolCallF1` is a metric that return F1-score based on precision and recall of tool calls made by an agent, comparing them to a set of expected calls (`reference_tool_calls`). While `ToolCallAccuracy` provides a binary score based on exact order and content match, `ToolCallF1` complements it by offering a softer evaluation useful for onboarding and iteration. It helps quantify how close the agent was to the expected behavior even if it over- or under-calls.

### Formula

ToolCallF1 is based on classic IR metrics.  It uses unordered matching: the order in which the tools are called does not impact the result, only the presence and correctness of tool names and parameters are considered.

$$
\text{Precision} = \frac{\text{tool calls that match both name and parameters}}{\text{tool calls that match both name and parameters} + \text{extra tool calls that were not expected}}
$$

$$
\text{Recall} = \frac{\text{tool calls that match both name and parameters}}{\text{tool calls that match both name and parameters} + \text{expected tool calls that were not made}}
$$

$$
\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### How is it different from Topic Adherence?

While both `ToolCallF1` and `TopicAdherenceScore` uses precision, recall, and F1-score, they evaluate different aspects:

| Metric                | Evaluates                               | Based on                     |
| --------------------- | --------------------------------------- | ---------------------------- |
| `ToolCallF1`          | Correctness of tool executions          | Structured tool call objects |
| `TopicAdherenceScore` | Whether the conversation stays on-topic | Comparison of domain topics  |

Use `ToolCallF1` when you want to track whether the agent correctly **executed tools**. Use `TopicAdherenceScore` when evaluating whether the **content or intention** stays within allowed topics.

### Example: Basic Usage

```python
import asyncio
from ragas.metrics.collections import ToolCallF1
from ragas.messages import HumanMessage, AIMessage, ToolCall

async def evaluate_tool_call_f1():
    # Define the conversation with tool calls
    user_input = [
        HumanMessage(content="What's the weather like in Paris today?"),
        AIMessage(
            content="Let me check that for you.",
            tool_calls=[ToolCall(name="weather_check", args={"location": "Paris"})],
        ),
        HumanMessage(content="And the UV index?"),
        AIMessage(
            content="Sure, here's the UV index for Paris.",
            tool_calls=[ToolCall(name="uv_index_lookup", args={"location": "Paris"})],
        ),
    ]

    # Define expected tool calls
    reference_tool_calls = [
        ToolCall(name="weather_check", args={"location": "Paris"}),
        ToolCall(name="uv_index_lookup", args={"location": "Paris"}),
    ]

    # Evaluate
    metric = ToolCallF1()
    result = await metric.ascore(
        user_input=user_input,
        reference_tool_calls=reference_tool_calls,
    )
    print(f"Tool Call F1: {result.value}")

if __name__ == "__main__":
    asyncio.run(evaluate_tool_call_f1())
```

Output:
```
Tool Call F1: 1.0
```

### Example: Extra Tool Called

When the agent makes an extra tool call not in the reference:

```python
user_input = [
    HumanMessage(content="What's the weather like in Paris today?"),
    AIMessage(
        content="Let me check that for you.",
        tool_calls=[ToolCall(name="weather_check", args={"location": "Paris"})],
    ),
    HumanMessage(content="And the UV index?"),
    AIMessage(
        content="Sure, here's the UV index and air quality for Paris.",
        tool_calls=[
            ToolCall(name="uv_index_lookup", args={"location": "Paris"}),
            ToolCall(name="air_quality", args={"location": "Paris"}),  # extra call
        ],
    ),
]

reference_tool_calls = [
    ToolCall(name="weather_check", args={"location": "Paris"}),
    ToolCall(name="uv_index_lookup", args={"location": "Paris"}),
]

result = await metric.ascore(
    user_input=user_input,
    reference_tool_calls=reference_tool_calls,
)
print(f"F1 Score: {result.value}")
```

Output:
```
F1 Score: 0.67
```

In this case:
- TP = 2 (weather_check, uv_index_lookup)
- FP = 1 (air_quality)
- FN = 0
- Precision = 2/3 = 0.67, Recall = 2/2 = 1.0, F1 = 0.67

### Scoring Examples

**Perfect match:**
```python
# All tools called correctly
Reference: [weather_check(location="Paris"), uv_index_lookup(location="Paris")]
Got:       [weather_check(location="Paris"), uv_index_lookup(location="Paris")]
F1 Score: 1.0
```

**Missing tool call:**
```python
# One expected tool not called
Reference: [weather_check(...), uv_index_lookup(...)]
Got:       [weather_check(...)]
F1 Score: 0.67 (TP=1, FP=0, FN=1)
```

**Wrong arguments:**
```python
# Tool name matches but args differ
Reference: [weather_check(location="Paris")]
Got:       [weather_check(location="London")]
F1 Score: 0.0 (no match, arguments must be exact)
```

### Legacy API (Deprecated)

!!! warning "Deprecation Notice"
    The legacy `ToolCallF1` from `ragas.metrics` is deprecated and will be removed in v1.0. Please migrate to `ragas.metrics.collections.ToolCallF1` which provides the same functionality with a modern API.

The legacy API can still be used but requires `MultiTurnSample`:

```python
from ragas.metrics import ToolCallF1  # Legacy import
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolCall

sample = MultiTurnSample(
    user_input=[
        HumanMessage(content="What's the weather like in Paris today?"),
        AIMessage(
            content="Let me check that for you.",
            tool_calls=[ToolCall(name="weather_check", args={"location": "Paris"})],
        ),
    ],
    reference_tool_calls=[
        ToolCall(name="weather_check", args={"location": "Paris"}),
    ],
)

scorer = ToolCallF1()
score = await scorer.multi_turn_ascore(sample)
```


## Agent Goal Accuracy


Agent goal accuracy is a metric that can be used to evaluate the performance of the LLM in identifying and achieving the goals of the user. This is a binary metric, with 1 indicating that the AI has achieved the goal and 0 indicating that the AI has not achieved the goal.

### With Reference

`AgentGoalAccuracyWithReference` evaluates whether the agent achieved the user's goal by comparing the workflow's end state against a provided reference outcome. The reference represents the expected/ideal outcome.

```python
import asyncio
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import AgentGoalAccuracyWithReference
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage


async def evaluate_agent_goal_accuracy_with_reference():
    # Setup LLM
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    user_input = [
        HumanMessage(
            content="Hey, book a table at the nearest best Chinese restaurant for 8:00pm"
        ),
        AIMessage(
            content="Sure, let me find the best options for you.",
            tool_calls=[
                ToolCall(
                    name="restaurant_search",
                    args={"cuisine": "Chinese", "time": "8:00pm"},
                )
            ],
        ),
        ToolMessage(
            content="Found a few options: 1. Golden Dragon, 2. Jade Palace"
        ),
        AIMessage(
            content="I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?"
        ),
        HumanMessage(content="Let's go with Golden Dragon."),
        AIMessage(
            content="Great choice! I'll book a table for 8:00pm at Golden Dragon.",
            tool_calls=[
                ToolCall(
                    name="restaurant_book",
                    args={"name": "Golden Dragon", "time": "8:00pm"},
                )
            ],
        ),
        ToolMessage(content="Table booked at Golden Dragon for 8:00pm."),
        AIMessage(
            content="Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!"
        ),
        HumanMessage(content="thanks"),
    ]

    metric = AgentGoalAccuracyWithReference(llm=llm)
    result = await metric.ascore(
        user_input=user_input,
        reference="Table booked at one of the chinese restaurants at 8 pm",
    )
    print(f"Agent Goal Accuracy: {result.value}")


if __name__ == "__main__":
    asyncio.run(evaluate_agent_goal_accuracy_with_reference())
```
Output
```
Agent Goal Accuracy: 1.0
```

### Without Reference

`AgentGoalAccuracyWithoutReference` evaluates whether the agent achieved the user's goal without requiring a reference. The metric infers both the user's intended goal and the achieved outcome from the conversation, then compares them.

```python
import asyncio
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import AgentGoalAccuracyWithoutReference
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage


async def evaluate_agent_goal_accuracy_without_reference():
    # Setup LLM
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    user_input = [
        HumanMessage(
            content="Hey, book a table at the nearest best Chinese restaurant for 8:00pm"
        ),
        AIMessage(
            content="Sure, let me find the best options for you.",
            tool_calls=[
                ToolCall(
                    name="restaurant_search",
                    args={"cuisine": "Chinese", "time": "8:00pm"},
                )
            ],
        ),
        ToolMessage(
            content="Found a few options: 1. Golden Dragon, 2. Jade Palace"
        ),
        AIMessage(
            content="I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?"
        ),
        HumanMessage(content="Let's go with Golden Dragon."),
        AIMessage(
            content="Great choice! I'll book a table for 8:00pm at Golden Dragon.",
            tool_calls=[
                ToolCall(
                    name="restaurant_book",
                    args={"name": "Golden Dragon", "time": "8:00pm"},
                )
            ],
        ),
        ToolMessage(content="Table booked at Golden Dragon for 8:00pm."),
        AIMessage(
            content="Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!"
        ),
        HumanMessage(content="thanks"),
    ]

    metric = AgentGoalAccuracyWithoutReference(llm=llm)
    result = await metric.ascore(user_input=user_input)
    print(f"Agent Goal Accuracy: {result.value}")


if __name__ == "__main__":
    asyncio.run(evaluate_agent_goal_accuracy_without_reference())
```
Output
```
Agent Goal Accuracy: 1.0
```

### Legacy API (Deprecated)

!!! warning "Deprecation Notice"
    The legacy `AgentGoalAccuracyWithReference` and `AgentGoalAccuracyWithoutReference` from `ragas.metrics` are deprecated and will be removed in v1.0. Please migrate to `ragas.metrics.collections` which provides the same functionality with a modern API.

The legacy API can still be used but requires `MultiTurnSample`:

```python
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from ragas.metrics import AgentGoalAccuracyWithReference  # Legacy import
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

sample = MultiTurnSample(
    user_input=[...],  # conversation messages
    reference="Table booked at one of the chinese restaurants at 8 pm",
)
scorer = AgentGoalAccuracyWithReference(llm=evaluator_llm)
score = await scorer.multi_turn_ascore(sample)
```
