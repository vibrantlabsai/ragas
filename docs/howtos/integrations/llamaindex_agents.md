# Evaluating LlamaIndex Agents

Building agents that can intelligently use tools and make decisions is only half the journey; ensuring that these agents are accurate, reliable, and performant is what truly defines their success. [LlamaIndex](https://docs.llamaindex.ai/en/stable/understanding/agent/) provides various ways to create agents including [FunctionAgents](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/), [CodeActAgents](https://docs.llamaindex.ai/en/stable/examples/agent/code_act_agent/), and [ReActAgents](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/). In this tutorial, we will explore how to evaluate these different agent types using both pre-built Ragas metrics and custom evaluation metrics.

Let's get started.

The tutorial is divided into three comprehensive sections:

1. **Evaluating with Off-the-Shelf Ragas Metrics**
   Here we will examine two fundamental evaluation tools: AgentGoalAccuracy, which measures how effectively an agent identifies and achieves the user's intended objective, and Tool Call Accuracy, which assesses the agent's ability to select and invoke appropriate tools in the correct sequence to complete tasks.

2. **Custom Metrics for CodeActAgent Evaluation**
   This section focuses on LlamaIndex's prebuilt CodeActAgent, demonstrating how to develop tailored evaluation metrics that address the specific requirements and capabilities of code-generating agents.

3. **Query Engine Tool Assessment**
   The final section explores how to leverage Ragas RAG metrics to evaluate query engine functionality within agents, providing insights into retrieval effectiveness and response quality when agents access information systems.

## Ragas Agentic Metrics

To demonstrate evaluations using Ragas metrics, we will create a simple workflow with a single LlamaIndex Function Agent, and use that to cover the basic functionality.

??? note "Click to View the Function Agent Setup"

    ```python
    from llama_index.llms.openai import OpenAI


    async def send_message(to: str, content: str) -> str:
        """Dummy function to simulate sending an email."""
        return f"Successfully sent mail to {to}"


    llm = OpenAI(model="gpt-4o-mini")
    ```


    ```python
    from llama_index.core.agent.workflow import FunctionAgent

    agent = FunctionAgent(
        tools=[send_message],
        llm=llm,
        system_prompt="You are a helpful assistant of Jane",
    )
    ```

### Agent Goal Accuracy

The true value of an AI agent lies in its ability to understand what users want and deliver it effectively. Agent Goal Accuracy serves as a fundamental metric that evaluates whether an agent successfully accomplishes what the user intended. This measurement is crucial as it directly reflects how well the agent interprets user needs and takes appropriate actions to fulfill them.

Ragas provides two key variants of this metric:

- [AgentGoalAccuracyWithReference](../../concepts/metrics/available_metrics/agents.md#with-reference) - A binary assessment (1 or 0) that compares the agent's final outcome against a predefined expected result.
- [AgentGoalAccuracyWithoutReference](../../concepts/metrics/available_metrics/agents.md#without-reference) - A binary assessment (1 or 0) that evaluates whether the agent achieved the user's goal based on inferred intent rather than predefined expectations.

With Reference is ideal for scenarios where the expected outcome is well-defined, such as in controlled testing environments or when testing against ground truth data. 


```python
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    AgentStream,
    ToolCall as LlamaToolCall,
    ToolCallResult,
)

handler = agent.run(user_msg="Send a message to jhon asking for a meeting")

events = []

async for ev in handler.stream_events():
    if isinstance(ev, (AgentInput, AgentOutput, LlamaToolCall, ToolCallResult)):
        events.append(ev)
    elif isinstance(ev, AgentStream):
        print(f"{ev.delta}", end="", flush=True)
    elif isinstance(ev, ToolCallResult):
        print(
            f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}"
        )

response = await handler
```
Output:
```
I have successfully sent a message to Jhon asking for a meeting.
```

```python
from ragas.integrations.llama_index import convert_to_ragas_messages

ragas_messages = convert_to_ragas_messages(events)
```


```python
from ragas.metrics import AgentGoalAccuracyWithoutReference
from ragas.llms import LlamaIndexLLMWrapper
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import ToolCall as RagasToolCall

evaluator_llm = LlamaIndexLLMWrapper(llm=llm)

sample = MultiTurnSample(
    user_input=ragas_messages,
)

agent_goal_accuracy_without_reference = AgentGoalAccuracyWithoutReference(
    llm=evaluator_llm
)
await agent_goal_accuracy_without_reference.multi_turn_ascore(sample)
```
Output:
```
1.0
```

```python
from ragas.metrics import AgentGoalAccuracyWithReference

sample = MultiTurnSample(
    user_input=ragas_messages,
    reference="Successfully sent a message to Jhon asking for a meeting",
)


agent_goal_accuracy_with_reference = AgentGoalAccuracyWithReference(llm=evaluator_llm)
await agent_goal_accuracy_with_reference.multi_turn_ascore(sample)
```
Output:
```
1.0
```

### Tool Call Accuracy

In agentic workflows, an AI agent's effectiveness depends heavily on its ability to select and use the right tools at the right time. The Tool Call Accuracy metric evaluates how precisely an agent identifies and invokes appropriate tools in the correct sequence to complete a user's request. This measurement ensures that agents not only understand what tools are available but also how to orchestrate them effectively to achieve the intended outcome.

- [ToolCallAccuracy](../../concepts/metrics/available_metrics/agents.md#tool-call-accuracy) compares the agent's actual tool usage against a reference sequence of expected tool calls. If the agent's tool selection or sequence differs from the reference, the metric returns a score of 0, indicating a failure to follow the optimal path to task completion.


```python
from ragas.metrics import ToolCallAccuracy

sample = MultiTurnSample(
    user_input=ragas_messages,
    reference_tool_calls=[
        RagasToolCall(
            name="send_message",
            args={
                "to": "jhon",
                "content": "Hi Jhon,\n\nI hope this message finds you well. I would like to schedule a meeting to discuss some important matters. Please let me know your availability.\n\nBest regards,\nJane",
            },
        ),
    ],
)

tool_accuracy_scorer = ToolCallAccuracy()
await tool_accuracy_scorer.multi_turn_ascore(sample)
```
Output:
```
1.0
```


## Evaluating LlamaIndex CodeAct Agents

LlamaIndex offers a prebuilt CodeAct Agent that can be used to write and execute code, inspired by the original CodeAct paper. The idea is: instead of outputting a simple JSON object, a Code Agent generates an executable code block—typically in a high-level language like Python. Writing actions in code rather than JSON-like snippets provides better:

- Composability: Code naturally allows nesting and reuse of functions; JSON actions lack this flexibility.
- Object management: Code elegantly handles operation outputs (image = generate_image()); JSON has no clean equivalent.
- Generality: Code expresses any computational task; JSON imposes unnecessary constraints.
- Representation in LLM training data: LLMs already understand code from training data, making it a more natural interface than specialized JSON.

??? note "Click to View the CodeActAgent Setup"

    ### Defining Functions

    ```python
    from llama_index.llms.openai import OpenAI

    # Configure the LLM
    llm = OpenAI(model="gpt-4o-mini")


    # Define a few helper functions
    def add(a: int, b: int) -> int:
        """Add two numbers together"""
        return a + b


    def subtract(a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b


    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b


    def divide(a: int, b: int) -> float:
        """Divide two numbers"""
        return a / b
    ```

    ### Create a Code Executor

    The CodeActAgent will require a specific code_execute_fn to execute the code generated by the agent.


    ```python
    from typing import Any, Dict, Tuple
    import io
    import contextlib
    import ast
    import traceback


    class SimpleCodeExecutor:
        """
        A simple code executor that runs Python code with state persistence.

        This executor maintains a global and local state between executions,
        allowing for variables to persist across multiple code runs.

        NOTE: not safe for production use! Use with caution.
        """

        def __init__(self, locals: Dict[str, Any], globals: Dict[str, Any]):
            """
            Initialize the code executor.

            Args:
                locals: Local variables to use in the execution context
                globals: Global variables to use in the execution context
            """
            # State that persists between executions
            self.globals = globals
            self.locals = locals

        def execute(self, code: str) -> Tuple[bool, str, Any]:
            """
            Execute Python code and capture output and return values.

            Args:
                code: Python code to execute

            Returns:
                Dict with keys `success`, `output`, and `return_value`
            """
            # Capture stdout and stderr
            stdout = io.StringIO()
            stderr = io.StringIO()

            output = ""
            return_value = None
            try:
                # Execute with captured output
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    # Try to detect if there's a return value (last expression)
                    try:
                        tree = ast.parse(code)
                        last_node = tree.body[-1] if tree.body else None

                        # If the last statement is an expression, capture its value
                        if isinstance(last_node, ast.Expr):
                            # Split code to add a return value assignment
                            last_line = code.rstrip().split("\n")[-1]
                            exec_code = (
                                code[: -len(last_line)] + "\n__result__ = " + last_line
                            )

                            # Execute modified code
                            exec(exec_code, self.globals, self.locals)
                            return_value = self.locals.get("__result__")
                        else:
                            # Normal execution
                            exec(code, self.globals, self.locals)
                    except:
                        # If parsing fails, just execute the code as is
                        exec(code, self.globals, self.locals)

                # Get output
                output = stdout.getvalue()
                if stderr.getvalue():
                    output += "\n" + stderr.getvalue()

            except Exception as e:
                # Capture exception information
                output = f"Error: {type(e).__name__}: {str(e)}\n"
                output += traceback.format_exc()

            if return_value is not None:
                output += "\n\n" + str(return_value)

            return output
    ```


    ```python
    code_executor = SimpleCodeExecutor(
        # give access to our functions defined above
        locals={
            "add": add,
            "subtract": subtract,
            "multiply": multiply,
            "divide": divide,
        },
        globals={
            # give access to all builtins
            "__builtins__": __builtins__,
            # give access to numpy
            "np": __import__("numpy"),
        },
    )
    ```

    ### Setup the CodeAct Agent


    ```python
    from llama_index.core.agent.workflow import CodeActAgent
    from llama_index.core.workflow import Context

    agent = CodeActAgent(
        code_execute_fn=code_executor.execute,
        llm=llm,
        tools=[add, subtract, multiply, divide],
    )

    # context to hold the agent's session/state/chat history
    ctx = Context(agent)
    ```

### Running and Evaluating the CodeAct agent


```python
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    AgentStream,
    ToolCall,
    ToolCallResult,
)

handler = agent.run("Calculate the sum of the first 10 fibonacci numbers", ctx=ctx)

events = []

async for event in handler.stream_events():
    if isinstance(event, (AgentInput, AgentOutput, ToolCall, ToolCallResult)):
        events.append(event)
    elif isinstance(event, AgentStream):
        print(f"{event.delta}", end="", flush=True)
```

    The first 10 Fibonacci numbers are 0, 1, 1, 2, 3, 5, 8, 13, 21, and 34. I will calculate their sum. 
    
    <execute>
    def fibonacci(n):
        fib_sequence = [0, 1]
        for i in range(2, n):
            next_fib = fib_sequence[-1] + fib_sequence[-2]
            fib_sequence.append(next_fib)
        return fib_sequence
    
    # Calculate the first 10 Fibonacci numbers
    first_10_fib = fibonacci(10)
    
    # Calculate the sum of the first 10 Fibonacci numbers
    sum_fib = sum(first_10_fib)
    print(sum_fib)
    </execute>The sum of the first 10 Fibonacci numbers is 88.

### Extract the ToolCall


```python
CodeAct_agent_tool_call = events[2]
agent_code = CodeAct_agent_tool_call.tool_kwargs["code"]

print(agent_code)
```
Output
```
    def fibonacci(n):
        fib_sequence = [0, 1]
        for i in range(2, n):
            next_fib = fib_sequence[-1] + fib_sequence[-2]
            fib_sequence.append(next_fib)
        return fib_sequence
    
    # Calculate the first 10 Fibonacci numbers
    first_10_fib = fibonacci(10)
    
    # Calculate the sum of the first 10 Fibonacci numbers
    sum_fib = sum(first_10_fib)
    print(sum_fib)
```

When assessing CodeAct agents, we can begin with foundational metrics that examine basic functionality, such as code compilability or appropriate argument selection. These straightforward evaluations provide a solid foundation before advancing to more sophisticated assessment approaches. 

Ragas offers powerful custom metric capabilities that enable increasingly nuanced evaluation as your requirements evolve.

- [AspectCritic](../../concepts/metrics/available_metrics/aspect_critic.md) - Provides a binary evaluation (pass/fail) that determines whether an agent's response satisfies specific user-defined criteria, using LLM-based judgment to deliver clear success indicators.
- [RubricScoreMetric](../../concepts/metrics/available_metrics/general_purpose.md#rubrics-based-criteria-scoring) - Evaluates agent responses against comprehensive, predefined quality rubrics with discrete scoring levels, enabling consistent performance assessment across multiple dimensions.


```python
def is_compilable(code_str: str, mode="exec") -> bool:
    try:
        compile(code_str, "<string>", mode)
        return True
    except Exception:
        return False


is_compilable(agent_code)
```
Output
```
True
```



```python
from ragas.metrics import AspectCritic
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LlamaIndexLLMWrapper

llm = OpenAI(model="gpt-4o-mini")
evaluator_llm = LlamaIndexLLMWrapper(llm=llm)

correct_tool_args = AspectCritic(
    name="correct_tool_args",
    llm=evaluator_llm,
    definition="Score 1 if the tool arguements use in the tool call are correct and 0 otherwise",
)

sample = SingleTurnSample(
    user_input="Calculate the sum of the first 10 fibonacci numbers",
    response=agent_code,
)

await correct_tool_args.single_turn_ascore(sample)
```
Output:
```
1
```


## Evaluating Query Engine Tool

When evaluating with Ragas metrics, we need to ensure that our data is formatted suitably for evaluations. When working with a query engine tool within an agentic system, we can approach the evaluation as we would for any retrieval-augmented generation (RAG) system.

We will extract all instances where the query engine tool was called during user interactions. Using that, we can construct a Ragas RAG evaluation dataset based on our event stream data. Once the dataset is ready, we can apply the full suite of Ragas evaluation metrics. In this section, we will set up a Functional Agent with Query Engine Tools. The agent has access to two "tools": one to query the 2021 Lyft 10-K and the other to query the 2021 Uber 10-K.

??? note "Click to View the Agent Setup"

    ### Setting the LLMs

    ```python
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core import Settings

    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    ```

    ### Build Query Engine Tools


    ```python
    from llama_index.core import StorageContext, load_index_from_storage

    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage/lyft")
        lyft_index = load_index_from_storage(storage_context)

        storage_context = StorageContext.from_defaults(persist_dir="./storage/uber")
        uber_index = load_index_from_storage(storage_context)

        index_loaded = True
    except:
        index_loaded = False
    ```


    ```python
    !mkdir -p 'data/10k/'
    !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
    !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'
    ```

    ```python
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

    if not index_loaded:
        # load data
        lyft_docs = SimpleDirectoryReader(
            input_files=["./data/10k/lyft_2021.pdf"]
        ).load_data()
        uber_docs = SimpleDirectoryReader(
            input_files=["./data/10k/uber_2021.pdf"]
        ).load_data()

        # build index
        lyft_index = VectorStoreIndex.from_documents(lyft_docs)
        uber_index = VectorStoreIndex.from_documents(uber_docs)

        # persist index
        lyft_index.storage_context.persist(persist_dir="./storage/lyft")
        uber_index.storage_context.persist(persist_dir="./storage/uber")
    ```


    ```python
    lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
    uber_engine = uber_index.as_query_engine(similarity_top_k=3)
    ```


    ```python
    from llama_index.core.tools import QueryEngineTool

    query_engine_tools = [
        QueryEngineTool.from_defaults(
            query_engine=lyft_engine,
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
        QueryEngineTool.from_defaults(
            query_engine=uber_engine,
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ]
    ```


    ### Agent Setup


    ```python
    from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
    from llama_index.core.workflow import Context

    agent = FunctionAgent(tools=query_engine_tools, llm=OpenAI(model="gpt-4o-mini"))

    # context to hold the session/state
    ctx = Context(agent)
    ```

### Running and Evaluating Agents


```python
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)

handler = agent.run("What's the revenue for Lyft in 2021 vs Uber?", ctx=ctx)

events = []

async for ev in handler.stream_events():
    if isinstance(ev, (AgentInput, AgentOutput, ToolCall, ToolCallResult)):
        events.append(ev)
    elif isinstance(ev, AgentStream):
        print(ev.delta, end="", flush=True)

response = await handler
```
Output:
```
In 2021, Lyft generated a total revenue of $3.21 billion, while Uber's total revenue was significantly higher at $17.455 billion.
```

We will extract all instances of `ToolCallResult` where the query engine tool was called during user interactions using that we can construct a proper RAG evaluation dataset based on your event stream data.


```python
from ragas.dataset_schema import SingleTurnSample

ragas_samples = []

for event in events:
    if isinstance(event, ToolCallResult):
        if event.tool_name in ["lyft_10k", "uber_10k"]:
            sample = SingleTurnSample(
                user_input=event.tool_kwargs["input"],
                response=event.tool_output.content,
                retrieved_contexts=[
                    node.text for node in event.tool_output.raw_output.source_nodes
                ],
            )
            ragas_samples.append(sample)
```


```python
from ragas.dataset_schema import EvaluationDataset

dataset = EvaluationDataset(samples=ragas_samples)
dataset.to_pandas()
```
Output:

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
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What was the total revenue for Uber in the yea...</td>
      <td>[Financial and Operational Highlights\nYear En...</td>
      <td>The total revenue for Uber in the year 2021 wa...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What was the total revenue for Lyft in the yea...</td>
      <td>[Significant items\n subject to estimates and ...</td>
      <td>The total revenue for Lyft in the year 2021 wa...</td>
    </tr>
  </tbody>
</table>
</div>



The resulting dataset will not include reference answers by default, so we’ll be limited to using metrics that do not require references. However, if you wish to run reference-based evaluations, you can add a reference column to the dataset and then apply the relevant Ragas metrics.

### Evaluating using Ragas RAG Metrics

Let's assess the effectiveness of query engines, particularly regarding retrieval quality and hallucination prevention. To accomplish this evaluation, We will employ two key Ragas metrics: faithfulness and context relevance. For more you can visit [here](../../concepts/metrics/available_metrics/).

This evaluation approach allows us to identify potential issues with either retrieval quality or response generation that could impact overall system performance.
- [Faithfulness](../../concepts/metrics/available_metrics/faithfulness.md) - Measures how accurately the generated response adheres to the facts presented in the retrieved context, ensuring claims made by the system can be directly supported by the information provided.
- [Context Relevance](../../concepts/metrics/available_metrics/nvidia_metrics.md#context-relevance) - Evaluates how effectively the retrieved information addresses the user's specific query by assessing its pertinence through dual LLM judgment mechanisms.



```python
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextRelevance
from ragas.llms import LlamaIndexLLMWrapper
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o")
evaluator_llm = LlamaIndexLLMWrapper(llm=llm)

faithfulness = Faithfulness(llm=evaluator_llm)
context_precision = ContextRelevance(llm=evaluator_llm)

result = evaluate(dataset, metrics=[faithfulness, context_precision])
```
```
Evaluating: 100%|██████████| 4/4 [00:03<00:00,  1.19it/s]
```


```python
result.to_pandas()
```
Output:

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
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
      <th>faithfulness</th>
      <th>nv_context_relevance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What was the total revenue for Uber in the yea...</td>
      <td>[Financial and Operational Highlights\nYear En...</td>
      <td>The total revenue for Uber in the year 2021 wa...</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What was the total revenue for Lyft in the yea...</td>
      <td>[Significant items\n subject to estimates and ...</td>
      <td>The total revenue for Lyft in the year 2021 wa...</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>
