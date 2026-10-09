# Evaluation Sample

An evaluation sample is a single structured data instance that is used to assess and measure the performance of your LLM application in specific scenarios. It represents a single unit of interaction or a specific use case that the AI application is expected to handle. In Ragas, evaluation samples are represented using the `SingleTurnSample` and `MultiTurnSample` classes.

## SingleTurnSample
SingleTurnSample represents a single-turn interaction between a user, LLM, and expected results for evaluation. It is suitable for evaluations that involve a single question and answer pair, possibly with additional context or reference information.


### Example
The following example demonstrates how to create a `SingleTurnSample` instance for evaluating a single-turn interaction in a RAG-based application. In this scenario, a user asks a question, and the AI provides an answer. We’ll create a SingleTurnSample instance to represent this interaction, including any retrieved contexts, reference answers, and evaluation rubrics.
```python
from ragas import SingleTurnSample

# User's question
user_input = "What is the capital of France?"

# Retrieved contexts (e.g., from a knowledge base or search engine)
retrieved_contexts = ["Paris is the capital and most populous city of France."]

# AI's response
response = "The capital of France is Paris."

# Reference answer (ground truth)
reference = "Paris"

# Evaluation rubric
rubric = {"accuracy": "Correct", "completeness": "High", "fluency": "Excellent"}

# Create the SingleTurnSample instance
sample = SingleTurnSample(
    user_input=user_input,
    retrieved_contexts=retrieved_contexts,
    response=response,
    reference=reference,
    rubric=rubric,
)
```

## MultiTurnSample

MultiTurnSample represents a multi-turn interaction between Human, AI and optionally a Tool and expected results for evaluation. It is suitable for representing conversational agents in more complex interactions for evaluation. In `MultiTurnSample`, the `user_input` attribute represents a sequence of messages that collectively form a multi-turn conversation between a human user and an AI system. These messages are instances of the classes  `HumanMessage`, `AIMessage`, and `ToolMessage`


### Example
The following example demonstrates how to create a `MultiTurnSample` instance for evaluating a multi-turn interaction. In this scenario, a user wants to know the current weather in New York City. The AI assistant will use a weather API tool to fetch the information and respond to the user.


```python
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall

# User asks about the weather in New York City
user_message = HumanMessage(content="What's the weather like in New York City today?")

# AI decides to use a weather API tool to fetch the information
ai_initial_response = AIMessage(
    content="Let me check the current weather in New York City for you.",
    tool_calls=[ToolCall(name="WeatherAPI", args={"location": "New York City"})],
)

# Tool provides the weather information
tool_response = ToolMessage(
    content="It's sunny with a temperature of 75°F in New York City."
)

# AI delivers the final response to the user
ai_final_response = AIMessage(
    content="It's sunny and 75 degrees Fahrenheit in New York City today."
)

# Combine all messages into a list to represent the conversation
conversation = [user_message, ai_initial_response, tool_response, ai_final_response]
```

Now, use the conversation to create a MultiTurnSample object, including any reference responses and evaluation rubrics.
```python
from ragas import MultiTurnSample

# Reference response for evaluation purposes
reference_response = "Provide the current weather in New York City to the user."


# Create the MultiTurnSample instance
sample = MultiTurnSample(
    user_input=conversation,
    reference=reference_response,
)
```