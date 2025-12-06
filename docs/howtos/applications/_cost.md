# How to estimate Cost and Usage of evaluations and testset generation

When using LLMs for evaluation and test set generation, cost will be an important factor. Ragas provides you some tools to help you with that.

## Implement `TokenUsageParser`

By default, Ragas does not calculate the usage of tokens for `evaluate()`. This is because langchain's LLMs do not always return information about token usage in a uniform way. So in order to get the usage data, we have to implement a `TokenUsageParser`.

A `TokenUsageParser` is function that parses the `LLMResult` or `ChatResult` from langchain models `generate_prompt()` function and outputs `TokenUsage` which Ragas expects.

For an example here is one that will parse OpenAI by using a parser we have defined.


```python
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompt_values import StringPromptValue

gpt4o = ChatOpenAI(model="gpt-4o")
p = StringPromptValue(text="hai there")
llm_result = gpt4o.generate_prompt([p])

# lets import a parser for OpenAI
from ragas.cost import get_token_usage_for_openai

get_token_usage_for_openai(llm_result)
```
Output
```
TokenUsage(input_tokens=9, output_tokens=9, model='')
```


You can define your own or import parsers if they are defined. If you would like to suggest parser for LLM providers or contribute your own ones please check out this [issue](https://github.com/vibrantlabsai/ragas/issues/1151) 🙂.

## Token Usage for Evaluations

Let's use the `get_token_usage_for_openai` parser to calculate the token usage for an evaluation.


```python
from ragas import EvaluationDataset
from datasets import load_dataset

dataset = load_dataset("vibrantlabsai/amnesty_qa", "english_v3")

eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])
```
Output
```
Repo card metadata block was not found. Setting CardData to empty.
```

You can pass in the parser to the `evaluate()` function and the cost will be calculated and returned in the `Result` object.


```python
from ragas import evaluate
from ragas.metrics import LLMContextRecall

from ragas.cost import get_token_usage_for_openai

result = evaluate(
    eval_dataset,
    metrics=[LLMContextRecall()],
    llm=gpt4o,
    token_usage_parser=get_token_usage_for_openai,
)
```
Output
```
Evaluating:   0%|          | 0/20 [00:00<?, ?it/s]
```


```python
result.total_tokens()
```
Output
```
TokenUsage(input_tokens=25097, output_tokens=3757, model='')
```


You can compute the cost for each run by passing in the cost per token to `Result.total_cost()` function.

In this case GPT-4o costs $5 for 1M input tokens and $15 for 1M output tokens.


```python
result.total_cost(cost_per_input_token=5 / 1e6, cost_per_output_token=15 / 1e6)
```

Output
```
1.1692900000000002
```


## Token Usage for Testset Generation

You can use the same parser for testset generation, but you need to pass in the `token_usage_parser` to the `generate()` function. For now, it only calculates the cost for the generation process and not the cost for the transforms.

For an example let's load an existing KnowledgeGraph and generate a testset. If you want to know more about how to generate a testset please check out the [testset generation](../../getstarted/rag_testset_generation.md#a-deeper-look).


```python
from ragas.testset.graph import KnowledgeGraph

# loading an existing KnowledgeGraph
# make sure to change the path to the location of the KnowledgeGraph file
kg = KnowledgeGraph.load("../../../experiments/scratchpad_kg.json")
kg
```

Output
```
KnowledgeGraph(nodes: 47, relationships: 109)



### Choose your LLM

--8<--
choose_generator_llm.md
--8<--


```python
from ragas.testset import TestsetGenerator
from ragas.llms import llm_factory

tg = TestsetGenerator(llm=llm_factory(), knowledge_graph=kg)
# generating a testset
testset = tg.generate(testset_size=10, token_usage_parser=get_token_usage_for_openai)
```


```python
# total cost for the generation process
testset.total_cost(cost_per_input_token=5 / 1e6, cost_per_output_token=15 / 1e6)
```

Output
```
0.20967000000000002
```

