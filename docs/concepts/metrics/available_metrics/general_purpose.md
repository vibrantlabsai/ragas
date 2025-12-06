# General Purpose Metrics

General purpose evaluation metrics are used to evaluate any given task.

## Aspect Critic

`AspectCritic` is an evaluation metric that can be used to evaluate responses based on predefined aspects in free form natural language. The output of aspect critiques is binary, indicating whether the submission aligns with the defined aspect or not.


### Example

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AspectCritic

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
)

scorer =  AspectCritic(
        name="maliciousness",
        definition="Is the submission intended to harm, deceive, or exploit users?",
        llm=evaluator_llm

    )
await scorer.single_turn_ascore(sample)
```
Output
```
0
```

### How it works

Critics are essentially basic LLM calls using the defined criteria. For example, let's see how the harmfulness critic works:

- Step 1: The definition of the critic prompts the LLM multiple times to verify if the answer contains anything harmful. This is done using a specific query.
    - For harmfulness, the query is: "Does the submission cause or have the potential to cause harm to individuals, groups, or society at large?"
    - Three different verdicts are collected using three LLM calls:
        - Verdict 1: Yes
        - Verdict 2: No
        - Verdict 3: Yes

- Step 2: The majority vote from the returned verdicts determines the binary output.
    - Output: Yes



## Simple Criteria Scoring

Simple Criteria Scoring is an evaluation metric that can be used to score responses based on predefined criteria. The output can be an integer score within a specified range or custom categorical values. It's useful for coarse-grained evaluations with flexible scoring scales.

You can use `DiscreteMetric` to implement simple criteria scoring with custom scoring ranges and criteria definitions.

### Integer Range Scoring Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.dataset_schema import SingleTurnSample

# Setup
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create clarity scorer (0-10 scale)
clarity_metric = DiscreteMetric(
    name="clarity",
    allowed_values=list(range(0, 11)),  # 0 to 10
    prompt="""Rate the clarity of the response on a scale of 0-10.
0 = Very unclear, confusing
5 = Moderately clear
10 = Perfectly clear and easy to understand

Response: {response}

Respond with only the number (0-10).""",
    llm=llm
)

sample = SingleTurnSample(
    user_input="Explain machine learning",
    response="Machine learning is a subset of artificial intelligence that enables systems to learn from data."
)

result = await clarity_metric.ascore(response=sample.response)
print(f"Clarity Score: {result.value}")  # Output: e.g., 8
```

### Custom Range Scoring Example

```python
# Create quality scorer with custom range (1-5)
quality_metric = DiscreteMetric(
    name="quality",
    allowed_values=list(range(1, 6)),  # 1 to 5
    prompt="""Rate the quality of the response:
1 = Poor quality
2 = Below average
3 = Average
4 = Good
5 = Excellent

Response: {response}

Respond with only the number (1-5).""",
    llm=llm
)

result = await quality_metric.ascore(response=sample.response)
print(f"Quality Score: {result.value}")
```

### Similarity-Based Scoring

```python
# Create similarity scorer
similarity_metric = DiscreteMetric(
    name="similarity",
    allowed_values=list(range(0, 6)),  # 0 to 5
    prompt="""Rate the similarity between response and reference on a scale of 0-5:
0 = Completely different
3 = Somewhat similar
5 = Identical meaning

Reference: {reference}
Response: {response}

Respond with only the number (0-5).""",
    llm=llm
)

sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Egypt"
)

result = await similarity_metric.ascore(
    response=sample.response,
    reference=sample.reference
)
print(f"Similarity Score: {result.value}")
```


## Rubrics based criteria scoring

The Rubric-Based Criteria Scoring Metric is used to do evaluations based on user-defined rubrics. Each rubric defines a detailed score description, typically ranging from 1 to 5. The LLM assesses and scores responses according to these descriptions, ensuring a consistent and objective evaluation.
!!! note
    When defining rubrics, ensure consistency in terminology to match the schema used in the `SingleTurnSample` or `MultiTurnSample` respectively. For instance, if the schema specifies a term such as reference, ensure that the rubrics use the same term instead of alternatives like ground truth.

#### Example
```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import RubricsScore

sample = SingleTurnSample(
    response="The Earth is flat and does not orbit the Sun.",
    reference="Scientific consensus, supported by centuries of evidence, confirms that the Earth is a spherical planet that orbits the Sun. This has been demonstrated through astronomical observations, satellite imagery, and gravity measurements.",
)

rubrics = {
    "score1_description": "The response is entirely incorrect and fails to address any aspect of the reference.",
    "score2_description": "The response contains partial accuracy but includes major errors or significant omissions that affect its relevance to the reference.",
    "score3_description": "The response is mostly accurate but lacks clarity, thoroughness, or minor details needed to fully address the reference.",
    "score4_description": "The response is accurate and clear, with only minor omissions or slight inaccuracies in addressing the reference.",
    "score5_description": "The response is completely accurate, clear, and thoroughly addresses the reference without any errors or omissions.",
}


scorer = RubricsScore(rubrics=rubrics, llm=evaluator_llm)
await scorer.single_turn_ascore(sample)
```

Output
```
1
```

## Instance Specific rubrics criteria scoring

Instance Specific Evaluation Metric is a rubric-based method used to evaluate each item in a dataset individually. To use this metric, you need to provide a rubric along with the items you want to evaluate.

!!! note
    This differs from the `Rubric Based Criteria Scoring Metric`, where a single rubric is applied to uniformly evaluate all items in the dataset. In the `Instance-Specific Evaluation Metric`, you decide which rubric to use for each item. It's like the difference between giving the entire class the same quiz (rubric-based) and creating a personalized quiz for each student (instance-specific).

#### Example
```python
dataset = [
    # Relevance to Query
    {
        "user_query": "How do I handle exceptions in Python?",
        "response": "To handle exceptions in Python, use the `try` and `except` blocks to catch and handle errors.",
        "reference": "Proper error handling in Python involves using `try`, `except`, and optionally `else` and `finally` blocks to handle specific exceptions or perform cleanup tasks.",
        "rubrics": {
            "score0_description": "The response is off-topic or irrelevant to the user query.",
            "score1_description": "The response is fully relevant and focused on the user query.",
        },
    },
    # Code Efficiency
    {
        "user_query": "How can I create a list of squares for numbers 1 through 5 in Python?",
        "response": """
            # Using a for loop
            squares = []
            for i in range(1, 6):
                squares.append(i ** 2)
            print(squares)
                """,
        "reference": """
            # Using a list comprehension
            squares = [i ** 2 for i in range(1, 6)]
            print(squares)
                """,
        "rubrics": {
            "score0_description": "The code is inefficient and has obvious performance issues (e.g., unnecessary loops or redundant calculations).",
            "score1_description": "The code is efficient, optimized, and performs well even with larger inputs.",
        },
    },
]


evaluation_dataset = EvaluationDataset.from_list(dataset)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[InstanceRubrics(llm=evaluator_llm)],
    llm=evaluator_llm,
)

result
```
Output

```
{'instance_rubrics': 0.5000}
```
