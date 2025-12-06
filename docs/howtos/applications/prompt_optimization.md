# A systematic approach for prompt optimization

Creating reliable and consistent prompts remains a significant challenge. As requirements multiply and prompt structures grow more complex, even minor modifications can lead to unexpected failures. This often turns traditional prompt engineering into a frustrating game of “whack-a-mole”—fix one issue, and two more seem to emerge.

This tutorial demonstrates how to implement a systematic, data-driven approach to prompt engineering through functional testing with Ragas.

## The Diabetes Medication Management Assistant

For our tutorial, we will focus on evaluating prompts for a Diabetes Medication Management Assistant—an AI tool designed to help diabetes patients manage their medication, monitor their health, and receive personalized support.

**Dataset Overview**

Our evaluation uses a carefully curated dataset of 15 representative queries:  

- 10 on-topic questions within the assistant's domain expertise (medication management, glucose monitoring, etc.)
- 5 out-of-scope questions designed to test the assistant's ability to recognize its limitations and decline to provide advice

This balanced dataset allows us to assess both the assistant's helpfulness when appropriate and its safety guardrails when faced with queries beyond its expertise.

First, download the dataset:
```
!curl -O https://huggingface.co/datasets/vibrantlabsai/diabetes_assistant_dataset/resolve/main/diabetes_assistant_dataset.csv
```
We'll test two nearly identical prompts that differ by only a single line - one with standard instructions and another with an added financial incentive statement. This minimal variation will help us investigate our hypothesis: do LLMs demonstrate improved instruction-following when presented with financial incentives?

## Understanding the Data

Our dataset consists of three key parts:
- `user_input`: These are the questions provided by diabetes patients.
- `retrieved_contexts`: This is the relevant information that the retriever gathered to answer the questions.
- `reference`: These are the gold-standard answers used for comparison.


```python
import pandas as pd

eval_df = pd.read_csv("diabetes_assistant_dataset.csv")
eval_df.head()
```




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
      <th>reference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I missed my afternoon insulin dose—what should...</td>
      <td>['Clinical guidelines recommend that if an ins...</td>
      <td>If you miss an insulin dose, first check your ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Based on my latest blood glucose readings, how...</td>
      <td>['Recent clinical guidelines emphasize the imp...</td>
      <td>Your insulin dosage adjustments should be base...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I often get alerts for low or high blood sugar...</td>
      <td>['Current clinical practices emphasize the imp...</td>
      <td>Monitor your blood sugar alerts by reviewing t...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I have a fear of needles. Are there alternativ...</td>
      <td>['For patients with needle phobia, clinical gu...</td>
      <td>There are alternative options available, inclu...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I'm switching from oral medications to insulin...</td>
      <td>["Transitioning from oral medications to insul...</td>
      <td>During your transition from oral medications t...</td>
    </tr>
  </tbody>
</table>
</div>



In a real-world scenario, you'd typically have more samples (perhaps 50-100) to get statistically significant results.

## Testing Financial Incentives on LLM Performance

Our goal is to test the hypothesis that including a financial incentive in the prompt can improve the model’s adherence to instructions. To achieve this, we will define two prompt templates for our Diabetes Medication Management Assistant:


- Standard Prompt: Contains the basic instructions for the Diabetes Medication Management Assistant.
- Incentivized Prompt: Uses the same basic instructions but adds an incentive statement:
“If you follow all the instructions precisely, you will earn a reward of $1,000,000.”

By comparing the performance of these two prompt variations, we aim to determine whether a financial incentive encourages the model to follow instructions more accurately.


```python
# Define our base prompt for the diabetes assistant
diabetes_assistant_prompt = (
    "You are a diabetes medication management assistant. Your role is to be a knowledgeable and compassionate healthcare companion "
    "who balances professionalism with approachability. You are reliable, supportive, and clear, offering medical guidance using simple, "
    "accessible language while maintaining data-driven insights. Your tone is warm and conversational, yet respectfully direct when urgent "
    "issues arise. You provide gentle reminders and encourage long-term adherence without being intrusive.\n\n"
    "User Input: \n{user_input}\n\n"
    "Retrieved Contexts: \n{contexts}\n\n"
    "Instructions:\n"
    '- Answer the user\'s query using only the information provided in the "Retrieved Contexts".\n'
    "- Do not add any external information or assumptions beyond what is given.\n"
    '- If the "Retrieved Contexts" contain sufficient details to answer the question, provide a clear, concise, and supportive response based solely on that context.\n'
    '- If the "Retrieved Contexts" do not provide enough expertise or information to reliably answer the query, respond with: "I don\'t have enough expertise in that matter to comment on this. Please consult your doctor."\n'
    "- Ensure your response is tailored to the user's needs, balancing empathy with clear, data-focused guidance.\n"
)

# Define an incentive that we'll test
incentive = "If you follow all the instructions precisely, you will earn a reward of $1,000,000."

# Create a variant with the incentive
incentive_prompt = diabetes_assistant_prompt + incentive
```

## Creating the Evaluation Dataset Function 

In this section, we define a function that transforms our raw dataset into the format required for Ragas evaluation.

The function first checks and converts the retrieved contexts into the correct list format if needed, then combines each user’s question with its related contexts using a template. It sends this complete prompt to the language model with a built-in retry mechanism to handle any errors, and finally compiles the responses into a Ragas Evaluation Dataset. You can read more about it [here](../../concepts/components/eval_dataset.md).


```python
import ast
import time
from tqdm import tqdm
from typing import List, Dict, Any
from ragas.dataset_schema import EvaluationDataset
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()


def create_ragas_evaluation_dataset(df: pd.DataFrame, prompt: str) -> EvaluationDataset:
    """
    Process a DataFrame into an evaluation dataset by:
    1. Converting retrieved contexts from strings to lists if needed
    2. For each sample, formatting a prompt with user input and contexts
    3. Calling the LLM with retry logic (up to 4 attempts)
    4. Recording responses in the dataset
    
    Args:
        df: DataFrame with user_input and retrieved_contexts columns
        prompt: Template string with placeholders for contexts and user input
        
    Returns:
        EvaluationDataset for RAGAS evaluation
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Check if any row has retrieved_contexts as string and convert all to lists
    if df["retrieved_contexts"].apply(type).eq(str).any():
        df["retrieved_contexts"] = df["retrieved_contexts"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    
    # Convert DataFrame to list of dictionaries
    samples: List[Dict[str, Any]] = df.to_dict(orient="records")
    
    # Process each sample
    for sample in tqdm(samples, desc="Processing samples"):
        user_input_str = sample.get("user_input", "")
        retrieved_contexts = sample.get("retrieved_contexts", [])
        
        # Ensure retrieved_contexts is a list
        if not isinstance(retrieved_contexts, list):
            retrieved_contexts = [str(retrieved_contexts)]
        
        # Join contexts and format prompt
        context_str = "\n".join(retrieved_contexts)
        formatted_prompt = prompt.format(
            contexts=context_str, user_input=user_input_str
        )

        # Implement retry logic
        max_attempts = 4  # 1 initial attempt + 3 retries
        for attempt in range(max_attempts):
            if attempt > 0:
                delay = attempt * 10
                print(f"Attempt {attempt} failed. Retrying in {delay} seconds...")
                time.sleep(delay)
            try:
                # Call the OpenAI API
                response = client.chat.completions.create(
                    model="gpt-4o-mini", 
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=0
                )
                sample["response"] = response.choices[0].message.content
                break  # Exit the retry loop if successful
            except Exception as e:
                print(f"Error on attempt {attempt+1}: {str(e)}")
                if attempt == max_attempts - 1:
                    print(f"Failed after {max_attempts} attempts. Skipping sample.")
                    sample["response"] = None

    # Create and return evaluation dataset
    eval_dataset = EvaluationDataset.from_list(data=samples)
    return eval_dataset
```

## Generating Responses for Evaluation

Now we'll use our function to create evaluation datasets for both prompt versions:


```python
# Create evaluation datasets for both prompt versions
print("Generating responses for base prompt...")
eval_dataset_base = create_ragas_evaluation_dataset(eval_df, prompt=diabetes_assistant_prompt)

print("Generating responses for incentive prompt...")
eval_dataset_incentive = create_ragas_evaluation_dataset(eval_df, prompt=incentive_prompt)
```
```
Generating responses for base prompt...
Processing samples: 100%|██████████| 15/15 [00:43<00:00,  2.88s/it]

Generating responses for incentive prompt...
Processing samples: 100%|██████████| 15/15 [00:39<00:00,  2.63s/it]
```

## Queries that should be answered

### Setting Up Evaluation Metrics

Ragas provides several built-in metrics, and we can also create custom metrics for specific requirements. For a list of all available metrics, you can check here.

### Choosing NVIDIA Metrics for Efficient Evaluation

For our evaluation, we'll use [NVIDIA metrics](../../concepts/metrics/available_metrics/nvidia_metrics.md) from the Ragas framework, which offer significant advantages for prompt engineering workflows:

- **Faster computation**: Requires fewer LLM calls than alternative metrics
- **Lower token consumption**: Reduces API costs during iterative testing
- **Robust evaluation**: Provides consistent measurements through dual LLM judgments

These characteristics make NVIDIA metrics particularly suitable for prompt optimization, where multiple iterations and experiments are often necessary.

For our diabetes assistant, we will use:
- [AnswerAccuracy](../../concepts/metrics/available_metrics/nvidia_metrics.md#answer-accuracy): Evaluates how well the model's response aligns with the reference answer.
- [ResponseGroundedness](../../concepts/metrics/available_metrics/nvidia_metrics.md#response-groundedness): Measures whether the response is grounded in the provided context, helping to identify hallucinations or made-up information.



```python
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics import (
    AnswerAccuracy,
    ResponseGroundedness,
)

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

metrics = [
    AnswerAccuracy(llm=evaluator_llm),
    ResponseGroundedness(llm=evaluator_llm),
]
```

### Preparing the Test Dataset


```python
from ragas import evaluate

# Evaluate both datasets with standard metrics (for answerable questions)
answerable_df = eval_df.iloc[:10] # First 10 questions should be answered
answerable_dataset_base = EvaluationDataset.from_list(
    [sample for i, sample in enumerate(eval_dataset_base.to_list()) if i < 10]
)
answerable_dataset_incentive = EvaluationDataset.from_list(
    [sample for i, sample in enumerate(eval_dataset_incentive.to_list()) if i < 10]
)
```

### Running the Evaluation


```python
print("Evaluating answerable questions with base prompt...")
result_answerable_base = evaluate(metrics=metrics, dataset=answerable_dataset_base)
result_answerable_base
```
Output
```
Evaluating answerable questions with base prompt...
Evaluating: 100%|██████████| 20/20 [00:02<00:00,  9.79it/s]

{'nv_accuracy': 0.6750, 'nv_response_groundedness': 1.0000}
```



```python
print("Evaluating answerable questions with incentive prompt...")
result_answerable_incentive = evaluate(metrics=metrics, dataset=answerable_dataset_incentive)
result_answerable_incentive
```
Output
```
Evaluating answerable questions with incentive prompt...
Evaluating: 100%|██████████| 20/20 [00:02<00:00,  9.19it/s]

{'nv_accuracy': 0.6750, 'nv_response_groundedness': 1.0000}
```


Impact of Incentivization:

For queries within the agent’s expertise, incentivization did not affect performance.

- Answer accuracy remains unchanged (0.6750 → 0.6750)
- Response groundedness score remains consistent (1.0000 → 1.0000)

## Queries that should NOT be answered (insufficient expertise)

### Preparing the Test Dataset

Queries that should NOT be answered (insufficient expertise)


```python
non_answerable_df = eval_df.iloc[10:]  # Last 5 questions should NOT be answered
non_answerable_dataset_base = EvaluationDataset.from_list(
    [sample for i, sample in enumerate(eval_dataset_base.to_list()) if i >= 10]
)
non_answerable_dataset_incentive = EvaluationDataset.from_list(
    [sample for i, sample in enumerate(eval_dataset_incentive.to_list()) if i >= 10]
)
```

### Setting Up Evaluation Metrics

Ragas offers several built-in metrics and allows you to create custom metrics tailored to your specific business needs. For our diabetes assistant, we will use the following metrics to evaluate its performance on queries that it should not answer.

### Continuing with NVIDIA Metrics for Efficiency

As with our previous evaluation, we'll use the NVIDIA AnswerAccuracy metric for its computational efficiency and low token consumption. For non-answerable questions, we'll complement this with a custom metric tailored to our specific requirement.

Let's understand what each metric measures:
- [AnswerAccuracy](../../concepts/metrics/available_metrics/nvidia_metrics.md#answer-accuracy): Evaluates how well the model's response aligns with the reference answer.
- Non-Answer Compliance: A custom metric that checks whether the model appropriately declines to answer when required, which is crucial for safety in a medical context. For example, Non-Answer Compliance is built using [AspectCritique](../../concepts/metrics/available_metrics/aspect_critic.md).

Ragas offers flexible tools to create custom metrics that measure your specific business goals. You can learn more about these capabilities by clicking [here](../../concepts/metrics/available_metrics/general_purpose.md).


```python
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics import (
    AnswerAccuracy,
    AspectCritic
)

# Create a specialized metric for evaluating when the model should NOT answer
no_answer_metric = AspectCritic(
    name="Non-Answer Compliance",
    definition="Return 1 if both reference and response appropriately decline to provide an answer on the same grounds (e.g., medical expertise limitations); return 0 if the response provides any answer when the reference declines to answer.",
    llm=evaluator_llm,
)

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

metrics = [
    AnswerAccuracy(llm=evaluator_llm),
    no_answer_metric,
]
```

### Running the Evaluation


```python
print("Evaluating non-answerable questions with base prompt...")
result_non_answerable_base = evaluate(metrics=metrics, dataset=non_answerable_dataset_base)
result_non_answerable_base
```
Output
```
Evaluating non-answerable questions with base prompt...
Evaluating: 100%|██████████| 10/10 [00:01<00:00,  5.44it/s]

{'nv_accuracy': 0.6000, 'Non-Answer Compliance': 0.4000}
```

```python
print("Evaluating non-answerable questions with incentive prompt...")
result_non_answerable_incentive = evaluate(metrics=metrics, dataset=non_answerable_dataset_incentive)
result_non_answerable_incentive
```
Output
```
Evaluating non-answerable questions with incentive prompt...
Evaluating: 100%|██████████| 10/10 [00:01<00:00,  6.28it/s]

{'nv_accuracy': 0.7000, 'Non-Answer Compliance': 0.6000}
```

Impact of Incentivization:

The incentivized prompt showed a slight improvement in answer accuracy (0.6 → 0.7)
Most importantly, the incentivized prompt was significantly better at declining to answer questions outside its expertise (40% → 60%)

## Iterative Improvement Process

Leveraging our evaluation metrics, we now adopt a data-driven approach to refine our prompt strategies. The process unfolds as follows:

1.	Establish a Baseline: Begin with an initial prompt.
2.	Performance Evaluation: Measure its performance using our defined metrics.
3.	Targeted Analysis: Identify shortcomings and implement focused improvements.
4.	Re-Evaluation: Test the revised prompt.
5.	Adopt and Iterate: Retain the version that performs better and repeat the cycle.

## Conclusion
This systematic approach offers clear advantages over a reactive “whack-a-mole” strategy:
- It quantifies improvements across all key requirements simultaneously.
- It maintains a consistent, reproducible testing framework.
- It enables immediate detection of any regressions.
- It bases decisions on objective data rather than intuition.

Through these iterative refinements, we steadily progress towards an optimal and robust prompt strategy.
