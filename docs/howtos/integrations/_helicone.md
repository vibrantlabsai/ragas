# Helicone

This notebook demonstrates how to integrate Helicone with Ragas for monitoring and evaluating RAG (Retrieval-Augmented Generation) systems.

## Prerequisites

Before you begin, make sure you have a Helicone account and API key:

1. Log into [Helicone](https://www.helicone.ai) or create an account if you don't have one.
2. Once logged in, navigate to the [Developer section](https://helicone.ai/developer) to generate an API key.

**Note**: Make sure to generate a write-only API key. For more information on Helicone authentication, refer to the [Helicone Auth documentation](https://docs.helicone.ai/getting-started/helicone-api-keys).

Store your Helicone API key securely, as you'll need it for the integration.

## Setup

First, let's install the required packages and set up our environment.


```python
!pip install datasets ragas openai
```


```python
import os

from datasets import Dataset

from ragas import evaluate
from ragas.integrations.helicone import helicone_config  # import helicone_config
from ragas.metrics import answer_relevancy, context_precision, faithfulness

# Set up Helicone
HELICONE_API_KEY = (
    "your_helicone_api_key_here"  # Replace with your actual Helicone API key
)
helicone_config.api_key = HELICONE_API_KEY
os.environ["OPENAI_API_KEY"] = (
    "your_openai_api_key_here"  # Replace with your actual OpenAI API key
)

# Verify Helicone API key is set
if HELICONE_API_KEY == "your_helicone_api_key_here":
    raise ValueError(
        "Please replace 'your_helicone_api_key_here' with your actual Helicone API key."
    )
```

## Prepare Data

Let's prepare some sample data for our RAG system evaluation.


```python
data_samples = {
    "question": ["When was the first Super Bowl?", "Who has won the most Super Bowls?"],
    "answer": [
        "The first Super Bowl was held on January 15, 1967.",
        "The New England Patriots have won the most Super Bowls, with six championships.",
    ],
    "contexts": [
        [
            "The First AFL–NFL World Championship Game, later known as Super Bowl I, was played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles, California."
        ],
        [
            "As of 2021, the New England Patriots have won the most Super Bowls with six championships, all under the leadership of quarterback Tom Brady and head coach Bill Belichick."
        ],
    ],
    "ground_truth": [
        "The first Super Bowl was held on January 15, 1967.",
        "The New England Patriots have won the most Super Bowls, with six championships as of 2021.",
    ],
}

dataset = Dataset.from_dict(data_samples)
print(dataset)
```

## Evaluate with Ragas

Now, let's use Ragas to evaluate our RAG system. Helicone will automatically log the API calls made during this evaluation.


```python
# Evaluate using Ragas
score = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])

# Display results
print(score.to_pandas())
```

## Viewing Results in Helicone

The API calls made during the Ragas evaluation are automatically logged in Helicone. You can view these logs in the Helicone dashboard to get insights into the performance and behavior of your RAG system.

To view the results:
1. Go to the [Helicone dashboard](https://www.helicone.ai/dashboard)
2. Navigate to the 'Requests' section
3. You should see the API calls made during the Ragas evaluation

You can analyze these logs to understand:
- The number of API calls made during evaluation
- The performance of each call (latency, tokens used, etc.)
- Any errors or issues that occurred during the evaluation

This integration allows you to combine the power of Ragas for RAG system evaluation with Helicone's robust monitoring and analytics capabilities.
