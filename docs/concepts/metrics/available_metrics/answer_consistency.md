## Answer Consistency

The **Answer Consistency** metric measures the robustness of a RAG system by evaluating how consistent the answers are when the same question is asked in different ways (e.g., paraphrased queries). This metric is crucial for assessing whether your system provides stable, reliable answers regardless of how users phrase their questions.

A high consistency score indicates that your RAG system is robust and provides semantically similar answers to query variations, while a low score suggests the system is sensitive to query phrasing and may produce inconsistent results.

### Key Features

- **Robustness Testing**: Evaluates system stability across query variations
- **Semantic Similarity**: Uses embeddings to measure answer similarity
- **Pairwise Comparison**: Computes consistency across all response pairs
- **Score Range**: 0.0 (completely inconsistent) to 1.0 (perfectly consistent)

### Example

```python
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import AnswerConsistency

# Setup embeddings
client = AsyncOpenAI()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", client=client)

# Create metric
scorer = AnswerConsistency(embeddings=embeddings)

# Evaluate consistency across query variations
# These are responses to paraphrased versions of the same question
result = await scorer.ascore(
    multi_responses=[
        "Paris is the capital of France.",
        "The capital city of France is Paris.",
        "France's capital is Paris.",
    ]
)
print(f"Answer Consistency Score: {result.value}")
```

Output:

```
Answer Consistency Score: 0.95
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        multi_responses=[
            "Paris is the capital of France.",
            "The capital city of France is Paris.",
            "France's capital is Paris.",
        ]
    )
    ```

### How It's Calculated

!!! example

    **Original Query**: "What is the capital of France?"
    
    **Query Variations**:
    - "Which city is the capital of France?"
    - "Tell me France's capital city"
    - "What's the capital city of France?"
    
    **Responses from RAG System**:
    1. "Paris is the capital of France."
    2. "The capital city of France is Paris."
    3. "France's capital is Paris."

Let's examine how consistency is calculated:

- **Step 1:** Embed each response using the specified embedding model
- **Step 2:** Normalize all embedding vectors
- **Step 3:** Compute pairwise cosine similarity between all response pairs
  - Response 1 ↔ Response 2: 0.98
  - Response 1 ↔ Response 3: 0.96
  - Response 2 ↔ Response 3: 0.97
- **Step 4:** Average all pairwise similarities: (0.98 + 0.96 + 0.97) / 3 = **0.97**

### Use Cases

1. **Robustness Testing**: Verify your RAG system handles query variations gracefully
2. **Quality Assurance**: Ensure consistent user experience across different phrasings
3. **Model Comparison**: Compare different retrieval or generation models for stability
4. **Prompt Engineering**: Test if prompt changes improve consistency
5. **A/B Testing**: Measure if system updates maintain or improve consistency

### Practical Workflow

To use this metric effectively:

1. **Generate Query Variations**: Create paraphrases of your test queries
   ```python
   # Example: Use an LLM to generate paraphrases
   original = "What is the capital of France?"
   variations = [
       "Which city is the capital of France?",
       "Tell me France's capital city",
       "What's the capital city of France?"
   ]
   ```

2. **Get Responses**: Run each variation through your RAG system
   ```python
   responses = []
   for query in [original] + variations:
       response = your_rag_system.query(query)
       responses.append(response)
   ```

3. **Evaluate Consistency**: Measure how similar the responses are
   ```python
   result = await scorer.ascore(multi_responses=responses)
   ```

### Configuration Options

```python
# Adjust minimum required responses (default: 2)
scorer = AnswerConsistency(
    embeddings=embeddings,
    min_responses=3  # Require at least 3 responses
)
```

### Interpreting Scores

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| **0.9 - 1.0** | Excellent consistency | System is very robust |
| **0.7 - 0.9** | Good consistency | Minor variations acceptable |
| **0.5 - 0.7** | Moderate consistency | Review retrieval/generation |
| **< 0.5** | Poor consistency | Investigate system issues |

### Common Issues and Solutions

**Low Consistency Scores:**

1. **Retrieval Sensitivity**: Different phrasings retrieve different contexts
   - *Solution*: Improve query understanding or use query expansion

2. **Generation Variability**: LLM produces different answers from similar contexts
   - *Solution*: Use lower temperature, add consistency constraints to prompts

3. **Context Ranking Issues**: Relevant contexts ranked differently across queries
   - *Solution*: Improve reranking or use more stable retrieval methods

### Example: Complete Evaluation Pipeline

```python
import asyncio
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import AnswerConsistency

async def evaluate_rag_consistency():
    # Setup
    client = AsyncOpenAI()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", client=client)
    scorer = AnswerConsistency(embeddings=embeddings)
    
    # Test case: Query variations
    test_cases = {
        "capital_france": [
            "Paris is the capital of France.",
            "The capital city of France is Paris.",
            "France's capital is Paris.",
            "Paris serves as France's capital city.",
        ],
        "einstein_theory": [
            "Einstein developed the theory of relativity.",
            "The theory of relativity was created by Albert Einstein.",
            "Albert Einstein is known for his theory of relativity.",
        ]
    }
    
    # Evaluate each test case
    for test_name, responses in test_cases.items():
        result = await scorer.ascore(multi_responses=responses)
        print(f"{test_name}: {result.value:.3f}")
        
        if result.value < 0.7:
            print(f"  ⚠️  Low consistency detected!")

if __name__ == "__main__":
    asyncio.run(evaluate_rag_consistency())
```

Output:
```
capital_france: 0.956
einstein_theory: 0.923
```

## Legacy Metrics API

The following examples use the legacy metrics API pattern. For new projects, we recommend using the collections-based API shown above.

!!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

### Example with SingleTurnSample

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerConsistency
from ragas.embeddings import LangchainEmbeddingsWrapper

sample = SingleTurnSample(
    multi_responses=[
        "Paris is the capital of France.",
        "The capital city of France is Paris.",
        "France's capital is Paris.",
    ]
)

scorer = AnswerConsistency(embeddings=LangchainEmbeddingsWrapper(evaluator_embedding))
score = await scorer.single_turn_ascore(sample)
print(f"Consistency: {score}")
```

Output:

```
0.9534
```