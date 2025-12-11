# Improve RAG Quickstart

The `improve_rag` template demonstrates how to compare different RAG approaches using real-world evaluation data. It includes naive (single retrieval) and agentic (multi-step retrieval) RAG modes.

## Create the Project

```sh
# Using uvx (no installation required)
uvx ragas quickstart improve_rag
cd improve_rag

# Or with ragas installed
ragas quickstart improve_rag
cd improve_rag
```

## Install Dependencies

```sh
uv sync
```

Or with pip:

```sh
pip install -e .
```

## Set Your API Key

```sh
export OPENAI_API_KEY="your-openai-key"
```

## Run the Evaluation

### Naive RAG Mode (Default)

```sh
uv run python evals.py
```

### Agentic RAG Mode

```sh
uv run python evals.py --agentic
```

!!! note "Agentic Mode Requirements"
    Agentic mode requires the `openai-agents` package. Install it with:
    ```sh
    pip install openai-agents
    ```

## Optional: MLflow Tracing

For detailed tracing of LLM calls, start MLflow before running:

```sh
mlflow ui --port 5000
```

Then run your evaluation. Traces will be automatically sent to MLflow if the server is running.

## Project Structure

```
improve_rag/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── rag.py                 # RAG implementation (naive & agentic)
├── evals.py               # Evaluation workflow
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/          # Test datasets (hf_doc_qa_eval.csv)
    ├── experiments/       # Evaluation results
    └── logs/              # Evaluation logs
```

## Understanding the RAG Modes

### Naive RAG

The naive approach performs a single retrieval step:

1. **Query** → BM25 retrieves top-k documents
2. **Context** → Retrieved documents form the context
3. **Generate** → LLM generates response from context

```python
rag = RAG(llm_client=client, retriever=retriever, mode="naive")
result = await rag.query("What is the Diffusers library?")
```

**Pros:**

- Simple and fast
- Predictable latency
- Lower cost (single LLM call)

**Cons:**

- May miss relevant documents with different terminology
- No query refinement
- Limited to single retrieval strategy

### Agentic RAG

The agentic approach lets an agent control the retrieval:

1. **Query** → Agent analyzes the question
2. **Search** → Agent decides what to search for (multiple searches possible)
3. **Refine** → Agent can refine searches based on results
4. **Generate** → Agent synthesizes final answer

```python
rag = RAG(llm_client=client, retriever=retriever, mode="agentic")
result = await rag.query("What command uploads an ESPnet model?")
```

**Pros:**

- Can try multiple search strategies
- Better at finding specific technical information
- Adapts search based on initial results

**Cons:**

- Higher latency (multiple LLM calls)
- Higher cost
- Less predictable behavior

## The Evaluation Dataset

The template includes `hf_doc_qa_eval.csv` with questions about HuggingFace documentation:

| Field | Description |
|-------|-------------|
| `question` | Technical question about HuggingFace tools |
| `expected_answer` | Ground truth answer |

Example questions:

- "What is the default checkpoint used by the sentiment analysis pipeline?"
- "What command is used to upload an ESPnet model?"
- "What is the purpose of the Diffusers library?"

## Understanding the Code

### The RAG Implementation (`rag.py`)

#### BM25Retriever

Uses BM25 (Best Matching 25) algorithm for document retrieval:

```python
class BM25Retriever:
    def __init__(self, dataset_name="m-ric/huggingface_doc"):
        # Loads HuggingFace documentation
        # Splits into chunks for better retrieval
        # Creates BM25 index

    def retrieve(self, query: str, top_k: int = 3):
        # Returns top-k most relevant documents
```

#### RAG Class

Unified interface for both modes:

```python
class RAG:
    def __init__(self, llm_client, retriever, mode="naive"):
        self.mode = mode
        if mode == "agentic":
            self._setup_agent()

    async def query(self, question: str, top_k: int = 3):
        if self.mode == "naive":
            return await self._naive_query(question, top_k)
        else:
            return await self._agentic_query(question, top_k)
```

### The Evaluation Script (`evals.py`)

The correctness metric compares model responses to expected answers:

```python
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="""Compare the model response to the expected answer...
    Return 'pass' if correct, 'fail' if incorrect.""",
    allowed_values=["pass", "fail"],
)
```

## Customization

### Change the Knowledge Base

Replace HuggingFace docs with your own documents:

```python
class CustomRetriever:
    def __init__(self, documents: list[str]):
        from langchain_community.retrievers import BM25Retriever
        self.retriever = BM25Retriever.from_texts(documents)

    def retrieve(self, query: str, top_k: int = 3):
        self.retriever.k = top_k
        return self.retriever.invoke(query)
```

### Use a Different Model

Change the model in `evals.py`:

```python
# Use GPT-4 for better accuracy
rag = RAG(llm_client=client, retriever=retriever, model="gpt-4o")

# Or use a different provider
from anthropic import Anthropic
client = Anthropic()
# Note: Would need to modify rag.py for non-OpenAI clients
```

### Add Custom Metrics

Evaluate additional aspects:

```python
from ragas.metrics import NumericalMetric

completeness = NumericalMetric(
    name="completeness",
    prompt="""How complete is the response (1-5)?
    Question: {question}
    Expected: {expected_answer}
    Response: {response}
    Score:""",
    allowed_values=(1, 5),
)

# Add to experiment
result = {
    **row,
    "correctness": correctness_score.value,
    "completeness": completeness.score(...).value,
}
```

### Modify the Agent Behavior

Customize the agentic search strategy in `rag.py`:

```python
def _setup_agent(self):
    @function_tool
    def retrieve(query: str) -> str:
        """Custom tool description..."""
        docs = self.retriever.retrieve(query, self.default_k)
        return "\n\n".join([doc.page_content for doc in docs])

    self._agent = Agent(
        name="Custom RAG Assistant",
        instructions="Your custom instructions...",
        tools=[retrieve]
    )
```

## Comparing Results

Run both modes and compare:

```sh
# Run naive mode
uv run python evals.py
# Results saved to experiments/YYYYMMDD-HHMMSS_naiverag.csv

# Run agentic mode
uv run python evals.py --agentic
# Results saved to experiments/YYYYMMDD-HHMMSS_agenticrag.csv
```

Analyze the results:

```python
import pandas as pd

naive = pd.read_csv("evals/experiments/..._naiverag.csv")
agentic = pd.read_csv("evals/experiments/..._agenticrag.csv")

print(f"Naive pass rate: {(naive['correctness_score'] == 'pass').mean():.1%}")
print(f"Agentic pass rate: {(agentic['correctness_score'] == 'pass').mean():.1%}")
```

## Troubleshooting

### MLflow Warnings

If you see MLflow warnings about failed traces, either:

1. Start MLflow: `mlflow ui --port 5000`
2. Or ignore them - the evaluation still works without tracing

### Agentic Mode Not Working

Ensure you have the agents package:

```sh
pip install openai-agents
```

### Slow First Run

The first run downloads the HuggingFace documentation dataset (~300MB). Subsequent runs use the cached data.

## Next Steps

- [RAG Evaluation Guide](rag_eval.md) - Simpler evaluation setup
- [Custom Metrics](../customizations/metrics/_write_your_own_metric.md) - Write your own metrics
- [Evaluate and Improve RAG](../applications/evaluate-and-improve-rag.md) - Production RAG evaluation
