# Datasets and Experiment Results

When we evaluate AI systems, we typically work with two main types of data:

1. **Evaluation Datasets**: These are stored under the `datasets` directory.
2. **Evaluation Results**: These are stored under the `experiments` directory.

## Evaluation Datasets

A dataset for evaluations contains:

1. Inputs: a set of inputs that the system will process.
2. Expected outputs (Optional): the expected outputs or responses from the system for the given inputs.
3. Metadata (Optional): additional information that can be stored alongside the dataset.

For example, in a Retrieval-Augmented Generation (RAG) system it might include query (input to the system), Grading notes (to grade the output from the system), and metadata like query complexity.

Metadata is particularly useful for slicing and dicing the dataset, allowing you to analyze results across different facets. For instance, you might want to see how your system performs on complex queries versus simple ones, or how it handles different languages.

## Experiment Results

Experiment results include:

1. All attributes from the dataset.
2. The response from the evaluated system.
3. Results of metrics.
4. Optional metadata, such as a URI pointing to the system trace for a given input.

For example, in a RAG system, the results might include Query, Grading notes, Response, Accuracy score (metric), link to the system trace, etc.

## Working with Datasets in Ragas

Ragas provides a `Dataset` class to work with evaluation datasets. Here's how you can use it:

### Creating a Dataset

```python
from ragas import Dataset

# Create a new dataset
dataset = Dataset(name="my_evaluation", backend="local/csv", root_dir="./data")

# Add a sample to the dataset
dataset.append(
    {
        "id": "sample_1",
        "query": "What is the capital of France?",
        "expected_answer": "Paris",
        "metadata": {"complexity": "simple", "language": "en"},
    }
)
```

### Loading an Existing Dataset

```python
# Load an existing dataset
dataset = Dataset.load(name="my_evaluation", backend="local/csv", root_dir="./data")
```

### Dataset Structure

Datasets in Ragas are flexible and can contain any fields you need for your evaluation. Common fields include:

- `id`: Unique identifier for each sample
- `query` or `input`: The input to your AI system
- `expected_output` or `ground_truth`: The expected response (if available)
- `metadata`: Additional information about the sample

### Best Practices for Dataset Creation

1. **Representative Samples**: Ensure your dataset represents the real-world scenarios your AI system will encounter.

2. **Balanced Distribution**: Include samples across different difficulty levels, topics, and edge cases.

3. **Quality Over Quantity**: It's better to have fewer high-quality, well-curated samples than many low-quality ones.

4. **Metadata Rich**: Include relevant metadata that allows you to analyze performance across different dimensions.

5. **Version Control**: Track changes to your datasets over time to ensure reproducibility.

## Dataset Storage and Management

### Local Storage

For local development and small datasets, you can use CSV files:

```python
dataset = Dataset(name="my_eval", backend="local/csv", root_dir="./datasets")
```

### Cloud Storage

For larger datasets or team collaboration, consider cloud backends:

```python
# Google Drive (experimental)
dataset = Dataset(name="my_eval", backend="gdrive", root_dir="folder_id")

# Other backends can be added as needed
```

### Dataset Versioning

Keep track of dataset versions for reproducible experiments:

```python
# Include version in dataset name
dataset = Dataset(name="my_eval_v1.2", backend="local/csv", root_dir="./datasets")
```

## Integration with Evaluation Workflows

Datasets integrate seamlessly with Ragas evaluation workflows:

```python
from ragas import experiment, Dataset

# Load your dataset
dataset = Dataset.load(name="my_evaluation", backend="local/csv", root_dir="./data")


# Define your experiment
@experiment()
async def my_experiment(row):
    # Process the input through your AI system
    response = await my_ai_system(row["query"])

    # Return results for metric evaluation
    return {
        **row,  # Include original data
        "response": response,
        "experiment_name": "baseline_v1",
    }


# Run evaluation on the dataset
results = await my_experiment.arun(dataset)
```

This integration allows you to maintain a clear separation between your test data (datasets) and your evaluation results (experiments), making it easier to track progress and compare different approaches.
