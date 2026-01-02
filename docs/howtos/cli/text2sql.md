# Text-to-SQL Evaluation Quickstart

The `text2sql` template evaluates text-to-SQL systems by comparing SQL execution results.

## Create the Project

```sh
ragas quickstart text2sql
cd text2sql
```

## Install Dependencies

```sh
uv sync
```

## Set Your API Key

```sh
export OPENAI_API_KEY="your-openai-key"
```

## Run the Evaluation

```sh
uv run python evals.py
```

## Project Structure

```
text2sql/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── text2sql_agent.py      # Text-to-SQL agent
├── db_utils.py            # Database utilities
├── evals.py               # Evaluation workflow
├── prompt.txt             # Base prompt template
├── prompt_v2.txt          # Improved prompt v2
├── prompt_v3.txt          # Improved prompt v3
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/
    │   └── booksql_sample.csv  # Sample book database queries
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## What It Evaluates

The template evaluates text-to-SQL generation:

- **Agent**: Converts natural language to SQL queries
- **Database**: Sample book database with authors, titles, genres
- **Test Cases**: Natural language questions → expected SQL queries
- **Metric**: Execution accuracy by comparing query results using datacompy

## Understanding the Code

### The Agent (`text2sql_agent.py`)

Converts natural language to SQL:

```python
from text2sql_agent import Text2SQLAgent

agent = Text2SQLAgent(client=openai_client)
sql = await agent.generate_sql("Find all books by Jane Austen")
```

### The Evaluation (`evals.py`)

Compares execution results:

```python
@discrete_metric(name="execution_accuracy", allowed_values=["correct", "incorrect"])
def execution_accuracy(expected_sql: str, predicted_success: bool, predicted_result):
    # Executes both SQLs and compares results using datacompy
    # Returns "correct" if results match, "incorrect" otherwise
```

## Test Data

The template includes `evals/datasets/booksql_sample.csv` with sample questions and expected SQL queries for a book database.

## Customization

### Use Your Own Database

Update `db_utils.py` to connect to your database:

```python
def get_db_connection():
    return sqlite3.connect("your_database.db")
```

### Try Different Prompts

The template includes three prompt versions in `prompt.txt`, `prompt_v2.txt`, and `prompt_v3.txt`. Test each to see which works best.

## Next Steps

- [Agent Evaluation](agent_evals.md) - Evaluate AI agents
- [Workflow Evaluation](workflow_eval.md) - Evaluate complex workflows
