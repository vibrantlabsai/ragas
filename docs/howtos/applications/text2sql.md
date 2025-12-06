# How to evaluate a Text to SQL Agent

In this guide, you'll learn how to systematically evaluate and improve a text-to-SQL system using Ragas.

What you'll accomplish:

- Set up a baseline text-to-SQL system for evaluation
- Learn how to create evaluation metrics 
- Build a reusable evaluation pipeline for your SQL agent  
- Implement improvements based on error analysis

## Setup your environment

We've created a simple module you can install and run so that you can focus on understanding the evaluation process instead of creating the application.

```bash
uv pip install "ragas-examples[text2sql]"
```

## Quick agent test

Test the text-to-SQL agent to see it convert natural language to SQL:

```python
import os
import asyncio
from openai import AsyncOpenAI
from ragas_examples.text2sql.text2sql_agent import Text2SQLAgent

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create agent
openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
agent = Text2SQLAgent(client=openai_client, model_name="gpt-5-mini")

# Test with a sample query
test_query = "How much open credit does customer Andrew Bennett?"
result = asyncio.run(agent.query(test_query))

print(f"Natural Query: {result['query']}")
print(f"Generated SQL: {result['sql']}")
```

??? note "Output"
    ```python
    Natural Query: How much open credit does customer Andrew Bennett?
    Generated SQL: select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Andrew Bennett" )
    ```

This generates SQL from the natural language query. Now let's build a systematic evaluation process.

### Download BookSQL 

Before running the agent or database utilities, download the gated BookSQL dataset from Hugging Face:

```bash
huggingface-cli login
uv run python -m ragas_examples.text2sql.data_utils --download-data
```

If you see authentication errors, visit the dataset page and accept terms first: [BookSQL on Hugging Face](https://huggingface.co/datasets/Exploration-Lab/BookSQL)

!!! note "Full code"
    You can view the full code for the agent and evaluation pipeline [here](https://github.com/vibrantlabsai/ragas/tree/main/examples/ragas_examples/text2sql).

## Prepare your dataset

We've prepared a balanced sample dataset with 99 examples (33 each of easy, medium, and hard queries) from the BookSQL dataset. You can start evaluating immediately or create your own dataset following the next section. 

**Download and examine the sample dataset:**

```bash
# Download the sample CSV from GitHub
curl -o booksql_sample.csv https://raw.githubusercontent.com/vibrantlabsai/ragas/main/examples/ragas_examples/text2sql/datasets/booksql_sample.csv
# View the first few rows to understand the structure
head -5 booksql_sample.csv
```

| Query                                                        | SQL                                                                                                                                                                                                                                    | Levels | split |
|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|-------|
| What is the balance due from Richard Aguirre?                | select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Richard Aguirre" )                                                                                               | medium | train |
| What is the balance due from Sarah Oconnor?                  | select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Sarah Oconnor" )                                                                                                 | medium | train |
| What is my average invoice from Jeffrey Moore?               | select avg(amount) from (select distinct transaction_id, amount from master_txn_table where customers = "Jeffrey Moore" and transaction_type = 'invoice')                                                                              | hard   | train |
| How much open credit does customer Andrew Bennett?           | select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Andrew Bennett" )                                                                                                | easy   | train |

??? info "📋 Optional: How we prepared the sample dataset"

    Download and examine the dataset

    For this guide, we'll use the [BookSQL dataset](https://huggingface.co/datasets/Exploration-Lab/BookSQL). Skip this section if you have your own dataset.

    **Download the dataset:**

    ```bash
    export HF_TOKEN=your-huggingface-token
    uv run python -m ragas_examples.text2sql.data_utils --download-data
    ```

    **Note:** BookSQL is gated. Visit [the dataset page](https://huggingface.co/datasets/Exploration-Lab/BookSQL), accept terms, and run `huggingface-cli login` if you encounter authentication errors.

    **Examine the dataset structure:**

    ```bash
    # Check the database schema
    sqlite3 BookSQL-files/BookSQL/accounting.sqlite ".schema" | head -20
    ```

    **Expected schema output:**

    ```sql
    CREATE TABLE master_txn_table(
                        id INTEGER ,
                        businessID INTEGER NOT NULL ,
                        Transaction_ID INTEGER NOT NULL,
                        Transaction_DATE DATE NOT NULL,
                        Transaction_TYPE TEXT NOT NULL,
                        Amount DOUBLE NOT NULL,
                        CreatedDATE DATE NOT NULL,
                        CreatedUSER TEXT NOT NULL,
                        Account TEXT NOT NULL,
                        AR_paid TEXT,
                        AP_paid TEXT,
                        Due_DATE DATE,
                        Open_balance DOUBLE,
                        Customers TEXT,
                        Vendor TEXT,
                        Product_Service TEXT,
                        Quantity INTEGER,
                        Rate DOUBLE,
                        Credit DOUBLE,
    ```

    The dataset contains:

    - **Database**: SQLite file with accounting data (invoices, clients, etc.)
    - **Questions**: Natural language queries in English
    - **SQL**: Corresponding SQL queries
    - **Difficulty levels**: Easy, Medium, Hard categories

    Create a balanced evaluation subset:

    ```bash
    uv run python -m ragas_examples.text2sql.data_utils --create-sample --samples 33 --validate --require-data
    ```

    This creates a balanced CSV with validated queries that return actual data.

    **Expected output:**

    ```
    📖 Loading data from BookSQL-files/BookSQL/train.json...
    📊 Loaded 70828 total records
    🚂 Found 70828 train records
    🔍 Removed 35189 duplicate records (same Query + SQL)
    📊 35639 unique records remaining
    📈 Difficulty distribution (after deduplication):
       • medium: 20576 records
       • hard: 11901 records
       • easy: 3162 records
    ✅ Added 33 validated 'easy' records
    ✅ Added 33 validated 'medium' records
    ✅ Added 33 validated 'hard' records
    💾 Saved 99 records to datasets/booksql_sample.csv
    📋 Final distribution:
       • medium: 33 records
       • hard: 33 records
       • easy: 33 records
    ```

    This creates `datasets/booksql_sample.csv` with 99 balanced examples across difficulty levels. 


BookSQL is released under CC BY-NC-SA (non‑commercial only). See details and citation below.

??? "📋 Licensing & citation details"

    !!! warning "License and usage"
        The BookSQL dataset is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. You may use it for non‑commercial research only. Commercial usage is not allowed.

    - **Dataset**: [`Exploration-Lab/BookSQL` on Hugging Face](https://huggingface.co/datasets/Exploration-Lab/BookSQL) · [GitHub repository](https://github.com/Exploration-Lab/BookSQL)
    - **Paper**: ACL Anthology — [BookSQL: A Large Scale Text-to-SQL Dataset for Accounting Domain](https://aclanthology.org/2024.naacl-long.28/)

    If you use BookSQL in your research, please cite the paper:

    ```bibtex
    @inproceedings{kumar-etal-2024-booksql,
        title = {BookSQL: A Large Scale Text-to-SQL Dataset for Accounting Domain},
        author = {Kumar, Rahul and Raja, Amar and Harsola, Shrutendra and Subrahmaniam, Vignesh and Modi, Ashutosh},
        booktitle = {Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
        month = {June},
        year = {2024},
        address = {Mexico City, Mexico},
        publisher = {Association for Computational Linguistics},
    }
    ```

For advice on how to create your own evaluation dataset, refer to [Datasets - Core Concepts](/concepts/datasets/).

## Set up your text-to-SQL system

### Create your prompt

**Extract the database schema:**

```bash
uv run python -m ragas_examples.text2sql.db_utils --schema
```

??? "📋 Expected schema output"

    ```
    === Database Schema ===
                 name  type                                     sql
    chart_of_accounts table CREATE TABLE chart_of_accounts(
                             id INTEGER ,
                             businessID INTEGER NOT NULL,
                             Account_name TEXT NOT NULL,
                             Account_type TEXT NOT NULL,
                             PRIMARY KEY(id,businessID,Account_name)
                             )
            customers table CREATE TABLE customers(
                             id INTEGER ,
                             businessID INTEGER NOT NULL,
                             customer_name TEXT NOT NULL,
                             customer_full_name TEXT ,
                             ... (continues for all columns)
                             PRIMARY KEY(id,businessID,Customer_name)
                             )
    ... (continues for all 7 tables with complete DDL)
    ```

**Write the prompt content:**

Our prompt follows this template structure:

```text
You are a SQL query generator for a business accounting database. Convert natural language queries to SQL queries.

DATABASE CONTEXT:
This is an accounting database (accounting.sqlite) containing business transaction and entity data.

TABLES AND THEIR PURPOSE:

- master_txn_table: Main transaction records for all business transactions
- chart_of_accounts: Account names and their types for all businesses  
- products_service: Products/services and their types used by businesses
- customers: Customer records with billing/shipping details
- vendors: Vendor records with billing address details
- payment_method: Payment methods used by businesses
- employees: Employee details including name, ID, hire date

DATABASE SCHEMA (DDL):

[Complete DDL statements for all tables]

INSTRUCTIONS:
Convert the user's natural language query into a valid SQL SELECT query. Return only the SQL query, no explanations or formatting.
```

## Define evaluation metrics

For text-to-SQL systems, we need metrics that evaluate the accuracy of results. We'll use execution accuracy as our primary metric to validate that generated SQL returns the correct data.

**Execution Accuracy Metric**: Compares the actual results between expected and predicted SQL queries using [datacompy](https://github.com/capitalone/datacompy). This validates that both queries return the same data, which is the ultimate test of correctness.

The evaluation system classifies results as:

- `"correct"`: Query succeeds and matches expected results  
- `"incorrect"`: Query doesn't succeed or succeeds but returns wrong results


### Setting up metric functions

Create your evaluation metrics using [Ragas discrete metrics](/concepts/metrics/overview). 

```python
# File: examples/ragas_examples/text2sql/evals.py
from ragas.metrics.discrete import discrete_metric
from ragas.metrics.result import MetricResult
from ragas_examples.text2sql.db_utils import execute_sql

@discrete_metric(name="execution_accuracy", allowed_values=["correct", "incorrect"])
def execution_accuracy(expected_sql: str, predicted_success: bool, predicted_result):
    """Compare execution results of predicted vs expected SQL using datacompy."""
    try:
        # Execute expected SQL
        expected_success, expected_result = execute_sql(expected_sql)
        if not expected_success:
            return MetricResult(
                value="incorrect",
                reason=f"Expected SQL failed to execute: {expected_result}"
            )
        
        # If predicted SQL fails, it's incorrect
        if not predicted_success:
            return MetricResult(
                value="incorrect",
                reason=f"Predicted SQL failed to execute: {predicted_result}"
            )
        
        # Both queries succeeded - compare DataFrames using datacompy
        if isinstance(expected_result, pd.DataFrame) and isinstance(predicted_result, pd.DataFrame):
            # Handle empty DataFrames
            if expected_result.empty and predicted_result.empty:
                return MetricResult(value="correct", reason="Both queries returned empty results")
            
            if expected_result.empty != predicted_result.empty:
                return MetricResult(
                    value="incorrect",
                    reason=f"Expected returned {len(expected_result)} rows, predicted returned {len(predicted_result)} rows"
                )
            
            # Use datacompy to compare DataFrames with index-based comparison
            comparison = datacompy.Compare(
                expected_result.reset_index(drop=True), 
                predicted_result.reset_index(drop=True),
                on_index=True,  # Compare row-by-row by index position
                abs_tol=1e-10,  # Very small tolerance for floating point comparison
                rel_tol=1e-10,
                df1_name='expected',
                df2_name='predicted'
            )
            
            if comparison.matches():
                return MetricResult(
                    value="correct",
                    reason=f"DataFrames match exactly ({len(expected_result)} rows, {len(expected_result.columns)} columns)"
                )
            else:
                return MetricResult(
                    value="incorrect",
                    reason="DataFrames do not match - different data returned"
                )
                
    except Exception as e:
        return MetricResult(
            value="incorrect",
            reason=f"Execution accuracy evaluation failed: {str(e)}"
        )
```

### The experiment function

The [experiment function](/concepts/experimentation) orchestrates the complete evaluation pipeline - running the text-to-SQL agent and computing metrics for each query:

```python
# File: examples/ragas_examples/text2sql/evals.py
from typing import Optional
from openai import AsyncOpenAI
from ragas import experiment
from ragas_examples.text2sql.text2sql_agent import Text2SQLAgent
from ragas_examples.text2sql.db_utils import execute_sql

@experiment()
async def text2sql_experiment(
    row,
    model: str,
    prompt_file: Optional[str],
):
    """Experiment function for text-to-SQL evaluation."""
    # Create text-to-SQL agent
    openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    agent = Text2SQLAgent(
        client=openai_client,
        model_name=model,
        prompt_file=prompt_file
    )
    
    # Generate SQL from natural language query
    result = await agent.query(row["Query"])

    # Execute predicted SQL
    try:
        predicted_success, predicted_result = execute_sql(result["sql"])
    except Exception as e:
        predicted_success, predicted_result = False, f"SQL execution failed: {str(e)}"

    # Score the response using execution accuracy
    accuracy_score = await execution_accuracy.ascore(
        expected_sql=row["SQL"],
        predicted_success=predicted_success,
        predicted_result=predicted_result,
    )

    return {
        "query": row["Query"],
        "expected_sql": row["SQL"],
        "predicted_sql": result["sql"],
        "level": row["Levels"],
        "execution_accuracy": accuracy_score.value,
        "accuracy_reason": accuracy_score.reason,
    }
```

### Dataset loader

Load your evaluation dataset into a [Ragas Dataset](/concepts/datasets) object for experiment execution:

```python
# File: examples/ragas_examples/text2sql/evals.py
import pandas as pd
from pathlib import Path
from typing import Optional
from ragas import Dataset

def load_dataset(limit: Optional[int] = None):
    """Load the text-to-SQL dataset from CSV file."""
    dataset_path = Path(__file__).parent / "datasets" / "booksql_sample.csv"
    
    # Read CSV
    df = pd.read_csv(dataset_path)
    
    # Limit dataset size if requested
    if limit is not None and limit > 0:
        df = df.head(limit)
    
    # Create Ragas Dataset
    dataset = Dataset(name="text2sql_booksql", backend="local/csv", root_dir=".")
    
    for _, row in df.iterrows():
        dataset.append({
            "Query": row["Query"],
            "SQL": row["SQL"], 
            "Levels": row["Levels"],
            "split": row["split"],
        })
    
    return dataset
```

The dataset loader includes a `limit` parameter for development workflows - start with small samples to catch basic errors quickly, then scale to full evaluation.

## Run baseline evaluation

### Execute evaluation pipeline and collect results

```python
import asyncio
from ragas_examples.text2sql.evals import text2sql_experiment, load_dataset

async def run_evaluation():
    """Run text-to-SQL evaluation with direct code approach."""
    # Load dataset
    dataset = load_dataset()
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Run the experiment
    results = await text2sql_experiment.arun(
        dataset, 
        name="gpt-5-mini-prompt-v1",
        model="gpt-5-mini",
        prompt_file=None,
    )
    
    # Report results
    print(f"✅ gpt-5-mini-prompt-v1: {len(results)} cases evaluated")
    
    # Calculate and display accuracy
    accuracy_rate = sum(1 for r in results if r["execution_accuracy"] == "correct") / max(1, len(results))
    print(f"gpt-5-mini-prompt-v1 Execution Accuracy: {accuracy_rate:.2%}")

# Run the evaluation
await run_evaluation()
```

??? "📋 Output (prompt v1)"

    ```text
    Loading dataset...
    Dataset loaded with 99 samples
    Running text-to-SQL evaluation with model: gpt-5-mini
    Using prompt file: prompt.txt
    Running experiment: 100%|██████████████████████| 99/99 [01:06<00:00,  1.49it/s]
    ✅ gpt-5-mini-prompt-v1: 99 cases evaluated
    gpt-5-mini-prompt-v1 Execution Accuracy: 2.02%
    ```

**Configuration options:**

- `model`: OpenAI model to use (default: "gpt-5-mini")
- `prompt_file`: Custom prompt file (default: None for built-in prompt)
- `limit`: Number of samples (default: None for all samples, specify a number to limit)
- `name`: Custom experiment name for result tracking

### Initial performance analysis

The evaluation generates comprehensive CSV results for analysis. Each row contains:

- `query`: Natural language input
- `expected_sql`: Ground truth SQL  
- `predicted_sql`: Generated SQL
- `level`: Difficulty (easy/medium/hard)
- `execution_accuracy`: correct/incorrect (result comparison)
- `accuracy_reason`: Detailed comparison results

## Analyze errors and failure patterns

After running evaluations, you can analyze the failure patterns to understand where your Text2SQL agent is making mistakes. Here are some sample rows from our baseline evaluation results to give you a complete picture of what the evaluation data looks like:

| Query | Expected SQL | Predicted SQL | Level | Execution Accuracy |
|-------|--------------|---------------|-------|-------------------|
| What is the balance due from Richard Aguirre? | `select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Richard Aguirre" )` | `SELECT customer_name, customer_full_name, Balance FROM customers WHERE customer_name = 'Richard Aguirre' OR customer_full_name = 'Richard Aguirre';` | medium | incorrect |
| How much open credit does customer Natalie Myers? | `select sum(open_balance) from ( select distinct transaction_id, open_balance from master_txn_table where customers = "Natalie Myers" )` | `SELECT Balance FROM customers WHERE customer_name = 'Natalie Myers' OR customer_full_name = 'Natalie Myers';` | easy | incorrect |
| What is my average invoice from Ashley Thompson? | `select avg(amount) from (select distinct transaction_id, amount from master_txn_table where customers = "Ashley Thompson" and transaction_type = 'invoice')` | `SELECT AVG(Amount) FROM master_txn_table WHERE Customers = 'Ashley Thompson' AND Transaction_TYPE = 'Invoice';` | hard | incorrect |
| Show number of transactions with Carol Smith | `select count(distinct transaction_id) from master_txn_table where customers = 'Carol Smith' or vendor = 'Carol Smith'` | `SELECT COUNT(*) AS num_transactions FROM master_txn_table WHERE Customers = 'Carol Smith';` | medium | incorrect |

**Key observations from these examples:**

**Execution Accuracy: 0%** - None of the queries return the same results as expected

**Common failure patterns:**

  - Using wrong tables (`customers` vs `master_txn_table`)
  - Missing deduplication (`count(distinct transaction_id)` vs `COUNT(*)`)
  - Incomplete filtering logic (missing `OR vendor = 'Name'` conditions)
  - Wrong column names (`Balance` vs `open_balance`)

This shows that while the agent generates valid SQL, it needs significant improvement in understanding the business logic and database schema relationships.

### Error Analysis

To analyze your failures systematically, manually review and annotate each row in your results CSV, categorizing the types of errors you observe. You can use AI to help you categorize with this prompt:

??? "📋 Error Analysis Categorization Prompt"

    ```text
    You are analyzing why a Text2SQL prediction failed. Given the following information, identify the error codes and provide a brief analysis.

    Available error codes:
    - AGGR_DISTINCT_MISSING: Used COUNT/SUM without DISTINCT or deduplication
    - WRONG_FILTER_COLUMN: Filtered on the wrong column 
    - WRONG_SOURCE_TABLE_OR_COLUMN: Selected metric from the wrong table/column
    - EXTRA_TRANSFORMATION_OR_CONDITION: Added ABS(), extra filters that change results
    - OUTPUT_COLUMN_ALIAS_MISMATCH: Output column names don't match
    - NULL_OR_EMPTY_RESULT: Result is None/empty due to wrong filters or source
    - GENERIC_VALUE_MISMATCH: Aggregation computed but numeric value differs for unclear reasons
    - OTHER: Fallback

    Query: [YOUR_QUERY]
    Expected SQL: [EXPECTED_SQL]
    Predicted SQL: [PREDICTED_SQL]
    Execution Accuracy: [ACCURACY_RESULT]
    Accuracy Reason: [ACCURACY_REASON]

    Respond with:
    - error_codes: array of applicable error codes (1 or more)
    - error_analysis: brief 1-3 sentence explanation of what went wrong
    ```

    Copy this prompt and use it with your preferred LLM to analyze individual failures from your results CSV.

For convenience, we made it a script that you can use to automatically categorize errors using GPT 5: `uv run python -m ragas_examples.text2sql.analyze_errors --input experiments/your_results.csv`

### Review Process

1. **Examine high-frequency error codes** from the summary
2. **Manually inspect 5-10 examples** of each major error type
3. **Verify the AI's categorization** matches your understanding
4. **Check if the "expected" SQL is actually correct** for your schema
5. **Look for patterns** the AI might have missed
6. **Prioritize fixes** based on impact and frequency

Only after manual validation should you use these insights to improve your prompts, few-shot examples, or agent architecture.

### Sample error analysis results

Here are some sample rows from the annotated results showing the types of errors that remain after the first round of improvements:

| Query | Expected SQL | Predicted SQL | Error Analysis | Error Codes |
|-------|--------------|---------------|----------------|-------------|
| How much we received from Fuel? | `select sum(credit) from master_txn_table as T1 join chart_of_accounts as T2 on T1.account = T2.account_name where account_type in ('Income', 'Other Income') and instr(account,"Fuel")` | `SELECT SUM(Amount) FROM (SELECT DISTINCT Transaction_ID, Amount FROM master_txn_table WHERE Customers = 'Fuel');` | The prediction summed Amount from master_txn_table instead of credit and omitted the join to chart_of_accounts, using Customers='Fuel' rather than filtering account names containing 'Fuel' and restricting to Income/Other Income. | `["WRONG_SOURCE_TABLE_OR_COLUMN", "WRONG_FILTER_COLUMN", "NULL_OR_EMPTY_RESULT"]` |
| What was the most recent invoice for Roberta Shaw? | `select transaction_id from master_txn_table where transaction_type = 'invoice' and customers = "Roberta Shaw" order by transaction_date desc limit 1` | `SELECT Transaction_ID, Transaction_DATE, Amount, Open_balance, CreatedDATE, CreatedUSER, Account, Customers, Product_Service FROM master_txn_table WHERE Transaction_TYPE = 'invoice' AND Customers = 'Roberta Shaw' ORDER BY Transaction_DATE DESC LIMIT 1;` | The prediction selected nine columns instead of only Transaction_ID as expected, causing a column/schema mismatch despite returning the same row. | `["OUTPUT_COLUMN_ALIAS_MISMATCH"]` |
| What is my average invoice from Jeremy Strong? | `select avg(amount) from (select distinct transaction_id, amount from master_txn_table where customers = "Jeremy Strong" and transaction_type = 'invoice')` | `SELECT AVG(Amount) FROM (SELECT DISTINCT Transaction_ID, Amount FROM master_txn_table WHERE Transaction_TYPE = 'invoice' AND Vendor = 'Jeremy Strong') AS t;` | The query filters on Vendor = 'Jeremy Strong' instead of the correct customers column, so it likely matched no rows. This leads to AVG(amount) returning NULL. | `["WRONG_FILTER_COLUMN", "NULL_OR_EMPTY_RESULT"]` |

**Key observations from results:**

- **Error patterns:**
  - **Missing OR conditions**: Queries about transactions "with" someone should check both `customers` and `vendor` columns
  - **Wrong column selection**: Using `Amount` instead of `credit` for financial queries
  - **Output schema mismatches**: Selecting too many columns or wrong column names
  - **Missing joins**: Not joining with `chart_of_accounts` for account-type filtering

These patterns inform the next iteration of prompt improvements, focusing on complete filtering logic and proper financial query handling.

Decide what to change in the prompt using generic rules, not per-row fixes. Avoid adding case-specific examples; prefer schema-grounded guardrails so that you are not overfitting to the data.

Repeat this loop iteratively:

- Run → Annotate → Review → Decide generic guardrails → Update `prompt_vX.txt` → Re-run → Compare → Repeat.
- Keep guardrails concise and schema-grounded so improvements generalize without overfitting.
 - Version your prompts (`prompt_v2.txt`, `prompt_v3.txt`, `prompt_v4.txt`) and maintain a brief changelog per version.
 - Stop when execution accuracy plateaus across two consecutive iterations or meets your business threshold.

## Improve your system  

### Create and use a new prompt version

We keep the baseline prompt intact and create a new version for iteration.

Create `prompt_v2.txt` to include concise, reusable guardrails. Keep them generic enough to apply broadly while grounded in the provided schema. Example of a section we added to `prompt_v1.txt` to create `prompt_v2.txt`:

```text
- Use exact table and column names from the schema; do not invent fields
- Prefer transactional facts from `master_txn_table`; use entity tables for static attributes
- Map parties correctly in filters:
  - Customer-focused → filter on `Customers`
  - Vendor-focused → filter on `Vendor`
- Disambiguate events via `Transaction_TYPE` (e.g., invoices → `Transaction_TYPE = 'invoice'`)
- Avoid double-counting by deduplicating on `Transaction_ID` for counts and aggregates:
  - Counts: `count(distinct Transaction_ID)`
  - Aggregates: compute over a deduplicated subquery on `(Transaction_ID, metric_column)`
- For open credit/balance due per customer, aggregate `Open_balance` from `master_txn_table` filtered by `Customers` with deduplication
- Do not add extra transforms or filters (e.g., `abs()`, `< 0`) unless explicitly asked
- Keep a single `SELECT`; avoid aliases for final column names
```

We save this improved prompt as `prompt_v2.txt`.

### Re-run evaluation with the new prompt

```python
import asyncio
from ragas_examples.text2sql.evals import text2sql_experiment, load_dataset

async def run_v2_evaluation():
    """Run evaluation with prompt v2."""
    # Load dataset
    dataset = load_dataset()
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Run experiment
    results = await text2sql_experiment.arun(
        dataset, 
        name="gpt-5-mini-prompt-v2",
        model="gpt-5-mini",
        prompt_file="prompt_v2.txt",
    )
    
    # Report results
    print(f"✅ gpt-5-mini-prompt-v2: {len(results)} cases evaluated")
    
    # Calculate accuracy
    accuracy_rate = sum(1 for r in results if r["execution_accuracy"] == "correct") / max(1, len(results))
    print(f"gpt-5-mini-prompt-v2 Execution Accuracy: {accuracy_rate:.2%}")

await run_v2_evaluation()
```

??? "📋 Output (prompt v2)"

    ```text
    Loading dataset...
    Dataset loaded with 99 samples
    Running text-to-SQL evaluation with model: gpt-5-mini
    Using prompt file: prompt_v2.txt
    Running experiment: 100%|██████████████████████| 99/99 [01:00<00:00,  1.63it/s]
    ✅ gpt-5-mini-prompt-v2: 99 cases evaluated
    gpt-5-mini-prompt-v2 Execution Accuracy: 60.61%
    ```

We see an improvement from 2.02% to 60.61% in execution accuracy with `prompt_v2`.

Review the new results CSV in `experiments/` and continue the loop again.

### Continue iterating: Create prompt v3

Even with the major improvements in `prompt_v2.txt`, the 60% accuracy still leaves room for growth. A deeper analysis of the failures reveals several recurring patterns:

1.  **Misunderstanding of Financial Concepts**: The model consistently defaults to aggregating the `Amount` column instead of the correct `Credit` (for income) or `Debit` (for expenses) columns. It also often fails to `JOIN` with `chart_of_accounts` to filter by account type (e.g., 'Income').
2.  **Adding Unnecessary Transformations**: The model frequently complicates queries with unrequested `DISTINCT` clauses or extra filters (like `Transaction_TYPE = 'invoice'`), which alter the results.
3.  **Incorrect Column Selection**: For "show all transactions" queries, it often uses `SELECT *` instead of the expected `SELECT DISTINCT Transaction_ID`, leading to schema mismatches. It also generates the wrong column names for aggregations (e.g. `max(transaction_date)` instead of `transaction_date`).
4.  **Incomplete Filtering**: It often misses `OR` conditions (e.g., checking both `Customers` and `Vendor` for a transaction "with" someone) or filters on the wrong column entirely.

Based on this deeper analysis, create `prompt_v3.txt` with even more specific, schema-grounded guidelines to address these recurring issues:

Key additions to `prompt_v3.txt`:

```text
### CORE QUERY GENERATION GUIDELINES

1.  **Use Correct Schema**: Use exact table and column names...
2.  **Simplicity First**: Keep the query as simple as possible...
...

### ADVANCED QUERY PATTERNS

5.  **Financial Queries (Revenue, Sales, Expenses)**:
    -   **Metric Selection**:
        -   For revenue, income, sales, or money **received**: aggregate the `Credit` column.
        -   For expenses, bills, or money **spent**: aggregate the `Debit` column.
        -   Use the `Amount` column only when...
    -   **Categorical Financial Queries**: For questions involving financial categories... you **MUST** `JOIN` `master_txn_table` with `chart_of_accounts`...

6.  **Filtering Logic**:
    -   **Ambiguous Parties**: For questions about transactions "with" or "involving" a person or company, you **MUST** check both `Customers` and `Vendor` columns. E.g., `WHERE Customers = 'Name' OR Vendor = 'Name'`.
    -   **Avoid Extra Filters**: Do not add implicit filters...

7.  **Column Selection and Naming**:
    -   **Avoid `SELECT *`**: When asked to "show all transactions", return only `DISTINCT Transaction_ID`...
    -   **"Most Recent" / "Last" Queries**: To get the 'most recent' or 'last' record, use `ORDER BY Transaction_DATE DESC LIMIT 1`. This preserves the original column names... Avoid using `MAX()`...

```

These new rules are designed to be generic but directly target the observed failure patterns.

**Re-run evaluation with `prompt_v3.txt`:**

```python
import asyncio
from ragas_examples.text2sql.evals import text2sql_experiment, load_dataset

async def run_v3_evaluation():
    """Run evaluation with prompt v3."""
    # Load dataset
    dataset = load_dataset()
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Run experiment
    results = await text2sql_experiment.arun(
        dataset, 
        name="gpt-5-mini-prompt-v3",
        model="gpt-5-mini",
        prompt_file="prompt_v3.txt",
    )
    
    # Report results
    print(f"✅ gpt-5-mini-prompt-v3: {len(results)} cases evaluated")
    
    # Calculate accuracy
    accuracy_rate = sum(1 for r in results if r["execution_accuracy"] == "correct") / max(1, len(results))
    print(f"gpt-5-mini-prompt-v3 Execution Accuracy: {accuracy_rate:.2%}")

await run_v3_evaluation()
```

We see an improvement from 60.61% to 70.71% in execution accuracy with `prompt_v3`.

### Key principles for continued iteration

The 70% accuracy achieved with `prompt_v3.txt` demonstrates the power of systematic iteration. You can continue this process to push accuracy even higher.

**Key principles for continued iteration:**

- Each iteration should target **3-5 high-frequency error patterns** from the latest results
- Keep new rules **generic and schema-grounded** to avoid overfitting
- **Stop when accuracy plateaus** across 2-3 consecutive iterations
- If you hit a plateau with prompt improvements, you can try experimenting with better models or return any sql error back to the LLM to fix it making an actual agentic flow. 

## Compare results

After running all prompt versions, we can compare the final results.

| Prompt | Execution Accuracy | Results CSV |
|---|---|---|
| v1 (`prompt.txt`) | 2.02% | `experiments/...-prompt-v1.csv` |
| v2 (`prompt_v2.txt`) | 60.61% | `experiments/...-prompt-v2.csv` |
| v3 (`prompt_v3.txt`) | 70.71% | `experiments/...-prompt-v3.csv` |

**Progress Analysis:**
- **v1 → v2**: Massive 58 percentage point jump from 2.02% to 60.61% through basic deduplication and business logic guidelines
- **v2 → v3**: Additional 10 percentage point improvement from 60.61% to 70.71% through enhanced financial query guidelines, better filtering logic, and column selection rules
- The improvements target specific failure patterns identified through error analysis: financial concepts, unnecessary transformations, and incomplete filtering

## Conclusion

This guide showed you how to build a systematic evaluation process for text-to-SQL systems. 

**Key takeaways:**

- Set up execution accuracy metrics to compare actual query results
- Follow the iterative process: evaluate → analyze errors → improve → repeat  

The evaluation framework gives you a reliable way to measure and improve your system, with Ragas handling the orchestration and result aggregation automatically.