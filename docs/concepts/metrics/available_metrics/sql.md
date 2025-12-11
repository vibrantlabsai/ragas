# SQL


## Execution based metrics
In these metrics the resulting SQL is compared after executing the SQL query on the database and then comparing the `response` with the expected results.

### DataCompy Score

`DataCompyScore` metric uses DataCompy, a python library that compares two pandas DataFrames. It provides a simple interface to compare two DataFrames and provides a detailed report of the differences. In this metric the `response` is executed on the database and the resulting data is compared with the expected data, i.e. `reference`. To enable comparison both `response` and `reference` should be in the form of a Comma-Separated Values as shown in the example.

DataFrames can be compared across rows or columns. This can be configured using `mode` parameter.

If mode is `row` then the comparison is done row-wise. If mode is `column` then the comparison is done column-wise.

$$
\text{Precision } = {|\text{Number of matching rows in response and reference}| \over |\text{Total number of rows in response}|}
$$

$$
\text{Recall } = {|\text{Number of matching rows in response and reference}| \over |\text{Total number of rows in reference}|}
$$

By default, the mode is set to `row`, and metric is F1 score which is the harmonic mean of precision and recall.

```python
from ragas.metrics.collections import DataCompyScore

data1 = """acct_id,dollar_amt,name,float_fld,date_fld
10000001234,123.45,George Maharis,14530.1555,2017-01-01
10000001235,0.45,Michael Bluth,1,2017-01-01
10000001236,1345,George Bluth,,2017-01-01
10000001237,123456,Bob Loblaw,345.12,2017-01-01
10000001238,1.05,Lucille Bluth,,2017-01-01
10000001238,1.05,Loose Seal Bluth,,2017-01-01
"""

data2 = """acct_id,dollar_amt,name,float_fld
10000001234,123.4,George Michael Bluth,14530.155
10000001235,0.45,Michael Bluth,
10000001236,1345,George Bluth,1
10000001237,123456,Robert Loblaw,345.12
10000001238,1.05,Loose Seal Bluth,111
"""

metric = DataCompyScore()
result = await metric.ascore(response=data1, reference=data2)
print(f"F1 Score: {result.value}")
print(f"Details: {result.reason}")
```

To change the mode to column-wise comparison, set the `mode` parameter to `column`.

```python
metric = DataCompyScore(mode="columns", metric="recall")
result = await metric.ascore(response=data1, reference=data2)
```

---

### DataCompyScore (Legacy)

!!! warning "Deprecated"
    `DataCompyScore` from `ragas.metrics` is deprecated and will be removed in a future version. Please use `DataCompyScore` from `ragas.metrics.collections` as shown above.

The legacy `DataCompyScore` uses the `SingleTurnSample` schema:

```python
from ragas.metrics import DataCompyScore
from ragas.dataset_schema import SingleTurnSample

data1 = """acct_id,dollar_amt,name,float_fld,date_fld
10000001234,123.45,George Maharis,14530.1555,2017-01-01
10000001235,0.45,Michael Bluth,1,2017-01-01
10000001236,1345,George Bluth,,2017-01-01
10000001237,123456,Bob Loblaw,345.12,2017-01-01
10000001238,1.05,Lucille Bluth,,2017-01-01
10000001238,1.05,Loose Seal Bluth,,2017-01-01
"""

data2 = """acct_id,dollar_amt,name,float_fld
10000001234,123.4,George Michael Bluth,14530.155
10000001235,0.45,Michael Bluth,
10000001236,1345,George Bluth,1
10000001237,123456,Robert Loblaw,345.12
10000001238,1.05,Loose Seal Bluth,111
"""
sample = SingleTurnSample(response=data1, reference=data2)
scorer = DataCompyScore()
await scorer.single_turn_ascore(sample)
```
To change the mode to column-wise comparison, set the `mode` parameter to `column`.


```python
scorer = DataCompyScore(mode="column", metric="recall")
```

## Non Execution based metrics

Executing SQL queries on the database can be time-consuming and sometimes not feasible. In such cases, we can use non-execution based metrics to evaluate the SQL queries. These metrics compare the SQL queries directly without executing them on the database.

### SQL Semantic Equivalence

`SQLSemanticEquivalence` is a metric that evaluates whether a generated SQL query is semantically equivalent to a reference query. The metric uses an LLM to analyze both queries in the context of the provided database schema and determine if they would produce the same results.

This is a binary metric:
- **1.0**: The SQL queries are semantically equivalent
- **0.0**: The SQL queries are not equivalent

The metric considers the database schema context to make accurate equivalence judgments, accounting for syntactic differences that don't affect semantics (e.g., `active = 1` vs `active = true`).

```python
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import SQLSemanticEquivalence

# Initialize the LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create the metric
metric = SQLSemanticEquivalence(llm=llm)

# Evaluate SQL equivalence
result = await metric.ascore(
    response="""
        SELECT p.product_name, SUM(oi.quantity) AS total_quantity
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        GROUP BY p.product_name;
    """,
    reference="""
        SELECT products.product_name, SUM(order_items.quantity) AS total_quantity
        FROM order_items
        INNER JOIN products ON order_items.product_id = products.product_id
        GROUP BY products.product_name;
    """,
    reference_contexts=[
        """
        Table order_items:
        - order_item_id: INT
        - order_id: INT
        - product_id: INT
        - quantity: INT
        """,
        """
        Table products:
        - product_id: INT
        - product_name: VARCHAR
        - price: DECIMAL
        """
    ]
)

print(f"Equivalent: {result.value == 1.0}")
print(f"Explanation: {result.reason}")
```

The result includes explanations of both queries and the reasoning for the equivalence determination.

---

### LLMSQLEquivalence (Legacy)

!!! warning "Deprecated"
    `LLMSQLEquivalence` is deprecated and will be removed in a future version. Please use `SQLSemanticEquivalence` from `ragas.metrics.collections` as shown above.

`LLMSQLEquivalence` is the legacy metric for SQL semantic equivalence evaluation. It uses the `SingleTurnSample` schema and requires setting the LLM separately.

```python
from ragas.metrics import LLMSQLEquivalence
from ragas.dataset_schema import SingleTurnSample

sample = SingleTurnSample(
    response="""
        SELECT p.product_name, SUM(oi.quantity) AS total_quantity
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        GROUP BY p.product_name;
    """,
    reference="""
        SELECT p.product_name, COUNT(oi.quantity) AS total_quantity
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        GROUP BY p.product_name;
    """,
    reference_contexts=[
        """
        Table order_items:
        - order_item_id: INT
        - order_id: INT
        - product_id: INT
        - quantity: INT
        """,
        """
        Table products:
        - product_id: INT
        - product_name: VARCHAR
        - price: DECIMAL
        """
    ]
)

scorer = LLMSQLEquivalence()
scorer.llm = openai_model
await scorer.single_turn_ascore(sample)
```
