---
search:
  exclude: true
---

# Adding to your CI pipeline with Pytest

You can add Ragas evaluations as part of your Continious Integration pipeline
to keep track of the qualitative performance of your RAG pipeline. Consider these as
part of your end-to-end test suite which you run before major changes and releases.

The usage is straight forward, but the main thing is to set the `in_ci` argument for the
`evaluate()` function to `True`. This runs Ragas metrics in a special mode that ensures
it produces more reproducible metrics but will be costlier.

You can easily write a Pytest test as follows

!!! note
    This dataset that is already populated with outputs from a reference RAG
    When testing your own system make sure you use outputs from RAG pipeline
    you want to test. For more information on how to build your datasets check
    [Building HF `Dataset` with your own Data](./data_preparation.md) docs.

```python
import pytest
from datasets import load_dataset

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

def assert_in_range(score: float, value: float, plus_or_minus: float):
    """
    Check if computed score is within the range of value +/- max_range
    """
    assert value - plus_or_minus <= score <= value + plus_or_minus


def test_amnesty_e2e():
    # loading the V2 dataset
    amnesty_qa = load_dataset("vibrantlabsai/amnesty_qa", "english_v2")["eval"]


    result = evaluate(
        amnesty_qa,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
        in_ci=True,
    )
    assert result["answer_relevancy"] >= 0.9
    assert result["context_recall"] >= 0.95
    assert result["context_precision"] >= 0.95
    assert_in_range(result["faithfulness"], value=0.4, plus_or_minus=0.1)
```

## Using Pytest Markers for Ragas E2E tests

Because these are long end-to-end test one thing that you can leverage is [Pytest Markers](https://docs.pytest.org/en/latest/example/markers.html) which help you mark your tests with special tags. It is recommended to mark Ragas tests with special tags, so you can run them only when needed.

To add a new `ragas_ci` tag to Pytest, add the following to your `conftest.py`
```python
def pytest_configure(config):
    """
    configure pytest
    """
    # add `ragas_ci`
    config.addinivalue_line(
        "markers", "ragas_ci: Set of tests that will be run as part of Ragas CI"
    )
```

now you can use `ragas_ci` to mark all the tests that are part of Ragas CI.

```python
import pytest
from datasets import load_dataset

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

def assert_in_range(score: float, value: float, plus_or_minus: float):
    """
    Check if computed score is within the range of value +/- max_range
    """
    assert value - plus_or_minus <= score <= value + plus_or_minus


@pytest.mark.ragas_ci
def test_amnesty_e2e():
    # loading the V2 dataset
    amnesty_qa = load_dataset("vibrantlabsai/amnesty_qa", "english_v2")["eval"]


    result = evaluate(
        amnesty_qa,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
        in_ci=True,
    )
    assert result["answer_relevancy"] >= 0.9
    assert result["context_recall"] >= 0.95
    assert result["context_precision"] >= 0.95
    assert_in_range(result["faithfulness"], value=0.4, plus_or_minus=0.1)
```
