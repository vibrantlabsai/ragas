import copy
import pickle

import pytest

from ragas.metrics import MetricResult


def _make_result(value):
    return MetricResult(
        value=value,
        reason="because",
        traces={"input": {"q": "a"}, "output": {"a": "b"}},
    )


@pytest.mark.parametrize("value", [0.5, "pass", [3, 1, 2]])
def test_metric_result_copy(value):
    result = _make_result(value)

    copied = copy.copy(result)

    assert copied.value == value
    assert copied.reason == "because"
    assert copied.traces == result.traces


@pytest.mark.parametrize("value", [0.5, "pass", [3, 1, 2]])
def test_metric_result_deepcopy(value):
    result = _make_result(value)

    copied = copy.deepcopy(result)

    assert copied.value == value
    assert copied.reason == "because"
    assert copied.traces == result.traces
    # a deep copy must not share the nested data with the original
    assert copied.traces is not result.traces


@pytest.mark.parametrize("value", [0.5, "pass", [3, 1, 2]])
def test_metric_result_pickle_round_trip(value):
    result = _make_result(value)

    restored = pickle.loads(pickle.dumps(result))

    assert restored.value == value
    assert restored.reason == "because"
    assert restored.traces == result.traces


def test_metric_result_missing_value_is_attribute_error():
    # an object built without __init__ has no value yet; looking up attributes
    # on it must raise AttributeError instead of recursing
    empty = MetricResult.__new__(MetricResult)

    with pytest.raises(AttributeError):
        empty.upper
    assert not hasattr(empty, "value_that_does_not_exist")


def test_metric_result_still_forwards_attributes_to_value():
    assert MetricResult(value="pass").upper() == "PASS"
    assert MetricResult(value=[1, 2]).count(2) == 1
