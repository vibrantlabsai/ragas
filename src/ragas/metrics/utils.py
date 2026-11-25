from typing import Any


def make_hashable_tool_arg(value: Any) -> Any:
    """Convert nested tool arguments to keys for exact matching."""
    if isinstance(value, dict):
        return frozenset(
            (key, make_hashable_tool_arg(item)) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return tuple(make_hashable_tool_arg(item) for item in value)
    if isinstance(value, set):
        return frozenset(make_hashable_tool_arg(item) for item in value)
    if isinstance(value, bool):
        # Python compares booleans equal to 0 and 1, but JSON distinguishes them.
        return (bool, value)
    return value


def fbeta_score(tp, fp, fn, beta=1.0):
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision == 0 and recall == 0:
        return 0.0

    beta_squared = beta**2
    fbeta = (
        (1 + beta_squared)
        * (precision * recall)
        / ((beta_squared * precision) + recall)
    )

    return fbeta
