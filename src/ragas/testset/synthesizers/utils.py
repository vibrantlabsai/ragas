import math
import typing as t


def calculate_split_values(
    probs: t.List[float], n: int
) -> t.Tuple[t.List[int], t.List[int]]:
    # calculate the number of samples for each scenario, rounding down and then
    # handing the leftover samples to the largest remainders so the splits add up
    # to exactly n
    exact_splits = [n * prob for prob in probs]
    splits = [math.floor(split) for split in exact_splits]
    leftover = max(n - sum(splits), 0)
    by_remainder = sorted(
        range(len(probs)), key=lambda i: exact_splits[i] - splits[i], reverse=True
    )
    for i in by_remainder[:leftover]:
        splits[i] += 1
    # convert this to split values like [0, 30, 60, 80]
    split_values = [0] + splits + [sum(splits)]
    split_values = [sum(split_values[:i]) for i in range(1, len(split_values))]
    return (splits, split_values)
