"""Deterministic paired case bootstrap with explicit finite arithmetic.

The independent resampling unit is a case, after averaging its paired repeats.
Percentile intervals are approximate; no distribution-free coverage guarantee
is implied. This module neither samples individual repeats nor uses global RNG.
"""

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Sequence

from .config import integer, number
from .observations import canonical_bytes


@dataclass(frozen=True)
class Interval:
    """Two-sided percentile bounds in the caller's oriented metric units."""

    lower: float
    upper: float
    standard_error: float


def finite_mean(values: Sequence[float]) -> float:
    """Compute a stable mean without allowing overflow or empty evidence."""
    if not values:
        raise ValueError("cannot average an empty comparison cohort")
    try:
        result = math.fsum(value / len(values) for value in values)
    except (OverflowError, ValueError) as exc:
        raise ValueError("comparison mean exceeds finite arithmetic") from exc
    if not math.isfinite(result):
        raise ValueError("comparison mean must be finite")
    return result


def quantile(ordered: Sequence[float], probability: float) -> float:
    """Linear interpolation between adjacent ordered values (Type 7)."""
    if not ordered or not 0 <= probability <= 1:
        raise ValueError("quantile requires values and a probability in [0, 1]")
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def stable_seed(seed: int, metric: str, group: str) -> int:
    """Namespace a local RNG independently of Python's randomized hash seed."""
    payload = canonical_bytes(["ragas-paired-bootstrap-v1", seed, metric, group])
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def bootstrap_interval(
    differences: Sequence[float],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> Interval:
    """Resample complete cases, preserving each case's paired difference.

    Work ceilings and statistical eligibility are checked by the comparison
    engine before calling this primitive. Only one sample mean per resample is
    retained, rather than allocating a cases-by-resamples matrix.
    """
    integer(resamples, "resamples", 2, 1000000)
    confidence = number(confidence, "confidence")
    if not 0 < confidence < 1 or not differences:
        raise ValueError("bootstrap requires differences and valid confidence")
    # A difference between two valid scores can be twice the score magnitude.
    # Validate this derived quantity separately from user-entered score values.
    values = []
    for value in differences:
        if type(value) not in (int, float):
            raise ValueError("bootstrap differences must be ordinary finite numbers")
        try:
            converted = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError("bootstrap difference exceeds finite arithmetic") from exc
        if not math.isfinite(converted) or abs(converted) > 2e150:
            raise ValueError("bootstrap difference exceeds its derived score range")
        values.append(converted)
    rng = random.Random(seed)
    count = len(values)
    distribution = []
    for _ in range(resamples):
        distribution.append(
            math.fsum(values[rng.randrange(count)] / count for _ in range(count))
        )
    distribution.sort()
    mean = finite_mean(distribution)
    variance = math.fsum(
        ((value - mean) ** 2) / (resamples - 1) for value in distribution
    )
    error = math.sqrt(variance)
    alpha = 1 - confidence
    lower = quantile(distribution, alpha / 2)
    upper = quantile(distribution, 1 - alpha / 2)
    if not all(math.isfinite(value) for value in (lower, upper, error)):
        raise ValueError("bootstrap produced nonfinite arithmetic")
    return Interval(lower, upper, error)
