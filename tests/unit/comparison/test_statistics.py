import random
from statistics import mean

import pytest

from ragas.comparison.statistics import bootstrap_interval, quantile, stable_seed


def test_bootstrap_accepts_differences_between_valid_extreme_scores():
    from ragas.comparison import ComparisonPlan, MetricSpec, compare

    before = [{"case_id": str(i), "score": -1e150} for i in range(20)]
    after = [{"case_id": str(i), "score": 1e150 - i * 1e147} for i in range(20)]
    report = compare(before, after, plan=ComparisonPlan(metrics=(MetricSpec("score"),)))
    assert report.verdict == "pass"


def test_quantile_matches_linear_order_statistic_interpolation():
    assert quantile((0.0, 2.0, 4.0, 8.0), 0) == 0
    assert quantile((0.0, 2.0, 4.0, 8.0), 1) == 8
    assert quantile((0.0, 2.0, 4.0, 8.0), 0.25) == 1.5
    assert quantile((0.0, 2.0, 4.0, 8.0), 0.5) == 3


def test_bootstrap_matches_independent_reference_and_preserves_global_rng():
    differences = (-0.4, -0.1, 0.2, 0.5)
    seed = stable_seed(17, "quality", "overall")
    rng = random.Random(seed)
    reference = sorted(
        mean(differences[rng.randrange(4)] for _ in range(4)) for _ in range(1000)
    )
    state = random.getstate()
    interval = bootstrap_interval(
        differences, resamples=1000, confidence=0.9, seed=seed
    )
    assert interval.lower == pytest.approx(quantile(tuple(reference), 0.05))
    assert interval.upper == pytest.approx(quantile(tuple(reference), 0.95))
    assert random.getstate() == state


def test_seed_is_namespaced_and_stable():
    assert stable_seed(17, "a", "bc") != stable_seed(17, "ab", "c")
    assert stable_seed(17, "a", "bc") == stable_seed(17, "a", "bc")


@pytest.mark.parametrize("values", [(), (float("nan"), 1), (float("inf"), 1)])
def test_invalid_statistics_cannot_produce_a_passing_interval(values):
    with pytest.raises(ValueError):
        bootstrap_interval(values, resamples=1000, confidence=0.9, seed=0)
