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


def meets_threshold(score: float, threshold: float) -> bool:
    """Whether a similarity/relevance score clears ``threshold``.

    Single definition of the boundary rule, deliberately inclusive: a score
    exactly equal to the threshold counts as passing.

    This exists because the boundary is easy to re-implement inconsistently.
    ``NonLLMContextRecall`` used ``>`` while ``NonLLMContextPrecisionWithReference``
    used ``>=``, so with their shared default threshold (0.5) and shared
    ``NonLLMStringSimilarity`` distance measure, a context sitting exactly on
    the boundary was relevant for one metric and not the other (issue #2777).
    Routing both through this helper makes that divergence a change to one
    function rather than something a new metric can reintroduce by accident.
    """
    return score >= threshold
