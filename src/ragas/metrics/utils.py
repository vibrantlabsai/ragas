def fbeta_score(tp_precision, fp, tp_recall, fn, beta=1.0):
    """
    Compute the F-beta score using separate true-positive counts for precision
    and recall directions.

    Parameters
    ----------
    tp_precision : int/float
        Number of response claims supported by the reference
        (true positives for precision).
    fp : int/float
        Number of response claims NOT supported by the reference
        (false positives).
    tp_recall : int/float
        Number of reference claims supported by the response
        (true positives for recall).
    fn : int/float
        Number of reference claims NOT supported by the response
        (false negatives).
    beta : float
        Weight of recall in the combined score. beta > 1 favours recall,
        beta < 1 favours precision.
    """
    if tp_precision + fp == 0:
        precision = 0
    else:
        precision = tp_precision / (tp_precision + fp)

    if tp_recall + fn == 0:
        recall = 0
    else:
        recall = tp_recall / (tp_recall + fn)

    if precision == 0 and recall == 0:
        return 0.0

    beta_squared = beta**2
    fbeta = (
        (1 + beta_squared)
        * (precision * recall)
        / ((beta_squared * precision) + recall)
    )

    return fbeta
