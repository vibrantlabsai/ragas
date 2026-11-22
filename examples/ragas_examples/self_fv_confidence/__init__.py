"""self-FV confidence — white-box uncertainty for open-weight LLM judges.

Port of self-function vectors (Chung et al. 2026, arXiv:2606.19353) as a
ragas-consumable confidence signal for discrete ICL verdicts. ``SelfFVConfidence``
is a genuine ragas ``SingleTurnMetric`` (drop it into ``ragas.evaluate(...)``).
"""
from .metric import (
    RAGAS_AVAILABLE,
    SelfFVConfidence,
    default_judge_prompt,
    uncertainty_to_confidence,
)
from .self_fv import (
    load_lm,
    attention_layer_indices,
    collect_mean_head_act,
    compute_cie,
    decompose_uncertainty,
)

__all__ = [
    "SelfFVConfidence",
    "uncertainty_to_confidence",
    "default_judge_prompt",
    "RAGAS_AVAILABLE",
    "load_lm",
    "attention_layer_indices",
    "collect_mean_head_act",
    "compute_cie",
    "decompose_uncertainty",
]
