"""ragas white-box confidence metric backed by self-function vectors.

This wraps the self-FV uncertainty decomposition (see ``self_fv.py``) as a
GENUINE ragas ``SingleTurnMetric`` — the same interface ``ragas.evaluate(...)``
drives for built-in metrics like ``Faithfulness``. The metric consumes a
``SingleTurnSample`` (``user_input`` / ``retrieved_contexts`` / ``response``),
builds a discrete-choice judge prompt, reads the LOCAL judge's internal
attention-head activations, and returns a confidence in ``[0, 1]`` through the
ragas-required ``_single_turn_ascore`` coroutine.

Unlike ragas' LLM-as-judge metrics (black-box API via ``llm=``), this needs the
judge's residual stream + per-head ``o_proj`` activations, so it is constructed
with a preloaded Hugging Face ``model`` + ``tokenizer`` (NOT a ragas
``BaseRagasLLM``). It therefore subclasses ``SingleTurnMetric`` directly and does
NOT mix in ``MetricWithLLM``.

Integration reference (upstream_ragas):
  * base class  : ``ragas.metrics.base.SingleTurnMetric``       (base.py:392)
  * overridden  : ``SingleTurnMetric._single_turn_ascore(...)`` (base.py:504-513)
  * overridden  : ``Metric.init(run_config)``                   (base.py:141-151)
  * sample type : ``ragas.dataset_schema.SingleTurnSample``     (dataset_schema.py:54)
"""
from __future__ import annotations
import math
import typing as t

from .self_fv import decompose_uncertainty

# ---------------------------------------------------------------------------
# ragas base classes. Imported for real when ragas is installed; the fallback
# stand-ins only keep this module importable (and the white-box ``score_prompt``
# path usable) in an environment without ragas — the PRIMARY, genuine path is
# the real ``SingleTurnMetric`` subclassing below.
# ---------------------------------------------------------------------------
try:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics.base import (
        MetricOutputType,
        MetricType,
        SingleTurnMetric,
    )

    RAGAS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without ragas installed
    RAGAS_AVAILABLE = False
    import enum

    class MetricType(enum.Enum):  # mirrors ragas.metrics.base.MetricType
        SINGLE_TURN = "single_turn"
        MULTI_TURN = "multi_turn"

    class MetricOutputType(enum.Enum):  # mirrors ragas.metrics.base.MetricOutputType
        BINARY = "binary"
        DISCRETE = "discrete"
        CONTINUOUS = "continuous"
        RANKING = "ranking"

    class SingleTurnMetric:  # minimal stand-in; real base used when ragas present
        """Placeholder used only when ragas is not importable."""

    if t.TYPE_CHECKING:  # keep type hints resolvable for static tooling
        from ragas.dataset_schema import SingleTurnSample


# Default faithfulness-style discrete judge template (2 options: A=faithful,
# B=not faithful). Users with a different label space should pass their own
# ``prompt_builder`` matching their ``option_token_ids``.
DEFAULT_JUDGE_HEADER = (
    "Judge whether the Answer is faithful to (fully supported by) the "
    "Context. Options: A=faithful, B=not faithful.\n\n"
)


def default_judge_prompt(row: t.Mapping[str, t.Any],
                         header: str = DEFAULT_JUDGE_HEADER) -> str:
    """Build a discrete-choice judge prompt from a ``SingleTurnSample`` dict.

    The prompt ends right before the answer token (``Faithful:``) so the next
    token distribution over the option ids is the verdict distribution. Mirrors
    the zero-shot form of ``example.py``'s ``fmt``.
    """
    ctx = row.get("retrieved_contexts") or []
    if isinstance(ctx, (list, tuple)):
        ctx = " ".join(str(c) for c in ctx)
    ctx = " ".join(str(ctx).split())[:1500]
    question = row.get("user_input") or ""
    answer = row.get("response") or ""
    return header + f"Context: {ctx}\nQuestion: {question}\nAnswer: {answer}\nFaithful:"


def uncertainty_to_confidence(total: float, aleatoric: float, epistemic: float,
                              n_labels: int, mode: str = "total") -> float:
    """Map the (Total, AU, EU) decomposition to a confidence in [0, 1].

    Entropies are in nats; the maximum for a uniform label distribution is
    ln(n_labels), used to normalize.

    mode:
      "total"     -> 1 - Total/ln(n)     (overall predictive confidence)
      "epistemic" -> 1 - EU_norm         (targets model-knowledge gaps /
                                          hallucination-like uncertainty)
      "aleatoric" -> 1 - AU/ln(n)        (targets irreducible input ambiguity)
    Empirically (SST-2/HaluEval) "total" and "epistemic" both separate correct
    from incorrect verdicts well on a capable judge; "aleatoric" rises when the
    input/context is insufficient (validated by context-ablation).
    """
    hmax = math.log(max(n_labels, 2))
    if mode == "epistemic":
        eu = max(epistemic, 0.0)              # EU can go slightly negative (N=1 approx)
        return float(max(0.0, min(1.0, 1.0 - eu / hmax)))
    if mode == "aleatoric":
        return float(max(0.0, min(1.0, 1.0 - aleatoric / hmax)))
    return float(max(0.0, min(1.0, 1.0 - total / hmax)))


class SelfFVConfidence(SingleTurnMetric):
    """White-box confidence scorer as a genuine ragas ``SingleTurnMetric``.

    Requires a loaded HF judge (model + tokenizer), the CIE-selected top heads,
    the injection target layer, and the label-space option token ids. Produce
    these once with the calibration pipeline (see ``example.py`` / the paper's
    S1 CIE head selection), then reuse across queries.

    Because it holds its own open-weight judge and reads internal activations,
    it does NOT use ragas' ``llm=`` black-box path (it subclasses
    ``SingleTurnMetric`` directly, not ``MetricWithLLM``). It plugs into
    ``ragas.evaluate(dataset, metrics=[scorer])`` and into direct
    ``await scorer.single_turn_ascore(sample)`` calls.

    Parameters
    ----------
    model, tokenizer
        A preloaded Hugging Face causal LM and its tokenizer (see
        ``self_fv.load_lm``).
    top_heads
        CIE-selected ``[(layer_idx, head_idx), ...]`` causal heads.
    target_layer
        Decoder-layer index at which the self-FV is injected.
    option_token_ids
        Token ids of the discrete verdict labels (e.g. ``" A"``, ``" B"``).
    fv_scale, mode
        Self-FV magnitude knob and confidence-mapping mode (see
        ``uncertainty_to_confidence``).
    prompt_builder
        Callable ``dict -> str`` turning a ``SingleTurnSample`` dict into the
        judge prompt. Defaults to :func:`default_judge_prompt`. Supply your own
        to match a non-default label space or to inject ICL demonstrations.
    name
        Metric name surfaced in ragas result columns.
    """

    def __init__(self, model, tokenizer, top_heads, target_layer,
                 option_token_ids, fv_scale: float = 0.3, mode: str = "total",
                 prompt_builder: t.Optional[t.Callable[[t.Mapping[str, t.Any]], str]] = None,
                 name: str = "self_fv_confidence"):
        # ragas Metric/SingleTurnMetric state (normally set by the parent
        # dataclass; we set it explicitly since we take white-box positional
        # args that have no sensible dataclass defaults).
        self.name = name
        self._required_columns = {
            MetricType.SINGLE_TURN: {"user_input", "retrieved_contexts", "response"}
        }
        self.output_type = MetricOutputType.CONTINUOUS
        self.allowed_values = (0.0, 1.0)

        # White-box judge configuration.
        self.model = model
        self.tok = tokenizer
        self.top_heads = top_heads
        self.target_layer = target_layer
        self.option_token_ids = option_token_ids
        self.fv_scale = fv_scale
        self.mode = mode
        self.prompt_builder = prompt_builder or default_judge_prompt

    # -- ragas Metric interface --------------------------------------------
    def init(self, run_config) -> None:
        """Validate the white-box judge is present.

        Overrides ``Metric.init`` (base.py:141). ragas calls this before an
        evaluation run; a white-box metric holds its own HF model, so there is
        nothing to configure from ``run_config`` beyond a sanity check.
        """
        if self.model is None or self.tok is None:
            raise ValueError(
                f"Metric '{self.name}' requires a preloaded HF model + tokenizer "
                "(white-box); none were provided."
            )

    async def _single_turn_ascore(self, sample, callbacks=None) -> float:
        """Score one ``SingleTurnSample`` -> confidence in [0, 1].

        Overrides ``SingleTurnMetric._single_turn_ascore`` (base.py:504-513),
        the coroutine ``single_turn_ascore`` / ``single_turn_score`` (and hence
        ``ragas.evaluate``) dispatch to. Builds the judge prompt from the
        sample's context+response, runs the self-FV uncertainty decomposition,
        and maps it to a confidence.
        """
        row = sample.to_dict()
        prompt = self.prompt_builder(row)
        out = self.score_prompt(prompt)
        return out["confidence"]

    # -- thin white-box path (usable without ragas) ------------------------
    def score_prompt(self, prompt: str) -> t.Dict[str, float]:
        """Return dict(total, aleatoric, epistemic, confidence, verdict) for a
        fully-formatted discrete-choice judge prompt (ending right before the
        answer token). This is the raw white-box path; the ragas interface
        (:meth:`_single_turn_ascore`) is the primary entry point."""
        import torch
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        d = decompose_uncertainty(
            self.model, inputs, self.option_token_ids,
            self.top_heads, self.target_layer, fv_scale=self.fv_scale,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]
        cols = torch.tensor(self.option_token_ids, device=logits.device)
        verdict = int(logits.index_select(-1, cols).argmax(-1))
        conf = uncertainty_to_confidence(
            d["total"], d["aleatoric"], d["epistemic"],
            n_labels=len(self.option_token_ids), mode=self.mode,
        )
        return {**d, "verdict": verdict, "confidence": conf}
