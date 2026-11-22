"""Minimal end-to-end example: uncertainty-calibrated faithfulness judge.

Builds a local open-weight faithfulness judge, calibrates self-FV causal heads
(paper S1) on a few HaluEval-QA examples, then scores held-out (context, answer)
pairs with a self-FV confidence. Low confidence flags an unreliable verdict.

Runs both ways:
    python -m self_fv_confidence.example Qwen/Qwen3-0.6B      # installed package
    python example.py Qwen/Qwen3-0.6B                          # standalone

The final block shows the GENUINE ragas path: wrapping an example in a
``SingleTurnSample`` and scoring it through ``metric.single_turn_score(...)``
(the same call ``ragas.evaluate`` makes), when ragas is installed.

Requires: transformers>=4.57, datasets, torch. Judge quality (and hence the
confidence's usefulness) depends on the model — use a capable base model.
"""
import sys
from datasets import load_dataset

# Dual-mode imports: relative when run as a package module, absolute when the
# file is executed directly from its own directory.
try:
    from .self_fv import (
        load_lm, attention_layer_indices, collect_mean_head_act, compute_cie,
    )
    from .metric import SelfFVConfidence, RAGAS_AVAILABLE
except ImportError:
    from self_fv import (
        load_lm, attention_layer_indices, collect_mean_head_act, compute_cie,
    )
    from metric import SelfFVConfidence, RAGAS_AVAILABLE

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
LETTERS = ["A", "B"]  # A=faithful, B=not faithful
HEADER = ("Judge whether the Answer is faithful to (fully supported by) the "
          "Context. Options: A=faithful, B=not faithful.\n\n")

def fmt(ctx, q, ans, letter=None):
    ctx = " ".join(str(ctx).split())[:1500]
    s = f"Context: {ctx}\nQuestion: {q}\nAnswer: {ans}\nFaithful:"
    return s + (f" {letter}\n\n" if letter is not None else "")

def build_prompt(demos, item):
    body = HEADER + "".join(fmt(*d[:3], LETTERS[d[3]]) for d in demos)
    return body + fmt(item[0], item[1], item[2])

def main():
    tok, model = load_lm(MODEL)
    opt_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS]
    att = attention_layer_indices(model)
    target_layer = att[len(att) // 3]

    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    items = []
    for r in ds.select(range(40)):
        items.append((r["knowledge"], r["question"], r["right_answer"], 0))
        items.append((r["knowledge"], r["question"], r["hallucinated_answer"], 1))
    demos, calib, test = items[:6], items[6:26], items[26:32]

    def toks(p): return tok(p, return_tensors="pt").to(model.device)

    # S1: CIE head selection (clean prompts) — one-time calibration
    mean_act = collect_mean_head_act(
        model, [toks(build_prompt(demos, x)) for x in calib], att_layers=att)
    noised = [toks(build_prompt([(*d[:3], 1 - d[3]) for d in demos], x)) for x in calib]
    top, _ = compute_cie(model, noised, [x[3] for x in calib], mean_act, opt_ids,
                         att_layers=att)

    # An ICL-aware prompt builder so the ragas SingleTurnSample path reuses the
    # few-shot demos. Maps SingleTurnSample fields -> the judge prompt.
    def icl_prompt_builder(row):
        ctx = row.get("retrieved_contexts") or []
        if isinstance(ctx, (list, tuple)):
            ctx = " ".join(str(c) for c in ctx)
        return build_prompt(demos, (ctx, row.get("user_input") or "",
                                    row.get("response") or ""))

    scorer = SelfFVConfidence(model, tok, top[:16], target_layer, opt_ids,
                              fv_scale=0.3, mode="total",
                              prompt_builder=icl_prompt_builder)

    # --- raw white-box path -------------------------------------------------
    print(f"# self-FV faithfulness judge ({MODEL})")
    for x in test:
        out = scorer.score_prompt(build_prompt(demos, x))
        print(f"verdict={LETTERS[out['verdict']]} gold={LETTERS[x[3]]} "
              f"confidence={out['confidence']:.2f} "
              f"(Total={out['total']:.3f} AU={out['aleatoric']:.3f} EU={out['epistemic']:+.3f})")

    # --- genuine ragas path -------------------------------------------------
    if RAGAS_AVAILABLE:
        import asyncio
        from ragas.dataset_schema import SingleTurnSample

        print("\n# via ragas SingleTurnMetric interface")
        for x in test:
            sample = SingleTurnSample(
                user_input=x[1], retrieved_contexts=[x[0]], response=x[2])
            conf = asyncio.run(scorer.single_turn_ascore(sample))
            print(f"gold={LETTERS[x[3]]} single_turn_ascore={conf:.2f}")
    else:
        print("\n[info] ragas not installed; skipped SingleTurnMetric demo. "
              "Install ragas to use scorer.single_turn_ascore / ragas.evaluate.")

if __name__ == "__main__":
    main()
