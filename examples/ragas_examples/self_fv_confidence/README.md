# self-FV confidence — white-box uncertainty for open-weight LLM judges

An experimental ragas example that ports **self-function vectors** (Chung et
al. 2026, [arXiv:2606.19353](https://arxiv.org/abs/2606.19353)) into a
confidence signal for **discrete LLM-as-judge verdicts** (faithfulness,
correctness, aspect-critic). It reads the judge's internal attention-head
activations to decompose the verdict's uncertainty into **aleatoric** (data /
context insufficiency) and **epistemic** (model-knowledge) components, then
returns a confidence in `[0, 1]` — so low-confidence verdicts can be routed to
human review.

## When it applies

- **White-box only.** Needs the residual stream + per-head `o_proj` activations,
  so the judge must be a **local Hugging Face model** (Qwen3, Gemma-4, Llama,
  Mistral, …), not an API endpoint. This is the core constraint.
- **Discrete verdict.** Uncertainty is entropy over the option tokens (e.g.
  `A=faithful / B=not`). Free-form generation is out of scope.
- **Capable judge.** The decomposition is only meaningful when the model
  actually performs the task; weak/instruct-on-raw-ICL models give degenerate
  or inverted signals.

## API

`SelfFVConfidence` is a **genuine ragas `SingleTurnMetric`** (from
`ragas.metrics.base`). It overrides `_single_turn_ascore`, so it plugs straight
into `ragas.evaluate(...)` and `await metric.single_turn_ascore(sample)`. Because
it reads a local model's internals it holds its own HF `model`+`tokenizer` and
does **not** use ragas' black-box `llm=` path.

```python
from self_fv_confidence import (
    load_lm, attention_layer_indices, collect_mean_head_act, compute_cie,
    SelfFVConfidence,
)

tok, model = load_lm("google/gemma-4-26B-A4B")          # local HF judge
att = attention_layer_indices(model)                     # skips linear-attn layers
# S1: one-time CIE causal-head calibration on a few labeled judge prompts
mean_act = collect_mean_head_act(model, clean_prompts, att_layers=att)
top, _   = compute_cie(model, noised_prompts, true_idx, mean_act, opt_ids, att_layers=att)

scorer = SelfFVConfidence(model, tok, top[:16], att[len(att)//3], opt_ids, mode="total")
```

Use it through the ragas metric interface:

```python
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample

sample = SingleTurnSample(
    user_input="Who painted the Mona Lisa?",
    retrieved_contexts=["The Mona Lisa was painted by Leonardo da Vinci."],
    response="Leonardo da Vinci painted the Mona Lisa.",
)

# 1) direct async score (returns a float in [0, 1])
confidence = await scorer.single_turn_ascore(sample)      # or scorer.single_turn_score(sample)

# 2) full ragas evaluation run (a `self_fv_confidence` column is produced)
result = evaluate(EvaluationDataset(samples=[sample]), metrics=[scorer])
```

By default the metric builds a zero-shot A=faithful/B=not judge prompt from the
sample's `retrieved_contexts` + `user_input` + `response`. Pass a
`prompt_builder=lambda row: ...` to use a different label space or inject
few-shot ICL demonstrations (see `example.py`).

A thin white-box escape hatch scores a fully-formatted prompt directly, without
ragas:

```python
out = scorer.score_prompt(judge_prompt)   # {verdict, confidence, total, aleatoric, epistemic}
```

See `example.py` for a runnable HaluEval-QA faithfulness demo (both paths). Run
it as `python -m self_fv_confidence.example <model>` or standalone
`python example.py <model>`.

**Verified end-to-end** on a Qwen3-0.6B judge: both `score_prompt` and the ragas
`single_turn_ascore(sample)` path return identical confidences. Tested with
`ragas==0.4.3` and a consistent langchain 0.2 set (`langchain==0.2.17`,
`langchain-core==0.2.43`, `langchain-community==0.2.19`,
`langchain-openai==0.1.25`) — ragas 0.4.3's import breaks under the langchain
0.3/1.x split (`langchain_community.chat_models.vertexai` was removed), so pin
the 0.2 generation if you hit `ModuleNotFoundError` on import.

## What generalizes beyond the paper's reference code

`self_fv.py` is a native-PyTorch-hook reimplementation (no `baukit`) that the
original repo's assumptions break on for modern models:

- derives `head_dim` and per-layer head counts from the **actual `o_proj`
  dimensions** (Qwen3/Gemma-4 decouple `head_dim` from `hidden/heads`);
- **selects only standard-attention layers** — on mechanism-hybrid models
  (Qwen3.6-35B-A3B: 30/40 layers are Gated DeltaNet linear attention) self-FV
  operates only on the softmax-attention layers;
- handles MoE FFNs, multimodal wrappers, and multi-GPU sharding.

## Validity (from the paper's Fig.2-style controls)

- **AU-control (context ablation):** removing retrieved context makes AU rise
  and EU stay flat on capable judges (Gemma-4-26B, Qwen3.6-35B) — i.e. the
  method correctly attributes *missing evidence* to the aleatoric axis.
- Error-detection AUROC on clean discrete tasks (SST-2, HaluEval) is ~0.8–0.97
  with a strong base judge; it degrades on noisy real-RAG faithfulness.

## Status

Experimental / research example. Not a drop-in core metric — it is scoped to
white-box open-weight judges and discrete verdicts, and its RAG value is still
being characterized.
