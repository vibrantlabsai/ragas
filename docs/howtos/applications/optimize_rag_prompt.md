# Optimize a RAG App's Prompt

`prompt_optimization` improves the system prompt of **your RAG app** (not Ragas's judge prompts) and proves the result with a Ragas eval on data the optimizer never saw.

```
Ragas eval (before, holdout) ─▶ Layer 1: few-shot search ─▶ Layer 2: MIPROv2 instructions ─▶ benchmark on test ─▶ Ragas eval (after, holdout) ─▶ compare
                                        ▲                                                                                   │
                                        └──────────────── next round while the holdout gain is significant ◀────────────────┘
```

## Quick start

```bash
pip install "ragas[dspy]"
ragas quickstart prompt_optimization
cd prompt_optimization
ragas-prompt-optimization --dry-run        # offline simulation, $0
echo ANTHROPIC_API_KEY=sk-ant-... > .env   # read automatically, never printed
ragas-prompt-optimization --smoke          # 3 live calls, checks key + prompt caching
ragas-prompt-optimization --max-usd 5      # full live run
```

Useful flags: `--dataset`, `--system-prompt`, `--goal`, `--student` (default `claude-haiku-5-5`), `--proposer` (default `claude-sonnet-5-5`), `--rounds`, `--min-gain`, `--demo-sets`, `--max-demos`, `--instructions`, `--trials`, `--threads`, `--out-dir`, `-v`.

## Inputs

**Golden dataset** (JSONL/JSON, list, or Ragas `Dataset`), one row per question:

| Field | Meaning |
|---|---|
| `user_input` | the question |
| `retrieved_contexts` | the chunks your retriever returns |
| `reference` | the correct short answer (for unanswerable rows: your refusal sentence) |
| `split` | `train` (demo pool), `eval` (search), `test` (pick the winner), `holdout` (before/after verdict only) |
| `answerable` | optional, default `true` |

**Goal file**:

```json
{"target_metric": "token_f1", "target_value": 0.9,
 "floors": {"token_recall": 0.80, "exact_match": "baseline"}, "floor_tolerance": 0.0}
```

Floors apply **only to metrics you list**. A floor is a number or `"baseline"` (the starting prompt's score minus `floor_tolerance`). Metrics: `token_precision`, `token_recall`, `token_f1`, `token_f0_5`, `exact_match` (normalized), and an adapter for the LLM-judged `Faithfulness`.

## How it works

1. **Layer 1, few-shot search.** The instruction is frozen. Candidate demo sets follow DSPy BootstrapRS's seed scheme (zero-shot, first-k, shuffled sets of size 1..k), but every demo is a verbatim golden `train` row in the output schema. Each set is scored on `eval`; the best set that passes the floors wins (ties go to fewer demos).
2. **Layer 2, instruction search.** The demos are frozen. `dspy.MIPROv2` in zero-shot mode proposes instructions with the proposer model and scores them in minibatches on `eval`.
3. **Benchmark.** The baseline, the Layer 1 winner and every Layer 2 candidate run on `test`. Candidates that break a floor are excluded. The highest target wins, with shorter prompts winning ties. This score is selection-biased, so it is never used as the verdict.
4. **Improvement loop.** The current and the candidate prompts are evaluated with the Ragas `@experiment` on `holdout`. A round is accepted only if the paired bootstrap 95% CI of the per-row gain is above 0, the mean gain is at least `min_gain`, and all floors hold. Each round uses fresh seeds. The loop stops when the target is reached, a round is not accepted, or the budget runs out.

LLM calls go through Ragas's `llm_factory` (`InstructorLLM`). `RagasAnthropicEngine` reuses its client and model inside DSPy. It never sends default sampling parameters (Claude 5.5 models reject non-default `temperature`/`top_p`), records spend against a hard `--max-usd` cap, and maps refusals to failed answers.

## Outputs (`runs/<timestamp>/`)

`comparison.md/json` (before vs after), `loop_report.md/json`, `round_N/` (layer state for resume, leaderboard), `ragas_eval/experiments/{before,after}.csv` (per-row answers and scores), `spend.json`, and `final/`:

- `final_prompt.md` contains the prompt with section markers that code can pull out:
  `<!-- RAGAS:INSTRUCTION:START/END -->`, `<!-- RAGAS:FEW_SHOTS:START/END -->` (each shot is `<!-- RAGAS:SHOT n id=... -->`), `<!-- RAGAS:INPUT:START/END -->`.
- `final_prompt.json` is the same content, machine-readable. Load it with `load_prompt_bundle()`.
- `anthropic_request_template.json` is a ready Messages API request with the cache breakpoints placed.

### Building cache breakpoints

Put the stable parts first and the per-request input last. **Breakpoint 1** is `cache_control: {"type": "ephemeral"}` on the system block (the instruction). **Breakpoint 2** goes on the last few-shot assistant turn. The live question and contexts come after both and are never cached. A cached prefix must be at least 512 tokens, and a request can have at most 4 breakpoints. Changing a byte in the instruction invalidates both breakpoints, while changing the shots invalidates only breakpoint 2.

## Test coverage

109 unit tests with no network access (simulated Anthropic client and stub models), plus a ≥90% coverage gate on the new code.

| # | Requirement | Tests |
|---|---|---|
| R1 | Anthropic calls send no default sampling params; explicit values are kept | `test_instructor_anthropic_params.py`, `test_prompt_opt_engine.py` |
| R2–R3 | `TokenOverlap` P/R/F1/F-beta, `ExactMatch(normalize=True)` | `test_token_overlap_collections.py` |
| R4 | Four disjoint splits, invalid rows rejected | `test_prompt_opt_dataset.py` |
| R5 | Layer 1: instruction frozen, demos are verbatim golden train rows, no leakage | `test_prompt_opt_few_shot_search.py` |
| R6 | Layer 2 (real MIPROv2): demos frozen, candidate 0 = original | `test_prompt_opt_instruction_search.py` |
| R7–R8 | User-set floors only; benchmark ranking and exclusions | `test_prompt_opt_metrics.py`, `test_prompt_opt_benchmark.py` |
| R9 | Loop accepts real gains and rejects no-gain and floor-breaking candidates | `test_prompt_opt_improvement_loop.py`, `test_prompt_opt_pipeline.py` |
| R10–R12 | Cache breakpoints, budget cap, refusals | `test_prompt_opt_engine.py`, `test_prompt_opt_budget_export.py` |
| R13–R14 | Export/load round trip, resume, fresh seeds per round | `test_prompt_opt_budget_export.py`, `test_prompt_opt_optimizer.py` |
| R15–R16 | CLI and quickstart end to end, `.env` loading, install hint | `test_prompt_opt_cli_tool.py`, `test_prompt_opt_optimizer.py` |
| R17 | No event loop created per score (Windows deadlock regression) | `test_prompt_opt_metrics.py` |

The pipeline tests use three simulated students as controls. A **prompt-sensitive** student must show a significant gain. A **prompt-insensitive** student must produce no claimed gain. A student that **drops recall** must have its candidates rejected by the floor.

```bash
uv run pytest tests/unit/optimizers/prompt_optimization tests/unit/test_token_overlap_collections.py \
  tests/unit/llms/test_instructor_anthropic_params.py
```

## Live results

Student `claude-haiku-5-5`, proposer `claude-sonnet-5-5`, effort `low`. All scores are on the 20-row **holdout** split.

| Dataset | Starting prompt | Token F1 before → after | Paired 95% CI | EM before → after | Spend |
|---|---|---|---|---|---|
| HR policy (80 rows, shipped with the tool) | "friendly, thorough, use general knowledge" | 0.091 → 0.891 | [+0.71, +0.88] | 0.00 → 0.70 | $0.13 |
| SQuAD 2.0 dev (80 rows, splits by article, 30% unanswerable) | "helpful, complete answer, explain reasoning" | 0.075 → 0.867 | [+0.65, +0.90] | 0.00 → 0.80 | $0.47 |

On SQuAD, an independent rescoring with the official SQuAD 2.0 rule (best match over all annotator answers) gives F1 0.18 → 0.87 and unanswerable accuracy 0.33 → 0.83. One answerable question in 14 was wrongly refused. Mean answer length dropped from 71 words to 4.

What the runs show:

1. **It generalizes.** SQuAD test (where the winner was chosen) scored 0.95 and holdout (unseen articles) scored 0.87. The gap is the expected selection bias. In both runs the second round was rejected because its CI included 0.
2. **Much of the gain is answer format.** On answerable SQuAD rows the baseline already contained a gold answer 13/14 times. Token metrics punish verbose answers. The content gain is mainly correct abstention. If your users want explanations, choose an LLM-judged target metric instead of token F1.
3. **Prefer absolute recall floors.** With `token_recall: "baseline"`, a verbose baseline has inflated recall, and an earlier HR run rejected a 0.85-F1 instruction for losing 0.03 recall.
4. **Review the learned instruction.** It can encode how your dataset was built (here: "two passages, one is a distractor"). Edit such lines if production retrieval differs.

## Limitations

- Token metrics compare against a single reference per row.
- Live models are non-deterministic, so rerun before trusting small gains.
- 20-row holdout splits give wide CIs, so use more rows for decisions that matter.
- The built-in engine targets Anthropic models. Other providers need their own engine.
