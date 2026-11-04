# prompt_optimization

Improve a RAG app's **own** generation prompt with DSPy, and prove the gain with Ragas evaluations.

```
ragas eval (before) ──► optimize ──► ragas eval (after) ──► before/after comparison
                          │
                          ├─ Layer 1: few-shot search   (instruction frozen, golden demos only)
                          ├─ Layer 2: MIPROv2           (demos frozen, new instructions)
                          └─ benchmark on test          (rank all candidates, apply your floors)
```

The before and after evaluations run on a **holdout** split that the optimizer never sees. A round is accepted only when the paired per-row gain on your target metric is significant (a 95% bootstrap CI above zero) and every floor still holds.

## Quick start

```bash
ragas quickstart prompt_optimization -o ./my-project
cd my-project/prompt_optimization

python optimize.py --dry-run          # offline simulation: whole flow, no network, $0
export ANTHROPIC_API_KEY=...          # or: ant auth login
python optimize.py --smoke            # 2-3 live calls (~$0.01): checks key and prompt caching
python optimize.py --max-usd 5        # full live run
```

Inside the ragas repo the same commands are `ragas-prompt-optimization` and `ragas-prompt-optimization-eval`.

## Inputs

| File | What it is |
|---|---|
| `datasets/hr_policy_qa_golden.jsonl` | Golden dataset: `user_input`, `retrieved_contexts`, `reference`, `split` (train / eval / test / holdout), `answerable`. 80 fictional HR-policy rows, 20 per split, 3 unanswerable per split. |
| `prompts/baseline_system_prompt.md` | The prompt your app uses today. It is deliberately weak in this example. |
| `config/optimization_goal.json` | The target metric and its goal, plus **your** floors. |

### Splits and what each is used for

| Split | Used by |
|---|---|
| train | Pool of few-shot demos (Layer 1), and MIPROv2's training data (Layer 2) |
| eval | Scoring Layer 1 demo sets and Layer 2 trials (minibatches) |
| test | Benchmark that ranks every candidate prompt and picks the winner |
| holdout | Ragas eval before and after; the improvement verdict. Never used for selection. |

Because the winner is picked on test, its test score is optimistic. The holdout numbers are the honest ones.

### Goal and floors

```json
{
  "target_metric": "token_f1",
  "target_value": 0.9,
  "floors": { "token_recall": 0.80, "exact_match": "baseline" },
  "floor_tolerance": 0.0
}
```

- Metrics available without an LLM judge: `token_f1`, `token_precision`, `token_recall`, `token_f0_5`, `exact_match` (normalized).
- Only metrics listed under `floors` are constrained.
- A floor is either an absolute number such as `0.7`, or `"baseline"`, which means "no worse than the starting prompt", minus `floor_tolerance`.
- Prefer an absolute recall floor: a verbose starting prompt has inflated recall, so `"baseline"` can reject concise, better prompts. See docs/howtos/applications/optimize_rag_prompt.md for live results.

## Outputs (`runs/<timestamp>/`)

| File | Contents |
|---|---|
| `comparison.md` / `.json` | Before vs after on holdout for every metric, with the paired CI on the target |
| `loop_report.md` / `.json` | Each round's verdict (improved / no significant change / regressed / floor violated), gain and CI |
| `round_N/round_summary.json` | All Layer 1 demo sets, all Layer 2 instructions, and the test leaderboard |
| `final/final_prompt.md` | The final prompt with section markers (see below) |
| `final/final_prompt.json` | The same prompt, machine-readable; load it with `ragas.optimizers.prompt_optimization.load_prompt_bundle` |
| `final/anthropic_request_template.json` | A ready request body with cache breakpoints |
| `ragas_eval/experiments/before.csv`, `after.csv` | Per-row Ragas experiment results |
| `spend.json` | Cost by stage and by model |

## Section markers and cache breakpoints

`final_prompt.md` is split by markers that code can parse:

```
<!-- RAGAS:INSTRUCTION:START --> ... <!-- RAGAS:INSTRUCTION:END -->     stable  -> cache breakpoint 1
<!-- RAGAS:FEW_SHOTS:START -->                                           stable  -> cache breakpoint 2
  <!-- RAGAS:SHOT 1 id=hr018 --> ### context / ### question / ### answer
<!-- RAGAS:FEW_SHOTS:END -->
<!-- RAGAS:INPUT:START --> {context} {question} <!-- RAGAS:INPUT:END -->  changes  -> never cached
```

To build the breakpoints in your app (or use `rag.py`, which does this for you):

1. Put the INSTRUCTION text in `system` with `"cache_control": {"type": "ephemeral"}`.
2. Add each shot as a user turn (context + question) followed by an assistant turn (answer), and put `cache_control` on the **last** shot's assistant turn.
3. Add the live context + question as the final user turn, with no `cache_control`.
4. Keep the cached bytes identical between requests. Nothing that varies may appear before the last breakpoint.
5. Caching starts once the prefix reaches 512 tokens on Claude Haiku 5.5 and Sonnet 5.5. At most 4 breakpoints per request.
6. Check it works: from the second identical request on, `usage.cache_read_input_tokens > 0`.

## Models and cost

- Student (the model your app runs): `claude-haiku-5-5`, at $0.10 input / $0.50 output per million tokens.
- Proposer (writes the instructions): `claude-sonnet-5-5`, at $2 / $10.
- One round costs about **$1** with the defaults, and three rounds about $3.
- `--max-usd` is a hard cap that stops the run cleanly.
- DSPy's disk cache makes repeated identical calls free.

All model calls go through `ragas.llms.llm_factory(..., provider="anthropic")` and a DSPy engine (`RagasAnthropicEngine`). The engine adds the cache breakpoints, sends no sampling parameters, sets `effort`, and charges every call to the budget.

## Useful flags

```
--rounds 3            maximum improvement rounds
--min-gain 0.02       minimum mean holdout gain to accept a round
--demo-sets 16        Layer 1 random demo sets      --max-demos 4
--instructions 8      Layer 2 instruction candidates --trials 20 --minibatch-size 10
--student / --proposer / --effort
--simulated-behavior insensitive|recall_dropping   (dry-run controls)
```
