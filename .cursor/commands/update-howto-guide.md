# Update How-to Guide

Updates the mentioned how-to guide to use src/ragas/metrics/collections API instead of the legacy ragas/metrics API and LLM factory pattern instead of langchainwrapper.

## File Format Decision

If the source is an `.ipynb` file (or if the `.md` filename starts with `_`, indicating it's derived from a notebook via `docs/ipynb_to_md.py`):

1. **Delete** the `.ipynb` file
2. **Delete** the corresponding `_xxx.md` file (if it exists)
3. **Create** a new `.md` file directly (without the `_` prefix)

This simplifies maintenance by having pure markdown docs instead of notebooks.


## Process

### Phase 1: Research (do NOT make changes yet)

Refer pr-description-customizations.md for the list of guides that are already updated. And finally update the doc after you're done. 

#### 1.1 Understand the Guide's Purpose
- Read the target file thoroughly
- Identify **what the guide is trying to achieve** (e.g., caching, run config, retry handling)
- Note the specific use case or need the guide addresses
- Understand what underlying tools/libraries are being used (e.g., instructor, liteLLM, httpx)

#### 1.2 Feasibility Check
Before doing anything else, check if the feature works with the new API:

1. **Check `src/ragas/experiment.py`** - Does experiment() support this feature?
2. **Check `src/ragas/evaluation.py`** - Is this an evaluate()-only feature?
3. **Check `src/ragas/metrics/collections/`** - Do collections metrics support this?
4. **Check if simpler alternatives exist** - Does a newer, simpler API make this guide obsolete? (e.g., decorator-based metrics vs subclassing, built-in features vs manual workarounds). Check concept docs and `src/ragas/metrics/` for modern patterns.

**If a simpler approach exists → recommend deletion** instead of migration. See "When to Recommend Deletion" section.

**If not supported in new API → STOP immediately:**
- Keep guide as-is
- Output this Slack message for the team:

```
📋 *Doc Update Skipped*: `<guide_path>`
*Link*: https://docs.ragas.io/en/latest/<guide_path_without_extension>/
*Reason*: <feature> only works with legacy `evaluate()` API, not yet supported in `experiment()`/collections
*Action*: Keep as-is until collections API adds support
```

**If supported → continue to 1.3**

#### 1.3 Present Plan & Wait for Approval

**⏸️ STOP HERE - Do NOT proceed to Phase 2 without explicit user approval.**

Present a clear summary:
1. **Current state**: What the guide currently does and how
2. **Proposed changes**: 
   - Imports to change (from old → new)
   - LLM/embeddings setup patterns to update
   - **How the specific use case/feature will be achieved** with the new API
   - Any restructuring or content changes
3. **Potential concerns**: Anything uncertain or risky
4. **Ask**: "Does this plan look good? Should I proceed?"

**Wait for user to say "yes", "proceed", "go ahead", or similar before continuing.**

---

### Phase 2: Execute (only after approval)

#### 2.1 Apply Updates

**Keep it Concise**: 
- Remove unnecessary explanations and verbose text
- Focus on the essential information needed to achieve the goal
- Use clear, direct language
- Avoid redundant examples - one good example is better than multiple similar ones

**Import Updates**:
```python
# Change from:
from ragas.metrics import MetricName

# To:
from ragas.metrics.collections import MetricName
```

**LLM Setup**:
```python
# Change from:
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

# To:
from openai import OpenAI
from ragas.llms import llm_factory
client = OpenAI(api_key="sk-...")
llm = llm_factory("gpt-4o", client=client)
```

**Embeddings Setup**:
```python
# Change from:
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# To:
from openai import OpenAI
from ragas.embeddings.base import embedding_factory  # Use .base to avoid deprecation warning
client = OpenAI(api_key="sk-...")
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)
```

**What to Fix**:
- Update imports and LLM/embeddings patterns
- Use `ragas.embeddings.base` import to avoid deprecation warnings
- **Replace all legacy code with modern approaches** (no need to keep legacy sections)
- Fix minor issues automatically
- Don't restructure content unless fixing issues

#### 2.2 Verify Accuracy & Test Code

**Verify with Web Search**:
- Search for official documentation of any libraries/tools mentioned (instructor, liteLLM, httpx, etc.)
- Confirm API signatures, parameter names, and usage patterns are correct
- Verify any claims about library behavior are accurate

**Run the Code**:
- Install any missing packages first: `uv pip install <package>`
- Extract ALL Python code blocks from the guide
- Save as `tests/docs/test_<guide_name>.py` (e.g., `test_run_config.py`)
- Use `.env` from root for API keys. .env only has openai keys, if you need anything else, let me know
- Run: `uv run python tests/docs/test_<guide_name>.py`
- **Verify the original use case/goal is achieved** with the new approach

**If tests fail**:
1. Check the underlying implementation in `src/` to understand correct usage
2. Fix the code in the guide based on what you learn from `src/`
3. Re-run the test
4. Repeat until tests pass
5. If stuck after multiple attempts, report the issue with details

**Keep the test file** - excluded from default `pytest` runs via `norecursedirs` in `pyproject.toml`.

**Both verification methods are required** - web search for accuracy, code execution for functionality.

#### 2.3 Check Navigation
- Verify file is in `mkdocs.yml`
- Note if location seems wrong or can be put in a more appropriate section

#### 2.4 Summarize Changes
- List all changes made
- Mention if anything is not tested due to any reasons (like missing packages, missing API keys, etc.)

## Notes
- **Two-phase workflow**: Research first, get approval, then execute
- **Never skip approval**: Always present the plan and wait for explicit "go ahead" before making changes
- **This is not just a straightforward migration** - understand if the original goal is achievable first
- **Keep guides concise** - remove fluff, focus on essential information
- **Verify accuracy** - use web search to confirm library APIs and behavior before writing
- **Test everything** - run all code examples before finalizing
- Only fix what's broken or outdated
- Check `src/` before updating to verify APIs exist
- Don't add legacy sections
- Use root `.env` for testing
- **Keep test files** in `tests/docs/` - excluded from default pytest runs

## When to Recommend Deletion

If a guide teaches **writing custom metrics by subclassing** (`MetricWithLLM`, `SingleTurnMetric`, etc.), it's likely obsolete. The decorator-based approach is simpler:

```python
from ragas.metrics import discrete_metric, numeric_metric, ranking_metric

@discrete_metric(name="my_metric", allowed_values=["pass", "fail"])
def my_metric(response: str, context: str) -> str:
    return "pass" if condition else "fail"
```

See `docs/concepts/metrics/overview/index.md` for details. Recommend deletion if decorators cover the use case.

## Reporting Gaps
If you identify a gap, use the Slack message template from section 1.2.
