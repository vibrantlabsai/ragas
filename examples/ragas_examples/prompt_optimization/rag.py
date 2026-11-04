"""Minimal RAG generation step that uses an exported, optimized prompt.

The exported bundle (``final_prompt.json``) is turned into an Anthropic
Messages request with two prompt-cache breakpoints:

1. after the INSTRUCTION (system block)
2. after the last FEW_SHOT (assistant turn)

Only the live context + question come after the last breakpoint, so repeated
requests reuse the cached prefix.

    python rag.py runs/<ts>/final/final_prompt.json "How many sick days do employees get?" \
        "Sick leave is tracked separately from PTO: employees receive 8 sick days per year."
"""

from __future__ import annotations

import copy
import sys
import typing as t
from pathlib import Path


def build_request(
    bundle_path: t.Union[str, Path],
    question: str,
    contexts: t.Sequence[str],
    model: str = "claude-haiku-5-5",
    max_tokens: int = 1024,
) -> t.Dict[str, t.Any]:
    """Fill the exported request template with a live question and contexts."""
    from ragas.optimizers.prompt_optimization import load_prompt_bundle
    from ragas.optimizers.prompt_optimization.bundle import format_contexts
    from ragas.optimizers.prompt_optimization.export import (
        anthropic_request_template,
        input_user_text,
    )

    bundle = load_prompt_bundle(bundle_path)
    request = copy.deepcopy(anthropic_request_template(bundle, model))
    request["max_tokens"] = max_tokens
    request["messages"][-1]["content"][0]["text"] = input_user_text(
        format_contexts(contexts), question
    )
    return request


def answer(
    bundle_path: t.Union[str, Path],
    question: str,
    contexts: t.Sequence[str],
    model: str = "claude-haiku-5-5",
    client: t.Any = None,
) -> t.Tuple[str, t.Any]:
    """Call Claude with the optimized prompt; returns (answer text, usage)."""
    import anthropic

    client = client or anthropic.Anthropic()
    response = client.messages.create(
        **build_request(bundle_path, question, contexts, model)
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text, response.usage


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    text, usage = answer(sys.argv[1], sys.argv[2], sys.argv[3:])
    print(text)
    print(f"\ncache_read_input_tokens={getattr(usage, 'cache_read_input_tokens', 0)}")
