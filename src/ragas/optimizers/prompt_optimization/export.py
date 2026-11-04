"""Export a prompt bundle with section markers and cache-breakpoint guidance."""

from __future__ import annotations

import json
import re
import typing as t
from pathlib import Path

from ragas.optimizers.prompt_optimization.bundle import FewShot, PromptBundle

INSTRUCTION_START = "<!-- RAGAS:INSTRUCTION:START -->"
INSTRUCTION_END = "<!-- RAGAS:INSTRUCTION:END -->"
FEW_SHOTS_START = "<!-- RAGAS:FEW_SHOTS:START -->"
FEW_SHOTS_END = "<!-- RAGAS:FEW_SHOTS:END -->"
INPUT_START = "<!-- RAGAS:INPUT:START -->"
INPUT_END = "<!-- RAGAS:INPUT:END -->"
SHOT_MARKER = "<!-- RAGAS:SHOT {n} id={row_id} -->"
_SHOT_RE = re.compile(r"<!-- RAGAS:SHOT (\d+) id=(\S+) -->")
_FIELD_RE = re.compile(r"^### (context|question|answer)\s*$", re.MULTILINE)

INPUT_TEMPLATE = "### context\n{context}\n\n### question\n{question}\n\n### answer\n"

CACHE_GUIDE = """\
## Building cache breakpoints

The prompt is split into three sections so the stable parts can be cached:

| Section | Changes per request? | Cache breakpoint |
|---|---|---|
| INSTRUCTION | No | Breakpoint 1, on the system block |
| FEW_SHOTS | No | Breakpoint 2, on the last few-shot assistant turn |
| INPUT (context + question) | Yes | None: always after the last breakpoint |

How to build them with the Anthropic Messages API:

1. Put the INSTRUCTION text in `system` as one text block with
   `"cache_control": {"type": "ephemeral"}`.
2. Add each few-shot as a user turn (its context and question) followed by an
   assistant turn (its answer). Put `"cache_control": {"type": "ephemeral"}`
   on the last few-shot assistant turn's text block.
3. Add the live request (retrieved context + question) as the final user turn,
   with no `cache_control`.
4. Keep the cached bytes identical between requests. Do not put timestamps,
   request IDs or the question anywhere before the last breakpoint.
5. Caching only starts when the cached prefix is long enough (512 tokens on
   Claude Haiku 5.5 and Sonnet 5.5). Use at most 4 breakpoints per request.
6. Check it works: from the second identical request on,
   `usage.cache_read_input_tokens` should be greater than 0.

`anthropic_request_template.json` next to this file is a ready-made request
body with both breakpoints in place.
"""


def few_shot_user_text(shot: FewShot) -> str:
    return f"### context\n{shot.context_text}\n\n### question\n{shot.question}"


def input_user_text(context: str, question: str) -> str:
    return f"### context\n{context}\n\n### question\n{question}"


def render_markdown(bundle: PromptBundle, scores: t.Optional[t.Mapping] = None) -> str:
    parts = ["# Optimized RAG prompt", ""]
    if scores:
        parts += ["```json", json.dumps(scores, indent=2), "```", ""]
    parts += [INSTRUCTION_START, bundle.instruction.strip(), INSTRUCTION_END, ""]
    parts.append(FEW_SHOTS_START)
    for n, shot in enumerate(bundle.few_shots, start=1):
        parts.append(SHOT_MARKER.format(n=n, row_id=shot.row_id))
        parts.append(f"### context\n{shot.context_text}\n")
        parts.append(f"### question\n{shot.question}\n")
        parts.append(f"### answer\n{shot.answer}")
    parts += [
        FEW_SHOTS_END,
        "",
        INPUT_START,
        INPUT_TEMPLATE.rstrip("\n"),
        INPUT_END,
        "",
    ]
    parts += [CACHE_GUIDE]
    return "\n".join(parts)


def anthropic_request_template(bundle: PromptBundle, model: str) -> t.Dict[str, t.Any]:
    """A Messages API request body with cache breakpoints after each stable section."""
    messages: t.List[t.Dict[str, t.Any]] = []
    for shot in bundle.few_shots:
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": few_shot_user_text(shot)}],
            }
        )
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": shot.answer}]}
        )
    if messages:
        messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": input_user_text("{context}", "{question}")}
            ],
        }
    )
    return {
        "model": model,
        "max_tokens": 1024,
        "system": [
            {
                "type": "text",
                "text": bundle.instruction,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": messages,
    }


def export_bundle(
    bundle: PromptBundle,
    out_dir: t.Union[str, Path],
    model: str,
    scores: t.Optional[t.Mapping[str, t.Any]] = None,
) -> t.Dict[str, Path]:
    """Write final_prompt.md, final_prompt.json and anthropic_request_template.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": out / "final_prompt.md",
        "json": out / "final_prompt.json",
        "request_template": out / "anthropic_request_template.json",
    }
    paths["markdown"].write_text(render_markdown(bundle, scores), encoding="utf-8")
    paths["json"].write_text(
        json.dumps({**bundle.to_dict(), "scores": dict(scores or {})}, indent=2),
        encoding="utf-8",
    )
    paths["request_template"].write_text(
        json.dumps(anthropic_request_template(bundle, model), indent=2),
        encoding="utf-8",
    )
    return paths


def _between(text: str, start: str, end: str) -> str:
    try:
        begin = text.index(start) + len(start)
        finish = text.index(end, begin)
    except ValueError as e:
        raise ValueError(f"marker {start!r} or {end!r} not found") from e
    return text[begin:finish]


def _parse_shot(n: int, row_id: str, body: str) -> FewShot:
    fields: t.Dict[str, str] = {}
    matches = list(_FIELD_RE.finditer(body))
    for i, match in enumerate(matches):
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        fields[match.group(1)] = body[match.end() : stop].strip("\n").rstrip()
    missing = {"context", "question", "answer"} - set(fields)
    if missing:
        raise ValueError(f"shot {n} is missing {sorted(missing)}")
    contexts = tuple(
        re.sub(r"^\[\d+\] ", "", chunk)
        for chunk in fields["context"].split("\n\n")
        if chunk
    )
    return FewShot(
        row_id=row_id,
        question=fields["question"],
        contexts=contexts,
        answer=fields["answer"],
    )


def parse_markdown(text: str) -> PromptBundle:
    """Parse a ``final_prompt.md`` back into a bundle using its markers."""
    instruction = _between(text, INSTRUCTION_START, INSTRUCTION_END).strip()
    shots_block = _between(text, FEW_SHOTS_START, FEW_SHOTS_END)
    markers = list(_SHOT_RE.finditer(shots_block))
    shots = []
    for i, marker in enumerate(markers):
        stop = markers[i + 1].start() if i + 1 < len(markers) else len(shots_block)
        body = shots_block[marker.end() : stop]
        shots.append(_parse_shot(int(marker.group(1)), marker.group(2), body))
    return PromptBundle(instruction=instruction, few_shots=shots, source="exported")


def load_prompt_bundle(path: t.Union[str, Path]) -> PromptBundle:
    """Load a bundle from ``final_prompt.json`` or ``final_prompt.md``."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".json":
        return PromptBundle.from_dict(json.loads(text))
    return parse_markdown(text)
