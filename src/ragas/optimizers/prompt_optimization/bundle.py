"""Data types shared by every stage: golden rows, few-shot demos, prompt bundles."""

from __future__ import annotations

import hashlib
import json
import typing as t
from dataclasses import asdict, dataclass, field

SPLITS = ("train", "eval", "test", "holdout")


@dataclass(frozen=True)
class GoldenRow:
    """One validated row of the golden dataset."""

    id: str
    question: str
    contexts: t.Tuple[str, ...]
    reference: str
    split: str
    answerable: bool = True

    @property
    def context_text(self) -> str:
        return format_contexts(self.contexts)


@dataclass(frozen=True)
class FewShot:
    """A few-shot demo copied verbatim from a golden training row."""

    row_id: str
    question: str
    contexts: t.Tuple[str, ...]
    answer: str

    @classmethod
    def from_row(cls, row: GoldenRow) -> "FewShot":
        return cls(
            row_id=row.id,
            question=row.question,
            contexts=row.contexts,
            answer=row.reference,
        )

    @property
    def context_text(self) -> str:
        return format_contexts(self.contexts)


@dataclass
class PromptBundle:
    """A complete prompt: instruction plus few-shot demos.

    The per-request input (retrieved context and question) is not part of the
    bundle; it is filled in at call time after the cached prefix.
    """

    instruction: str
    few_shots: t.List[FewShot] = field(default_factory=list)
    source: str = "baseline"
    metadata: t.Dict[str, t.Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "instruction": self.instruction,
                "few_shots": [asdict(s) for s in self.few_shots],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def with_instruction(self, instruction: str, source: str) -> "PromptBundle":
        return PromptBundle(
            instruction=instruction,
            few_shots=list(self.few_shots),
            source=source,
            metadata=dict(self.metadata),
        )

    def with_few_shots(
        self, few_shots: t.Sequence[FewShot], source: str
    ) -> "PromptBundle":
        return PromptBundle(
            instruction=self.instruction,
            few_shots=list(few_shots),
            source=source,
            metadata=dict(self.metadata),
        )

    @property
    def prompt_length(self) -> int:
        """Characters in the instruction and demos; used as a tie-breaker."""
        return len(self.instruction) + sum(
            len(s.question) + len(s.context_text) + len(s.answer)
            for s in self.few_shots
        )

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            "instruction": self.instruction,
            "few_shots": [
                {**asdict(s), "contexts": list(s.contexts)} for s in self.few_shots
            ],
            "source": self.source,
            "metadata": self.metadata,
            "fingerprint": self.fingerprint(),
        }

    @classmethod
    def from_dict(cls, data: t.Mapping[str, t.Any]) -> "PromptBundle":
        return cls(
            instruction=data["instruction"],
            few_shots=[
                FewShot(
                    row_id=s["row_id"],
                    question=s["question"],
                    contexts=tuple(s["contexts"]),
                    answer=s["answer"],
                )
                for s in data.get("few_shots", [])
            ],
            source=data.get("source", "loaded"),
            metadata=dict(data.get("metadata", {})),
        )


def format_contexts(contexts: t.Sequence[str]) -> str:
    """Render retrieved chunks as numbered blocks."""
    return "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(contexts, start=1))
