"""Layer 2: MIPROv2 instruction-only search with the Layer 1 demos frozen."""

from __future__ import annotations

import logging
import typing as t

from ragas.optimizers.prompt_optimization.bundle import GoldenRow, PromptBundle
from ragas.optimizers.prompt_optimization.config import InstructionSearchConfig
from ragas.optimizers.prompt_optimization.dataset import to_dspy_example
from ragas.optimizers.prompt_optimization.signature import (
    build_program,
    import_dspy,
    make_adapter,
    program_demo_answers,
    program_instruction,
)

logger = logging.getLogger(__name__)


class InstructionProposer(t.Protocol):
    """Proposes instruction variants for a bundle whose demos stay fixed."""

    def propose(
        self,
        bundle: PromptBundle,
        train_rows: t.Sequence[GoldenRow],
        eval_rows: t.Sequence[GoldenRow],
        metric: t.Callable[..., float],
        seed_offset: int = 0,
    ) -> t.List[PromptBundle]: ...


def _recording_mipro_class() -> t.Any:
    dspy = import_dspy()

    class RecordingMIPROv2(dspy.MIPROv2):
        """MIPROv2 that keeps every proposed instruction, not only trialed ones."""

        proposed_instructions: t.Dict[int, t.List[str]]

        def _propose_instructions(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
            result = super()._propose_instructions(*args, **kwargs)
            self.proposed_instructions = {k: list(v) for k, v in result.items()}
            return result

    return RecordingMIPROv2


class MIPROv2InstructionProposer:
    """Runs DSPy MIPROv2 in zero-shot mode (instructions only).

    ``max_bootstrapped_demos=0`` and ``max_labeled_demos=0`` turn off demo
    search, so the program keeps the Layer 1 demos it was given. Trials are
    scored on the eval split in minibatches. Every proposed instruction is
    returned (candidate 0 is the original), each paired with the frozen
    demos, so the benchmark can rank all of them on test.
    """

    def __init__(
        self,
        student_lm: t.Any,
        proposer_lm: t.Any,
        config: t.Optional[InstructionSearchConfig] = None,
        num_threads: int = 8,
        adapter: t.Any = None,
    ):
        self.student_lm = student_lm
        self.proposer_lm = proposer_lm
        self.config = config or InstructionSearchConfig()
        self.num_threads = num_threads
        self.adapter = adapter or make_adapter()
        self.last_run: t.Dict[str, t.Any] = {}

    def propose(
        self,
        bundle: PromptBundle,
        train_rows: t.Sequence[GoldenRow],
        eval_rows: t.Sequence[GoldenRow],
        metric: t.Callable[..., float],
        seed_offset: int = 0,
    ) -> t.List[PromptBundle]:
        dspy = import_dspy()
        if len(train_rows) < 2:
            raise ValueError("MIPROv2 needs at least 2 training rows")

        program = build_program(bundle)
        frozen_demos = program_demo_answers(program)
        trainset = [to_dspy_example(r) for r in train_rows]
        valset = [to_dspy_example(r) for r in eval_rows]
        use_minibatch = len(valset) > self.config.minibatch_size

        optimizer = _recording_mipro_class()(
            metric=metric,
            prompt_model=self.proposer_lm,
            task_model=self.student_lm,
            auto=None,
            num_candidates=self.config.num_candidates,
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
            num_threads=self.num_threads,
            seed=self.config.seed + seed_offset,
            verbose=False,
            track_stats=True,
        )
        with dspy.context(lm=self.student_lm, adapter=self.adapter):
            best = optimizer.compile(
                program,
                trainset=trainset,
                valset=valset,
                num_trials=self.config.num_trials,
                minibatch=use_minibatch,
                minibatch_size=self.config.minibatch_size if use_minibatch else 35,
                fewshot_aware_proposer=False,
            )

        # Eval scores MIPROv2 measured (Evaluate reports 0-100).
        eval_scores: t.Dict[str, float] = {}
        demos_frozen = program_demo_answers(best) == frozen_demos
        for entry in getattr(best, "candidate_programs", []) or []:
            candidate = entry["program"]
            demos_frozen = (
                demos_frozen and program_demo_answers(candidate) == frozen_demos
            )
            text = program_instruction(candidate)
            eval_scores[text] = max(
                eval_scores.get(text, 0.0), float(entry["score"]) / 100.0
            )
        if not demos_frozen:
            # Never trust demos coming back from MIPROv2: bundles below are
            # rebuilt from the instruction text plus the Layer 1 demos anyway.
            logger.warning("MIPROv2 changed demos; reattaching the Layer 1 demos")

        proposed = getattr(optimizer, "proposed_instructions", {}).get(0) or [
            program_instruction(best)
        ]
        bundles: t.List[PromptBundle] = []
        seen: t.Set[str] = set()
        for index, instruction in enumerate(proposed):
            text = instruction.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            source = "layer2:original" if index == 0 else f"layer2:instruction={index}"
            candidate_bundle = bundle.with_instruction(text, source=source)
            candidate_bundle.metadata.update(
                {"proposed_index": index, "mipro_eval_score": eval_scores.get(text)}
            )
            bundles.append(candidate_bundle)

        self.last_run = {
            "num_proposed": len(proposed),
            "num_unique": len(bundles),
            "demos_frozen_in_mipro": demos_frozen,
            "best_instruction": program_instruction(best),
            "minibatch": use_minibatch,
        }
        return bundles
