"""RAGPromptOptimizer: one optimization round (Layer 1 -> Layer 2 -> benchmark)."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import typing as t
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from ragas.optimizers.prompt_optimization.benchmark import (
    BenchmarkResult,
    benchmark_candidates,
)
from ragas.optimizers.prompt_optimization.budget import BudgetTracker
from ragas.optimizers.prompt_optimization.bundle import GoldenRow, PromptBundle
from ragas.optimizers.prompt_optimization.config import (
    FewShotSearchConfig,
    OptimizationGoal,
)
from ragas.optimizers.prompt_optimization.export import export_bundle
from ragas.optimizers.prompt_optimization.few_shot_search import (
    FewShotSearchResult,
    search_few_shots,
)
from ragas.optimizers.prompt_optimization.instruction_search import InstructionProposer
from ragas.optimizers.prompt_optimization.metrics import Scorer, build_dspy_metric
from ragas.optimizers.prompt_optimization.rag_evaluation import (
    AnswerFn,
    EvaluationReport,
    evaluate_bundle,
)

ROUND_SEED_STRIDE = 1000


def _accepts_kwarg(fn: t.Callable[..., t.Any], name: str) -> bool:
    """Whether ``fn`` takes keyword ``name`` (keeps older custom proposers working)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is p.VAR_KEYWORD for p in params)


logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Outcome of one optimization round."""

    winner: PromptBundle
    few_shot_search: FewShotSearchResult
    layer2_candidates: t.List[PromptBundle]
    benchmark: BenchmarkResult
    out_dir: t.Optional[Path] = None
    resumed_layers: t.List[str] = field(default_factory=list)

    def summary(self, target_metric: str) -> t.Dict[str, t.Any]:
        return {
            "winner_source": self.winner.source,
            "winner_fingerprint": self.winner.fingerprint(),
            "layer1": {
                "best": self.few_shot_search.best.summary(target_metric),
                "no_candidate_passed": self.few_shot_search.no_candidate_passed,
                "resolved_floors": self.few_shot_search.resolved_floors,
                "candidates": [
                    c.summary(target_metric) for c in self.few_shot_search.candidates
                ],
            },
            "layer2": [
                {"source": b.source, "instruction": b.instruction, **b.metadata}
                for b in self.layer2_candidates
            ],
            "benchmark": self.benchmark.to_dict(target_metric),
            "resumed_layers": self.resumed_layers,
        }


class RAGPromptOptimizer:
    """Improves a RAG app's prompt in two layers, then benchmarks on test.

    Args:
        answer_fn: Runs the student model on one row with a given bundle.
        scorer: Ragas metrics used for every split.
        goal: Target metric and user floors.
        rows: All golden rows (needed to look up references by id).
        proposer: Layer 2 instruction proposer (MIPROv2). ``None`` skips
            Layer 2 and benchmarks only the Layer 1 result.
        few_shot_config: Layer 1 settings.
        budget: Optional spend tracker; its ``stage`` is updated per layer.
        num_threads: Parallel student calls per evaluation.
    """

    def __init__(
        self,
        answer_fn: AnswerFn,
        scorer: Scorer,
        goal: OptimizationGoal,
        rows: t.Sequence[GoldenRow],
        proposer: t.Optional[InstructionProposer] = None,
        few_shot_config: t.Optional[FewShotSearchConfig] = None,
        budget: t.Optional[BudgetTracker] = None,
        num_threads: int = 8,
        export_model: str = "claude-haiku-5-5",
    ):
        self.answer_fn = answer_fn
        self.scorer = scorer
        self.goal = goal
        self.rows_by_id = {r.id: r for r in rows}
        self.proposer = proposer
        self.few_shot_config = few_shot_config or FewShotSearchConfig()
        self.budget = budget
        self.num_threads = num_threads
        self.export_model = export_model
        self._eval_cache: t.Dict[t.Tuple[str, str], EvaluationReport] = {}

    # -- helpers ---------------------------------------------------------

    def _stage(self, name: str) -> None:
        if self.budget is not None:
            self.budget.stage = name

    def evaluate(
        self, bundle: PromptBundle, rows: t.Sequence[GoldenRow], split: str
    ) -> EvaluationReport:
        key = (bundle.fingerprint(), split)
        if key not in self._eval_cache:
            self._eval_cache[key] = evaluate_bundle(
                bundle, rows, self.answer_fn, self.scorer, split, self.num_threads
            )
        return self._eval_cache[key]

    def round_few_shot_config(self, round_no: int) -> FewShotSearchConfig:
        """Layer 1 config for a round: fresh demo-set seeds every round.

        With the same seeds, a later round would redraw the same demo sets and
        (with the DSPy cache) get the same answers, so it could never find
        anything new.
        """
        cfg = self.few_shot_config
        return replace(cfg, seed=cfg.seed + ROUND_SEED_STRIDE * (round_no - 1))

    def _state_key(self, bundle: PromptBundle, round_no: int) -> str:
        payload = json.dumps(
            {
                "bundle": bundle.fingerprint(),
                "round": round_no,
                "goal": self.goal.to_dict(),
                "few_shot": asdict(self.round_few_shot_config(round_no)),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @staticmethod
    def _load_state(path: Path, key: str) -> t.Optional[t.Dict[str, t.Any]]:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if data.get("state_key") == key else None

    # -- main entry ------------------------------------------------------

    def optimize(
        self,
        bundle: PromptBundle,
        splits: t.Mapping[str, t.Sequence[GoldenRow]],
        reference: t.Optional[PromptBundle] = None,
        round_no: int = 1,
        out_dir: t.Optional[t.Union[str, Path]] = None,
    ) -> OptimizationResult:
        """Run Layer 1, Layer 2 and the test benchmark for one round.

        Args:
            bundle: The prompt to start from (the current prompt).
            splits: Rows by split; uses train, eval and test. Holdout is not
                touched here.
            reference: The prompt ``"baseline"`` floors refer to. Defaults to
                ``bundle``; the improvement loop passes the original prompt.
            round_no: Round number, used in state keys and output folders.
            out_dir: Where to save layer state, reports and the exported prompt.
        """
        reference = reference or bundle
        train, eval_rows, test = splits["train"], splits["eval"], splits["test"]
        round_dir = Path(out_dir) / f"round_{round_no}" if out_dir else None
        if round_dir:
            round_dir.mkdir(parents=True, exist_ok=True)
        key = self._state_key(bundle, round_no)
        resumed: t.List[str] = []

        # Layer 1 -----------------------------------------------------------
        self._stage(f"round{round_no}:layer1")
        reference_eval = self.evaluate(reference, eval_rows, "eval")
        layer1_path = round_dir / "layer1_state.json" if round_dir else None
        saved1 = self._load_state(layer1_path, key) if layer1_path else None
        if saved1 is not None:
            layer1 = self._restore_layer1(saved1, eval_rows, reference_eval)
            resumed.append("layer1")
        else:
            layer1 = search_few_shots(
                bundle,
                train,
                eval_rows,
                self.answer_fn,
                self.scorer,
                self.goal,
                self.round_few_shot_config(round_no),
                baseline_aggregates=reference_eval.aggregates,
                num_threads=self.num_threads,
            )
        if saved1 is None and layer1_path:
            layer1_path.write_text(
                json.dumps(
                    {
                        "state_key": key,
                        "best": layer1.best.bundle.to_dict(),
                        "candidates": [
                            c.summary(self.goal.target_metric)
                            for c in layer1.candidates
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        # Layer 2 -----------------------------------------------------------
        self._stage(f"round{round_no}:layer2")
        layer2: t.List[PromptBundle] = []
        if self.proposer is not None:
            layer2_path = round_dir / "layer2_state.json" if round_dir else None
            saved2 = self._load_state(layer2_path, key) if layer2_path else None
            if saved2 is not None:
                layer2 = [PromptBundle.from_dict(d) for d in saved2["candidates"]]
                resumed.append("layer2")
            else:
                metric = build_dspy_metric(self.goal, self.scorer, self.rows_by_id)
                extra = (
                    {"seed_offset": round_no - 1}
                    if _accepts_kwarg(self.proposer.propose, "seed_offset")
                    else {}
                )
                layer2 = self.proposer.propose(
                    layer1.best.bundle, train, eval_rows, metric, **extra
                )
                if layer2_path:
                    layer2_path.write_text(
                        json.dumps(
                            {
                                "state_key": key,
                                "candidates": [b.to_dict() for b in layer2],
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

        # Benchmark on test ---------------------------------------------------
        self._stage(f"round{round_no}:benchmark")
        reference_test = self.evaluate(reference, test, "test")
        benchmark = benchmark_candidates(
            baseline=bundle,
            candidates=[layer1.best.bundle, *layer2],
            test_rows=test,
            answer_fn=self.answer_fn,
            scorer=self.scorer,
            goal=self.goal,
            baseline_aggregates=reference_test.aggregates,
            num_threads=self.num_threads,
        )
        winner = benchmark.winner.bundle

        result = OptimizationResult(
            winner=winner,
            few_shot_search=layer1,
            layer2_candidates=layer2,
            benchmark=benchmark,
            out_dir=round_dir,
            resumed_layers=resumed,
        )
        if round_dir:
            (round_dir / "round_summary.json").write_text(
                json.dumps(result.summary(self.goal.target_metric), indent=2),
                encoding="utf-8",
            )
            export_bundle(
                winner,
                round_dir,
                model=self.export_model,
                scores={
                    "test": benchmark.winner.report.aggregates,
                    "selection_biased": True,
                },
            )
        return result

    def _restore_layer1(
        self,
        saved: t.Mapping[str, t.Any],
        eval_rows: t.Sequence[GoldenRow],
        reference_eval: EvaluationReport,
    ) -> FewShotSearchResult:
        """Rebuild the Layer 1 result from saved state without re-searching."""
        from ragas.optimizers.prompt_optimization.few_shot_search import (
            FewShotCandidate,
        )
        from ragas.optimizers.prompt_optimization.metrics import check_floors

        best_bundle = PromptBundle.from_dict(saved["best"])
        floors = self.goal.resolve_floors(reference_eval.aggregates)
        report = self.evaluate(best_bundle, eval_rows, "eval")
        best = FewShotCandidate(
            seed=best_bundle.source.split("=")[-1],
            bundle=best_bundle,
            report=report,
            floor_check=check_floors(report.aggregates, floors),
        )
        return FewShotSearchResult(best=best, candidates=[best], resolved_floors=floors)
