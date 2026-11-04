"""The DSPy program that stands in for the RAG app's generation step."""

from __future__ import annotations

import typing as t

from ragas.optimizers.prompt_optimization.bundle import GoldenRow, PromptBundle

DSPY_INSTALL_HINT = (
    "Prompt optimization requires DSPy. Install it with:\n"
    "  uv pip install 'ragas[dspy]'   # or: pip install 'ragas[dspy]'"
)


def import_dspy() -> t.Any:
    try:
        import dspy
    except ImportError as e:
        # Covered by test_import_error_without_dspy (patched import).
        raise ImportError(DSPY_INSTALL_HINT) from e
    return dspy


def make_signature(instruction: str) -> t.Any:
    """A context + question -> answer signature carrying ``instruction``."""
    dspy = import_dspy()

    class RAGAnswerSignature(dspy.Signature):
        context: str = dspy.InputField(
            desc="Retrieved documents, numbered [1], [2], ..."
        )
        question: str = dspy.InputField(desc="The user's question")
        answer: str = dspy.OutputField(desc="The answer to the question")

    return RAGAnswerSignature.with_instructions(instruction)


def demo_example(shot: t.Any) -> t.Any:
    """A golden few-shot as a ``dspy.Example`` in the signature's field order."""
    dspy = import_dspy()
    return dspy.Example(
        context=shot.context_text, question=shot.question, answer=shot.answer
    ).with_inputs("context", "question")


def build_program(bundle: PromptBundle) -> t.Any:
    """``dspy.Predict`` with the bundle's instruction and few-shot demos."""
    dspy = import_dspy()
    program = dspy.Predict(make_signature(bundle.instruction))
    program.demos = [demo_example(s) for s in bundle.few_shots]
    return program


def program_instruction(program: t.Any) -> str:
    """The instruction text of a single-predictor program."""
    predictors = program.predictors() if hasattr(program, "predictors") else [program]
    return predictors[0].signature.instructions


def program_demo_answers(program: t.Any) -> t.List[t.Tuple[str, str, str]]:
    """(context, question, answer) of each demo, used to check demos stay frozen."""
    predictors = program.predictors() if hasattr(program, "predictors") else [program]
    return [(d.context, d.question, d.answer) for d in predictors[0].demos]


def make_adapter() -> t.Any:
    """ChatAdapter without the JSON fallback (that path needs forced tool use)."""
    dspy = import_dspy()
    return dspy.ChatAdapter(use_json_adapter_fallback=False)


class DSPyAnswerer:
    """``AnswerFn`` that answers a row by running the bundle's DSPy program."""

    def __init__(self, lm: t.Any, adapter: t.Any = None):
        self.lm = lm
        self.adapter = adapter or make_adapter()
        self._programs: t.Dict[str, t.Any] = {}

    def _program(self, bundle: PromptBundle) -> t.Any:
        key = bundle.fingerprint()
        if key not in self._programs:
            self._programs[key] = build_program(bundle)
        return self._programs[key]

    def __call__(self, bundle: PromptBundle, row: GoldenRow) -> str:
        dspy = import_dspy()
        program = self._program(bundle)
        with dspy.context(lm=self.lm, adapter=self.adapter):
            prediction = program(context=row.context_text, question=row.question)
        return str(prediction.answer)
