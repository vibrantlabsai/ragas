import logging
import typing as t
from dataclasses import dataclass, field

from langchain_core.callbacks import Callbacks

from ragas.dataset_schema import SingleMetricAnnotation
from ragas.losses import Loss
from ragas.optimizers.base import Optimizer
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)


@dataclass
class DSPyOptimizer(Optimizer):
    """
    Advanced prompt optimizer using DSPy's MIPROv2.

    MIPROv2 performs sophisticated prompt optimization by combining:
    - Instruction optimization (prompt engineering)
    - Demonstration optimization (few-shot examples)
    - Combined search over both spaces

    Requires: pip install dspy-ai or uv add ragas[dspy]

    Parameters
    ----------
    num_candidates : int
        Number of prompt variants to try during optimization.
    max_bootstrapped_demos : int
        Maximum number of auto-generated examples to use.
    max_labeled_demos : int
        Maximum number of human-annotated examples to use.
    init_temperature : float
        Exploration temperature for optimization.
    """

    num_candidates: int = 10
    max_bootstrapped_demos: int = 5
    max_labeled_demos: int = 5
    init_temperature: float = 1.0
    _dspy: t.Optional[t.Any] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        try:
            import dspy

            self._dspy = dspy
        except ImportError as e:
            raise ImportError(
                "DSPy optimizer requires dspy-ai. Install with:\n"
                "  uv add 'ragas[dspy]'  # or: pip install 'ragas[dspy]'\n"
            ) from e

    def optimize(
        self,
        dataset: SingleMetricAnnotation,
        loss: Loss,
        config: t.Dict[t.Any, t.Any],
        run_config: t.Optional[RunConfig] = None,
        batch_size: t.Optional[int] = None,
        callbacks: t.Optional[Callbacks] = None,
        with_debugging_logs: bool = False,
        raise_exceptions: bool = True,
    ) -> t.Dict[str, str]:
        """
        Optimize metric prompts using DSPy MIPROv2.

        Steps:

        1. Convert Ragas PydanticPrompt to DSPy Signature
        2. Create DSPy Module with signature
        3. Convert dataset to DSPy Examples
        4. Run MIPROv2 optimization
        5. Extract optimized prompts
        6. Convert back to Ragas format

        Parameters
        ----------
        dataset : SingleMetricAnnotation
            Annotated dataset with ground truth scores.
        loss : Loss
            Loss function to optimize.
        config : Dict[Any, Any]
            Additional configuration parameters.
        run_config : RunConfig, optional
            Runtime configuration.
        batch_size : int, optional
            Batch size for evaluation.
        callbacks : Callbacks, optional
            Langchain callbacks for tracking.
        with_debugging_logs : bool
            Enable debug logging.
        raise_exceptions : bool
            Whether to raise exceptions during optimization.

        Returns
        -------
        Dict[str, str]
            Optimized prompts for each prompt name.
        """
        if self.metric is None:
            raise ValueError("No metric provided for optimization.")

        if self.llm is None:
            raise ValueError("No llm provided for optimization.")

        if self._dspy is None:
            raise RuntimeError("DSPy module not loaded.")

        logger.info(f"Starting DSPy optimization for metric: {self.metric.name}")

        from ragas.optimizers.dspy_adapter import (
            create_dspy_metric,
            pydantic_prompt_to_dspy_signature,
            ragas_dataset_to_dspy_examples,
            setup_dspy_llm,
        )

        setup_dspy_llm(self._dspy, self.llm)

        prompts = self.metric.get_prompts()
        optimized_prompts = {}

        for prompt_name, prompt in prompts.items():
            logger.info(f"Optimizing prompt: {prompt_name}")

            signature = pydantic_prompt_to_dspy_signature(prompt)
            module = self._dspy.Predict(signature)
            examples = ragas_dataset_to_dspy_examples(dataset, prompt_name)

            teleprompter = self._dspy.MIPROv2(
                num_candidates=self.num_candidates,
                max_bootstrapped_demos=self.max_bootstrapped_demos,
                max_labeled_demos=self.max_labeled_demos,
                init_temperature=self.init_temperature,
            )

            metric_fn = create_dspy_metric(loss, dataset.name)

            optimized = teleprompter.compile(
                module,
                trainset=examples,
                metric=metric_fn,
            )

            optimized_instruction = self._extract_instruction(optimized)
            optimized_prompts[prompt_name] = optimized_instruction

            logger.info(
                f"Optimized prompt for {prompt_name}: {optimized_instruction[:100]}..."
            )

        return optimized_prompts

    def _extract_instruction(self, optimized_module: t.Any) -> str:
        """
        Extract the optimized instruction from DSPy module.

        Parameters
        ----------
        optimized_module : Any
            The optimized DSPy module from MIPROv2.

        Returns
        -------
        str
            The optimized instruction string.
        """
        if hasattr(optimized_module, "signature"):
            sig = optimized_module.signature
            if hasattr(sig, "instructions"):
                return sig.instructions
            elif hasattr(sig, "__doc__"):
                return sig.__doc__ or ""

        if hasattr(optimized_module, "extended_signature"):
            return str(optimized_module.extended_signature)

        return ""
