import hashlib
import json
import logging
import typing as t
from dataclasses import dataclass, field

from langchain_core.callbacks import Callbacks

from ragas.cache import CacheInterface
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
    auto : str, optional
        Automatic configuration level: 'light', 'medium', or 'heavy'.
        Controls the depth of optimization search.
    num_threads : int, optional
        Number of parallel threads for optimization.
    max_errors : int, optional
        Maximum errors tolerated during optimization before stopping.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Enable verbose logging during optimization.
    track_stats : bool
        Track and report optimization statistics.
    log_dir : str, optional
        Directory for saving optimization logs and progress.
    metric_threshold : float, optional
        Minimum acceptable metric value to achieve.
    cache : CacheInterface, optional
        Cache backend for storing optimization results.
    """

    num_candidates: int = 10
    max_bootstrapped_demos: int = 5
    max_labeled_demos: int = 5
    init_temperature: float = 1.0
    auto: t.Optional[t.Literal["light", "medium", "heavy"]] = "light"
    num_threads: t.Optional[int] = None
    max_errors: t.Optional[int] = None
    seed: int = 9
    verbose: bool = False
    track_stats: bool = True
    log_dir: t.Optional[str] = None
    metric_threshold: t.Optional[float] = None
    cache: t.Optional[CacheInterface] = field(default=None, repr=False)
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

        self._validate_parameters()

    def _validate_parameters(self):
        """Validate optimizer parameters."""
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")

        if self.max_bootstrapped_demos < 0:
            raise ValueError("max_bootstrapped_demos must be non-negative")

        if self.max_labeled_demos < 0:
            raise ValueError("max_labeled_demos must be non-negative")

        if self.init_temperature <= 0:
            raise ValueError("init_temperature must be positive")

        if self.auto not in ["light", "medium", "heavy", None]:
            raise ValueError("auto must be 'light', 'medium', 'heavy', or None")

        if self.num_threads is not None and self.num_threads <= 0:
            raise ValueError("num_threads must be positive if specified")

        if self.max_errors is not None and self.max_errors < 0:
            raise ValueError("max_errors must be non-negative if specified")

        if self.metric_threshold is not None and (
            self.metric_threshold < 0 or self.metric_threshold > 1
        ):
            raise ValueError("metric_threshold must be between 0 and 1")

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

        if self.cache is not None:
            cache_key = self._generate_cache_key(dataset, loss, config)
            if self.cache.has_key(cache_key):
                logger.info(
                    f"Cache hit for DSPy optimization of metric: {self.metric.name}"
                )
                return self.cache.get(cache_key)

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
                auto=self.auto,
                num_threads=self.num_threads,
                max_errors=self.max_errors,
                seed=self.seed,
                verbose=self.verbose,
                track_stats=self.track_stats,
                log_dir=self.log_dir,
                metric_threshold=self.metric_threshold,
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

        if self.cache is not None:
            cache_key = self._generate_cache_key(dataset, loss, config)
            self.cache.set(cache_key, optimized_prompts)
            logger.info("Cached optimization results")

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

    def _generate_cache_key(
        self,
        dataset: SingleMetricAnnotation,
        loss: Loss,
        config: t.Dict[t.Any, t.Any],
    ) -> str:
        """
        Generate a unique cache key for optimization results.

        Parameters
        ----------
        dataset : SingleMetricAnnotation
            Annotated dataset with ground truth scores.
        loss : Loss
            Loss function to optimize.
        config : Dict[Any, Any]
            Additional configuration parameters.

        Returns
        -------
        str
            SHA256 hash of the optimization parameters.
        """
        if self.metric is None:
            raise ValueError("Metric must be set to generate cache key")

        cache_data = {
            "metric_name": self.metric.name,
            "dataset_hash": hashlib.sha256(
                json.dumps(dataset.model_dump(), sort_keys=True).encode()
            ).hexdigest(),
            "loss_name": loss.__class__.__name__,
            "num_candidates": self.num_candidates,
            "max_bootstrapped_demos": self.max_bootstrapped_demos,
            "max_labeled_demos": self.max_labeled_demos,
            "init_temperature": self.init_temperature,
            "auto": self.auto,
            "num_threads": self.num_threads,
            "max_errors": self.max_errors,
            "seed": self.seed,
            "verbose": self.verbose,
            "track_stats": self.track_stats,
            "log_dir": self.log_dir,
            "metric_threshold": self.metric_threshold,
            "config": config,
        }

        key_string = json.dumps(cache_data, sort_keys=True, default=str)
        cache_key = hashlib.sha256(key_string.encode("utf-8")).hexdigest()
        return cache_key
