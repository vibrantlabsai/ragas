from __future__ import annotations

import asyncio
import logging
import typing as t
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from pydantic import ValidationError
from tqdm import tqdm

from ragas._analytics import EvaluationEvent, _analytics_batcher
from ragas.async_utils import apply_nest_asyncio, run
from ragas.callbacks import ChainType, new_group
from ragas.dataset_schema import MetricAnnotation, MultiTurnSample, SingleTurnSample
from ragas.llms import BaseRagasLLM
from ragas.losses import BinaryMetricLoss, MSELoss
from ragas.metrics.validators import AllowedValuesType
from ragas.prompt import FewShotPydanticPrompt, PromptMixin
from ragas.run_config import RunConfig
from ragas.utils import camel_to_snake, get_metric_language

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from pydantic import BaseModel

    from ragas.config import DemonstrationConfig, InstructionConfig
    from ragas.dataset import Dataset
    from ragas.embeddings import BaseRagasEmbedding, BaseRagasEmbeddings
    from ragas.metrics.result import MetricResult
    from ragas.prompt.simple_prompt import Prompt

    # Type alias for embedding model parameters (union of old and new embedding interfaces)
    EmbeddingModelType = t.Union[BaseRagasEmbedding, BaseRagasEmbeddings]

logger = logging.getLogger(__name__)


VALID_COLUMNS = [
    "user_input",
    "retrieved_contexts",
    "reference_contexts",
    "response",
    "reference",
    "rubric",
]


class MetricType(Enum):
    """
    Enumeration of metric types in Ragas.

    Attributes
    ----------
    SINGLE_TURN : str
        Represents a single-turn metric type.
    MULTI_TURN : str
        Represents a multi-turn metric type.
    """

    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"


class MetricOutputType(Enum):
    BINARY = "binary"
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    RANKING = "ranking"


@dataclass
class Metric(ABC):
    """
    Abstract base class for metrics in Ragas.

    Attributes
    ----------
    name : str
        The name of the metric.
    required_columns : Dict[str, Set[str]]
        A dictionary mapping metric type names to sets of required column names. This is
        a property and raises `ValueError` if columns are not in `VALID_COLUMNS`.
    """

    _required_columns: t.Dict[MetricType, t.Set[str]] = field(default_factory=dict)
    name: str = field(default="", repr=True)

    def __post_init__(self):
        if self.name == "":
            self.name = camel_to_snake(self.__class__.__name__)

    @property
    def required_columns(self) -> t.Dict[str, t.Set[str]]:
        required_columns = {}
        # ignore any value that contains marker suffixes like ":optional" or ":ignored"
        for k, v in self._required_columns.items():
            required_columns[k.name] = {
                column
                for column in v
                if not column.endswith(":optional") and not column.endswith(":ignored")
            }
        return required_columns

    @required_columns.setter
    def required_columns(self, required_columns: t.Dict[MetricType, t.Set[str]]):
        rc = {}
        for metric_type, columns in required_columns.items():
            for column in columns:
                base_column = column.split(":")[0]
                if base_column not in VALID_COLUMNS:
                    raise ValueError(
                        f"Invalid column '{column}'. Base column '{base_column}' must be one of {VALID_COLUMNS}"
                    )
            rc[metric_type] = columns
        self._required_columns = rc

    def get_required_columns(
        self, with_optional: bool = False
    ) -> t.Dict[str, t.Set[str]]:
        if with_optional:
            # get all the required columns with optional columns, remove the optional suffix
            required_columns = {}
            for k, v in self._required_columns.items():
                # if any column ends with ":optional", add it to the required columns after removing the suffix
                # if any column ends with ":ignored", do not include it
                required_columns[k.name] = set()
                for column in v:
                    if column.endswith(":ignored"):
                        continue
                    if column.endswith(":optional"):
                        required_columns[k.name].add(column[: -len(":optional")])
                    else:
                        required_columns[k.name].add(column)
            return required_columns
        else:
            return self.required_columns

    @abstractmethod
    def init(self, run_config: RunConfig) -> None:
        """
        Initialize the metric with the given run configuration.

        Parameters
        ----------
        run_config : RunConfig
            Configuration for the metric run including timeouts and other settings.
        """
        ...


@dataclass
class MetricWithLLM(Metric, PromptMixin):
    """
    A metric class that uses a language model for evaluation.

    Attributes
    ----------
    llm : Optional[BaseRagasLLM]
        The language model used for the metric. Both BaseRagasLLM and InstructorBaseRagasLLM
        are accepted at runtime via duck typing (both have compatible methods).
    """

    llm: t.Optional[BaseRagasLLM] = None
    output_type: t.Optional[MetricOutputType] = None

    def init(self, run_config: RunConfig) -> None:
        """
        Initialize the metric with run configuration and validate LLM is present.

        Parameters
        ----------
        run_config : RunConfig
            Configuration for the metric run.

        Raises
        ------
        ValueError
            If no LLM is provided to the metric.
        """
        if self.llm is None:
            raise ValueError(
                f"Metric '{self.name}' has no valid LLM provided (self.llm is None). Please instantiate the metric with an LLM to run."
            )
        # Only BaseRagasLLM has set_run_config method, not InstructorBaseRagasLLM
        if isinstance(self.llm, BaseRagasLLM):
            self.llm.set_run_config(run_config)

    def _optimize_instruction(
        self,
        instruction_config: InstructionConfig,
        dataset: MetricAnnotation,
        callbacks: Callbacks,
        run_config: RunConfig,
        batch_size: t.Optional[int],
        with_debugging_logs: bool,
        raise_exceptions: bool,
    ):
        if self.llm is None:
            raise ValueError(
                f"Metric '{self.name}' has no valid LLM provided (self.llm is None). Please initantiate a the metric with an LLM to run."  # noqa
            )
        optimizer = instruction_config.optimizer
        if optimizer.llm is None:
            optimizer.llm = instruction_config.llm

        # figure out the loss function
        if instruction_config.loss is None:
            if self.output_type is None:
                raise ValueError(
                    f"Output type for metric '{self.name}' is not defined. Please set the output type in the metric or in the instruction config."
                )
            if self.output_type.name == MetricOutputType.BINARY.name:
                loss_fun = BinaryMetricLoss()
            elif (
                self.output_type.name == MetricOutputType.CONTINUOUS.name
                or self.output_type.name == MetricOutputType.DISCRETE.name
            ):
                loss_fun = MSELoss()
            else:
                raise NotImplementedError(
                    f"Output type '{self.output_type.name}' not implemented"
                )
        else:
            loss_fun = instruction_config.loss

        # Optimize the prompts
        optimizer.metric = self
        optimizer_config = instruction_config.optimizer_config or {}
        optimized_prompts = optimizer.optimize(
            dataset[self.name],
            loss_fun,
            optimizer_config,
            callbacks=callbacks,
            run_config=run_config,
            batch_size=batch_size,
            with_debugging_logs=with_debugging_logs,
            raise_exceptions=raise_exceptions,
        )

        # replace the instruction in the metric with the optimized instruction
        prompts = self.get_prompts()
        for key, val in optimized_prompts.items():
            prompts[key].instruction = val
        self.set_prompts(**prompts)

    def _optimize_demonstration(
        self, demonstration_config: DemonstrationConfig, dataset: MetricAnnotation
    ):
        # get the prompt annotations for this metric
        prompt_annotations = dataset[self.name].get_prompt_annotations()
        prompts = self.get_prompts()
        for prompt_name, prompt_annotation_list in prompt_annotations.items():
            # create a new FewShotPydanticPrompt with these annotations
            if prompt_name not in prompts:
                raise ValueError(
                    f"Prompt '{prompt_name}' not found in metric '{self.name}'. Please check the prompt names in the annotation dataset."
                )
            pydantic_prompt = prompts[prompt_name]
            input_model, output_model = (
                pydantic_prompt.input_model,
                pydantic_prompt.output_model,
            )
            # convert annotations into examples
            input_examples, output_examples = [], []
            for i, prompt_annotation in enumerate(prompt_annotation_list):
                try:
                    input_examples.append(
                        input_model.model_validate(prompt_annotation.prompt_input)
                    )
                    # use the edited output if it is provided
                    if prompt_annotation.edited_output is not None:
                        output_examples.append(
                            output_model.model_validate(prompt_annotation.edited_output)
                        )
                    else:
                        output_examples.append(
                            output_model.model_validate(prompt_annotation.prompt_output)
                        )
                except ValidationError as e:
                    logger.warning(
                        f"Skipping prompt '{prompt_name}' example {i} because of validation error: {e}"
                    )
                    continue
            embedding_model = demonstration_config.embedding
            few_shot_prompt = FewShotPydanticPrompt.from_pydantic_prompt(
                pydantic_prompt=pydantic_prompt,
                embeddings=embedding_model,
            )

            # add the top k examples to the few shot prompt
            few_shot_prompt.top_k_for_examples = demonstration_config.top_k
            few_shot_prompt.threshold_for_examples = demonstration_config.threshold

            # add examples to the few shot prompt
            for input_example, output_example in tqdm(
                zip(input_examples, output_examples),
                total=len(input_examples),
                desc=f"Few-shot examples [{prompt_name}]",
            ):
                few_shot_prompt.add_example(input_example, output_example)
            prompts[prompt_name] = few_shot_prompt
        self.set_prompts(**prompts)

    def train(
        self,
        path: str,
        demonstration_config: t.Optional[DemonstrationConfig] = None,
        instruction_config: t.Optional[InstructionConfig] = None,
        callbacks: t.Optional[Callbacks] = None,
        run_config: t.Optional[RunConfig] = None,
        batch_size: t.Optional[int] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
    ) -> None:
        """
        Train the metric using local JSON data

        Parameters
        ----------
        path : str
            Path to local JSON training data file
        demonstration_config : DemonstrationConfig, optional
            Configuration for demonstration optimization
        instruction_config : InstructionConfig, optional
            Configuration for instruction optimization
        callbacks : Callbacks, optional
            List of callback functions
        run_config : RunConfig, optional
            Run configuration
        batch_size : int, optional
            Batch size for training
        with_debugging_logs : bool, default=False
            Enable debugging logs
        raise_exceptions : bool, default=True
            Whether to raise exceptions during training

        Raises
        ------
        ValueError
            If path is not provided or not a JSON file
        """
        # Validate input parameters
        if not path:
            raise ValueError("Path to training data file must be provided")

        if not path.endswith(".json"):
            raise ValueError("Train data must be in json format")

        run_config = run_config or RunConfig()
        callbacks = callbacks or []

        # Load the dataset from JSON file
        dataset = MetricAnnotation.from_json(path, metric_name=self.name)

        # only optimize the instruction if instruction_config is provided
        if instruction_config is not None:
            self._optimize_instruction(
                instruction_config=instruction_config,
                dataset=dataset,
                callbacks=callbacks,
                run_config=run_config,
                batch_size=batch_size,
                with_debugging_logs=with_debugging_logs,
                raise_exceptions=raise_exceptions,
            )

        # if demonstration_config is provided, optimize the demonstrations
        if demonstration_config is not None:
            self._optimize_demonstration(
                demonstration_config=demonstration_config,
                dataset=dataset,
            )


@dataclass
class MetricWithEmbeddings(Metric):
    embeddings: t.Optional[t.Union[BaseRagasEmbeddings, BaseRagasEmbedding]] = None

    def init(self, run_config: RunConfig):
        if self.embeddings is None:
            raise ValueError(
                f"Metric '{self.name}' has no valid embeddings provided (self.embeddings is None). Please initantiate a the metric with an embeddings to run."  # noqa
            )
        # Only legacy BaseRagasEmbeddings has set_run_config method
        if hasattr(self.embeddings, "set_run_config"):
            self.embeddings.set_run_config(run_config)  # type: ignore[attr-defined]


class SingleTurnMetric(Metric):
    """
    A metric class for evaluating single-turn interactions.

    This class provides methods to score single-turn samples, both synchronously and asynchronously.
    """

    def _only_required_columns_single_turn(
        self, sample: SingleTurnSample
    ) -> SingleTurnSample:
        """
        Simplify the sample to only include the required columns.
        """
        required_columns = self.get_required_columns(with_optional=True).get(
            MetricType.SINGLE_TURN.name, set()
        )
        if not required_columns:
            return sample
        return SingleTurnSample(**sample.model_dump(include=required_columns))

    def single_turn_score(
        self,
        sample: SingleTurnSample,
        callbacks: Callbacks = None,
    ) -> float:
        """
        Synchronously score a single-turn sample.

        May raise ImportError if nest_asyncio is not installed in a Jupyter-like environment.
        """
        callbacks = callbacks or []
        # only get the required columns
        sample = self._only_required_columns_single_turn(sample)
        rm, group_cm = new_group(
            self.name,
            inputs=sample.to_dict(),
            callbacks=callbacks,
            metadata={"type": ChainType.METRIC},
        )

        async def _async_wrapper():
            try:
                result = await self._single_turn_ascore(
                    sample=sample, callbacks=group_cm
                )
            except Exception as e:
                if not group_cm.ended:
                    rm.on_chain_error(e)
                raise e
            else:
                if not group_cm.ended:
                    rm.on_chain_end({"output": result})
                return result

        apply_nest_asyncio()
        score = run(_async_wrapper)

        # track the evaluation event
        _analytics_batcher.add_evaluation(
            EvaluationEvent(
                metrics=[self.name],
                num_rows=1,
                evaluation_type=MetricType.SINGLE_TURN.name,
                language=get_metric_language(self),
            )
        )
        return score

    async def single_turn_ascore(
        self,
        sample: SingleTurnSample,
        callbacks: Callbacks = None,
        timeout: t.Optional[float] = None,
    ) -> float:
        """
        Asynchronously score a single-turn sample with an optional timeout.

        May raise asyncio.TimeoutError if the scoring process exceeds the specified timeout.
        """
        callbacks = callbacks or []
        # only get the required columns
        sample = self._only_required_columns_single_turn(sample)
        rm, group_cm = new_group(
            self.name,
            inputs=sample.to_dict(),
            callbacks=callbacks,
            metadata={"type": ChainType.METRIC},
        )
        try:
            score = await asyncio.wait_for(
                self._single_turn_ascore(sample=sample, callbacks=group_cm),
                timeout=timeout,
            )
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})

        # track the evaluation event
        _analytics_batcher.add_evaluation(
            EvaluationEvent(
                metrics=[self.name],
                num_rows=1,
                evaluation_type=MetricType.SINGLE_TURN.name,
                language=get_metric_language(self),
            )
        )
        return score

    @abstractmethod
    async def _single_turn_ascore(
        self,
        sample: SingleTurnSample,
        callbacks: Callbacks,
    ) -> float:
        """
        Abstract method to be implemented by subclasses for actual scoring logic.
        """
        ...


class MultiTurnMetric(Metric):
    """
    A metric class for evaluating multi-turn conversations.

    This class extends the base Metric class to provide functionality
    for scoring multi-turn conversation samples.
    """

    def _only_required_columns_multi_turn(
        self, sample: MultiTurnSample
    ) -> MultiTurnSample:
        """
        Simplify the sample to only include the required columns.
        """
        required_columns = self.get_required_columns(with_optional=True).get(
            MetricType.MULTI_TURN.name, set()
        )
        if not required_columns:
            return sample
        return MultiTurnSample(**sample.model_dump(include=required_columns))

    def multi_turn_score(
        self,
        sample: MultiTurnSample,
        callbacks: Callbacks = None,
    ) -> float:
        """
        Score a multi-turn conversation sample synchronously.

        May raise ImportError if nest_asyncio is not installed in Jupyter-like environments.
        """
        callbacks = callbacks or []
        sample = self._only_required_columns_multi_turn(sample)
        rm, group_cm = new_group(
            self.name,
            inputs=sample.to_dict(),
            callbacks=callbacks,
            metadata={"type": ChainType.METRIC},
        )

        async def _async_wrapper():
            try:
                result = await self._multi_turn_ascore(
                    sample=sample, callbacks=group_cm
                )
            except Exception as e:
                if not group_cm.ended:
                    rm.on_chain_error(e)
                raise e
            else:
                if not group_cm.ended:
                    rm.on_chain_end({"output": result})
                return result

        apply_nest_asyncio()
        score = run(_async_wrapper)

        # track the evaluation event
        _analytics_batcher.add_evaluation(
            EvaluationEvent(
                metrics=[self.name],
                num_rows=1,
                evaluation_type=MetricType.SINGLE_TURN.name,
                language=get_metric_language(self),
            )
        )
        return score

    async def multi_turn_ascore(
        self,
        sample: MultiTurnSample,
        callbacks: Callbacks = None,
        timeout: t.Optional[float] = None,
    ) -> float:
        """
        Score a multi-turn conversation sample asynchronously.

        May raise asyncio.TimeoutError if the scoring process exceeds the specified timeout.
        """
        callbacks = callbacks or []
        sample = self._only_required_columns_multi_turn(sample)

        rm, group_cm = new_group(
            self.name,
            inputs=sample.to_dict(),
            callbacks=callbacks,
            metadata={"type": ChainType.METRIC},
        )
        try:
            score = await asyncio.wait_for(
                self._multi_turn_ascore(sample=sample, callbacks=group_cm),
                timeout=timeout,
            )
        except Exception as e:
            if not group_cm.ended:
                rm.on_chain_error(e)
            raise e
        else:
            if not group_cm.ended:
                rm.on_chain_end({"output": score})

        # track the evaluation event
        _analytics_batcher.add_evaluation(
            EvaluationEvent(
                metrics=[self.name],
                num_rows=1,
                evaluation_type=MetricType.SINGLE_TURN.name,
                language=get_metric_language(self),
            )
        )

        return score

    @abstractmethod
    async def _multi_turn_ascore(
        self,
        sample: MultiTurnSample,
        callbacks: Callbacks,
    ) -> float:
        """
        Abstract method to be implemented by subclasses for actual multi-turn scoring logic.
        """
        ...


class Ensember:
    """
    Combine multiple llm outputs for same input (n>1) to a single output
    """

    def from_discrete(
        self, inputs: list[list[t.Dict]], attribute: str
    ) -> t.List[t.Dict]:
        """
        Simple majority voting for binary values, ie [0,0,1] -> 0
        inputs: list of list of dicts each containing verdict for a single input
        """

        if not isinstance(inputs, list):
            inputs = [inputs]

        if not all(len(item) == len(inputs[0]) for item in inputs):
            logger.warning("All inputs must have the same length")
            return inputs[0]

        if not all(attribute in item for input in inputs for item in input):
            logger.warning(f"All inputs must have {attribute} attribute")
            return inputs[0]

        if len(inputs) == 1:
            return inputs[0]

        verdict_agg = []
        for i in range(len(inputs[0])):
            item = inputs[0][i]
            verdicts = [inputs[k][i][attribute] for k in range(len(inputs))]
            verdict_counts = dict(Counter(verdicts).most_common())
            item[attribute] = list(verdict_counts.keys())[0]
            verdict_agg.append(item)

        return verdict_agg


@t.runtime_checkable
class ModeMetric(t.Protocol):
    name: str
    mode: str


ensembler = Ensember()


@dataclass
class SimpleBaseMetric(ABC):
    """
    Base class for simple metrics that return MetricResult objects.

    This class provides the foundation for metrics that evaluate inputs
    and return structured MetricResult objects containing scores and reasoning.

    Attributes
    ----------
    name : str
        The name of the metric.
    allowed_values : AllowedValuesType
        Allowed values for the metric output. Can be a list of strings for
        discrete metrics, a tuple of floats for numeric metrics, or an integer
        for ranking metrics.

    Examples
    --------
    >>> from ragas.metrics import discrete_metric
    >>>
    >>> @discrete_metric(name="sentiment", allowed_values=["positive", "negative"])
    >>> def sentiment_metric(user_input: str, response: str) -> str:
    ...     return "positive" if "good" in response else "negative"
    >>>
    >>> result = sentiment_metric(user_input="How are you?", response="I'm good!")
    >>> print(result.value)  # "positive"
    """

    name: str
    allowed_values: AllowedValuesType = field(default_factory=lambda: ["pass", "fail"])

    @abstractmethod
    def score(self, **kwargs) -> "MetricResult":
        """
        Synchronously calculate the metric score.

        Parameters
        ----------
        **kwargs : dict
            Input parameters required by the specific metric implementation.

        Returns
        -------
        MetricResult
            The evaluation result containing the score and reasoning.
        """
        pass

    @abstractmethod
    async def ascore(self, **kwargs) -> "MetricResult":
        """
        Asynchronously calculate the metric score.

        Parameters
        ----------
        **kwargs : dict
            Input parameters required by the specific metric implementation.

        Returns
        -------
        MetricResult
            The evaluation result containing the score and reasoning.
        """
        pass

    def batch_score(
        self,
        inputs: t.List[t.Dict[str, t.Any]],
    ) -> t.List["MetricResult"]:
        """
        Synchronously calculate scores for a batch of inputs.

        Parameters
        ----------
        inputs : List[Dict[str, Any]]
            List of input dictionaries, each containing parameters for the metric.

        Returns
        -------
        List[MetricResult]
            List of evaluation results, one for each input.
        """
        return [self.score(**input_dict) for input_dict in inputs]

    async def abatch_score(
        self,
        inputs: t.List[t.Dict[str, t.Any]],
    ) -> t.List["MetricResult"]:
        """
        Asynchronously calculate scores for a batch of inputs in parallel.

        Parameters
        ----------
        inputs : List[Dict[str, Any]]
            List of input dictionaries, each containing parameters for the metric.

        Returns
        -------
        List[MetricResult]
            List of evaluation results, one for each input.
        """
        async_tasks = []
        for input_dict in inputs:
            # Process input asynchronously
            async_tasks.append(self.ascore(**input_dict))

        # Run all tasks concurrently and return results
        return await asyncio.gather(*async_tasks)


def create_auto_response_model(name: str, **fields) -> t.Type["BaseModel"]:
    """
    Create a response model and mark it as auto-generated by Ragas.

    This function creates a Pydantic model using create_model and marks it
    with a special attribute to indicate it was auto-generated. This allows
    the save() method to distinguish between auto-generated models (which
    are recreated on load) and custom user models.

    Parameters
    ----------
    name : str
        Name for the model class
    **fields
        Field definitions in create_model format.
        Each field is specified as: field_name=(type, default_or_field_info)

    Returns
    -------
    Type[BaseModel]
        Pydantic model class marked as auto-generated

    Examples
    --------
    >>> from pydantic import Field
    >>> # Simple model with required fields
    >>> ResponseModel = create_auto_response_model(
    ...     "ResponseModel",
    ...     value=(str, ...),
    ...     reason=(str, ...)
    ... )
    >>>
    >>> # Model with Field validators and descriptions
    >>> ResponseModel = create_auto_response_model(
    ...     "ResponseModel",
    ...     value=(str, Field(..., description="The predicted value")),
    ...     reason=(str, Field(..., description="Reasoning for the prediction"))
    ... )
    """
    from pydantic import create_model

    model = create_model(name, **fields)
    setattr(model, "__ragas_auto_generated__", True)  # type: ignore[attr-defined]
    return model


@dataclass(repr=False)
class SimpleLLMMetric(SimpleBaseMetric):
    """LLM-based metric that uses prompts to generate structured responses."""

    prompt: t.Optional[t.Union[str, "Prompt"]] = None
    _response_model: t.Type["BaseModel"] = field(init=False)

    def __post_init__(self):
        if isinstance(self.prompt, str):
            from ragas.prompt.simple_prompt import Prompt

            self.prompt = Prompt(self.prompt)

    def get_variables(self) -> t.List[str]:
        if isinstance(self.prompt, (type(None), str)):
            fstr = self.prompt
        else:
            fstr = self.prompt.instruction
        if fstr is None:
            return []
        import string

        vars = [
            field_name
            for _, field_name, _, _ in string.Formatter().parse(fstr)
            if field_name
        ]
        return vars

    def score(self, **kwargs) -> "MetricResult":
        from ragas.metrics.result import MetricResult

        llm = kwargs.pop("llm")  # Extract llm from kwargs for compatibility
        traces = {}
        traces["input"] = kwargs

        # get prompt
        if not self.prompt:
            raise Exception("prompt not passed")
        prompt_input = self.prompt.format(**kwargs)

        response = llm.generate(prompt_input, response_model=self._response_model)
        traces["output"] = response.model_dump()
        result = MetricResult(**response.model_dump())
        result.traces = traces
        return result

    async def ascore(self, **kwargs) -> "MetricResult":
        from ragas.metrics.result import MetricResult

        llm = kwargs.pop("llm")  # Extract llm from kwargs for compatibility
        traces = {}

        # get prompt
        if not self.prompt:
            raise Exception("prompt not passed")
        prompt_input = self.prompt.format(**kwargs)

        traces["input"] = prompt_input
        response = await llm.agenerate(
            prompt_input,
            response_model=self._response_model,
        )
        traces["output"] = response.model_dump()
        result = MetricResult(**response.model_dump())  # Fixed missing parentheses
        result.traces = traces
        return result

    def batch_score(
        self, inputs: t.List[t.Dict[str, t.Any]], **kwargs
    ) -> t.List["MetricResult"]:
        # Override base method to maintain compatibility
        llm = kwargs.get("llm") or inputs[0].get("llm") if inputs else None
        if llm:
            # Add llm to each input
            inputs_with_llm = [{**input_dict, "llm": llm} for input_dict in inputs]
            return super().batch_score(inputs_with_llm)
        return super().batch_score(inputs)

    async def abatch_score(
        self, inputs: t.List[t.Dict[str, t.Any]], **kwargs
    ) -> t.List["MetricResult"]:
        # Override base method to maintain compatibility
        llm = kwargs.get("llm") or inputs[0].get("llm") if inputs else None
        if llm:
            # Add llm to each input
            inputs_with_llm = [{**input_dict, "llm": llm} for input_dict in inputs]
            return await super().abatch_score(inputs_with_llm)
        return await super().abatch_score(inputs)

    def save(self, path: t.Optional[str] = None) -> None:
        """
        Save the metric configuration to a JSON file.

        Parameters:
        -----------
        path : str, optional
            File path to save to. If not provided, saves to "./{metric.name}.json"
            Use .gz extension for compression.

        Note:
        -----
        If the metric has a response_model, its schema will be saved for reference
        but the model itself cannot be serialized. You'll need to provide it when loading.

        Examples:
        ---------
        All these work:
        >>> metric.save()                      # → ./response_quality.json
        >>> metric.save("custom.json")         # → ./custom.json
        >>> metric.save("/path/to/metrics/")   # → /path/to/metrics/response_quality.json
        >>> metric.save("no_extension")        # → ./no_extension.json
        >>> metric.save("compressed.json.gz")  # → ./compressed.json.gz (compressed)
        """
        import gzip
        import json
        import warnings
        from pathlib import Path

        # Handle default path
        if path is None:
            # Default to current directory with metric name as filename
            file_path = Path(f"./{self.name}.json")
        else:
            file_path = Path(path)

            # If path is a directory, append the metric name as filename
            if file_path.is_dir():
                file_path = file_path / f"{self.name}.json"
            # If path has no extension, add .json
            elif not file_path.suffix:
                file_path = file_path.with_suffix(".json")

        # Collect warning messages for data loss
        warning_messages = []

        if hasattr(self, "_response_model") and self._response_model:
            # Only warn for custom response models, not auto-generated ones
            if not getattr(self._response_model, "__ragas_auto_generated__", False):
                warning_messages.append(
                    "- Custom response_model will be lost (set it manually after loading)"
                )

        # Serialize the prompt (may add embedding_model warning)
        prompt_data = self._serialize_prompt(warning_messages)

        # Determine the metric type
        metric_type = self.__class__.__name__

        # Get metric-specific config
        config = self._get_metric_config()

        # Emit consolidated warning if there's data loss
        if warning_messages:
            warnings.warn(
                "Some metric components cannot be saved and will be lost:\n"
                + "\n".join(warning_messages)
                + "\n\nYou'll need to provide these when loading the metric."
            )

        data = {
            "format_version": "1.0",
            "metric_type": metric_type,
            "name": self.name,
            "prompt": prompt_data,
            "config": config,
            "response_model_info": self._serialize_response_model_info(),
        }
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "wt", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
        except (OSError, IOError) as e:
            raise ValueError(f"Cannot save metric to {file_path}: {e}")

    def _serialize_prompt(self, warning_messages: t.List[str]) -> t.Dict[str, t.Any]:
        """Serialize the prompt for storage."""
        from ragas.prompt.dynamic_few_shot import DynamicFewShotPrompt
        from ragas.prompt.simple_prompt import Prompt

        if isinstance(self.prompt, str):
            return {"type": "string", "instruction": self.prompt}
        elif isinstance(self.prompt, DynamicFewShotPrompt):
            if self.prompt.example_store.embedding_model:
                warning_messages.append(
                    "- embedding_model will be lost (provide it when loading: load(path, embedding_model=YourModel))"
                )

            return {
                "type": "DynamicFewShotPrompt",
                "instruction": self.prompt.instruction,
                "examples": [
                    {"input": inp, "output": out}
                    for inp, out in self.prompt.example_store._examples
                ],
                "max_similar_examples": self.prompt.max_similar_examples,
                "similarity_threshold": self.prompt.similarity_threshold,
            }
        elif isinstance(self.prompt, Prompt):
            return {
                "type": "Prompt",
                "instruction": self.prompt.instruction,
                "examples": [
                    {"input": inp, "output": out} for inp, out in self.prompt.examples
                ],
            }
        else:
            raise ValueError(f"Unsupported prompt type: {type(self.prompt)}")

    def _get_metric_config(self) -> t.Dict[str, t.Any]:
        """Get metric-specific configuration."""
        config = {}
        # Convert tuples to lists for JSON serialization
        allowed_values = self.allowed_values
        if isinstance(allowed_values, tuple):
            allowed_values = list(allowed_values)
        config["allowed_values"] = allowed_values
        return config

    def _serialize_response_model_info(self) -> t.Optional[t.Dict]:
        """Serialize response model information for storage."""
        if not hasattr(self, "_response_model") or not self._response_model:
            return None

        return {
            "class_name": self._response_model.__name__,
            "module": self._response_model.__module__
            if hasattr(self._response_model, "__module__")
            else None,
            "schema": self._response_model.model_json_schema()
            if hasattr(self._response_model, "model_json_schema")
            else None,
            "note": "You must provide this model when loading",
        }

    @classmethod
    def _read_metric_type(cls, path: str) -> t.Dict[str, t.Any]:
        """
        Read just the metric type from a saved metric file.

        Parameters:
        -----------
        path : str
            File path to read from. Supports .gz compressed files.

        Returns:
        --------
        dict
            Dictionary containing at least the 'metric_type' field

        Raises:
        -------
        ValueError
            If file cannot be read or parsed
        """
        import gzip
        import json
        from pathlib import Path

        file_path = Path(path)

        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            return data
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Cannot read metric type from {path}: {e}")

    @classmethod
    def _validate_metric_type(cls, path: str) -> None:
        """
        Validate that the saved metric type matches the expected class.

        Parameters:
        -----------
        path : str
            File path to validate

        Raises:
        -------
        ValueError
            If metric type doesn't match expected class name
        """
        data = cls._read_metric_type(path)
        expected_type = cls.__name__
        actual_type = data.get("metric_type")

        if actual_type != expected_type:
            raise ValueError(
                f"Cannot load {actual_type} as {expected_type}. "
                f"The saved metric is of type '{actual_type}', but you are trying to load it as '{expected_type}'."
            )

    @classmethod
    def load(
        cls,
        path: str,
        response_model: t.Optional[t.Type["BaseModel"]] = None,
        embedding_model: t.Optional["EmbeddingModelType"] = None,
    ) -> "SimpleLLMMetric":
        """
        Load a metric from a JSON file.

        Parameters:
        -----------
        path : str
            File path to load from. Supports .gz compressed files.
        response_model : Optional[Type[BaseModel]]
            Pydantic model to use for response validation. Required for custom SimpleLLMMetrics.
        embedding_model : Optional[Any]
            Embedding model for DynamicFewShotPrompt. Required if the original used one.

        Returns:
        --------
        SimpleLLMMetric
            Loaded metric instance

        Raises:
        -------
        ValueError
            If file cannot be loaded, is invalid, or missing required models
        """
        import gzip
        import json
        from pathlib import Path

        file_path = Path(path)

        # Load JSON data
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Cannot load metric from {path}: {e}")

        # Validate format
        if data.get("format_version") != "1.0":
            import warnings

            warnings.warn(
                f"Loading metric with format version {data.get('format_version')}, expected 1.0"
            )

        # Reconstruct the prompt
        prompt = cls._deserialize_prompt(data["prompt"], embedding_model)

        # Get config
        config = data.get("config", {})

        # Create the metric instance
        metric = cls(name=data["name"], prompt=prompt, **config)

        # Set response model if provided
        if response_model:
            metric._response_model = response_model

        return metric

    @classmethod
    def _deserialize_prompt(
        cls,
        prompt_data: t.Dict[str, t.Any],
        embedding_model: t.Optional["EmbeddingModelType"] = None,
    ):
        """Deserialize a prompt from saved data."""
        from ragas.prompt.dynamic_few_shot import DynamicFewShotPrompt
        from ragas.prompt.simple_prompt import Prompt

        prompt_type = prompt_data.get("type")

        if prompt_type == "string":
            if "instruction" not in prompt_data:
                raise ValueError(
                    "Prompt data missing required 'instruction' field for string prompt"
                )
            return prompt_data["instruction"]
        elif prompt_type == "Prompt":
            if "instruction" not in prompt_data:
                raise ValueError(
                    "Prompt data missing required 'instruction' field for Prompt"
                )
            examples = [
                (ex["input"], ex["output"]) for ex in prompt_data.get("examples", [])
            ]
            return Prompt(instruction=prompt_data["instruction"], examples=examples)
        elif prompt_type == "DynamicFewShotPrompt":
            if "instruction" not in prompt_data:
                raise ValueError(
                    "Prompt data missing required 'instruction' field for DynamicFewShotPrompt"
                )

            if not embedding_model:
                import warnings

                warnings.warn(
                    "DynamicFewShotPrompt was saved with an embedding model but none provided. "
                    "Similarity-based example selection will not work."
                )

            # Create base prompt first
            base_prompt = Prompt(instruction=prompt_data["instruction"])

            # Create DynamicFewShotPrompt
            # Note: embedding_model can be None, the constructor handles it gracefully
            dynamic_prompt = DynamicFewShotPrompt.from_prompt(
                base_prompt,
                embedding_model,  # type: ignore[arg-type]
                max_similar_examples=prompt_data.get("max_similar_examples", 3),
                similarity_threshold=prompt_data.get("similarity_threshold", 0.7),
            )

            # Add examples
            for ex in prompt_data.get("examples", []):
                dynamic_prompt.add_example(ex["input"], ex["output"])

            return dynamic_prompt
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")

    @abstractmethod
    def get_correlation(
        self, gold_labels: t.List[str], predictions: t.List[str]
    ) -> float:
        """
        Calculate the correlation between gold scores and predicted scores.
        This is a placeholder method and should be implemented based on the specific metric.
        """
        pass

    def align_and_validate(
        self,
        dataset: "Dataset",
        embedding_model: "EmbeddingModelType",
        llm: "BaseRagasLLM",
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs: t.Dict[str, t.Any],
    ):
        """
        Args:
            dataset: experiment to align the metric with.
            embedding_model: The embedding model used for dynamic few-shot prompting.
            llm: The LLM instance to use for scoring.

        Align the metric with the specified experiments and validate it against a gold standard experiment.
        This method combines alignment and validation into a single step.
        """
        train_dataset, test_dataset = dataset.train_test_split(
            test_size=test_size, random_state=random_state
        )

        self.align(train_dataset, embedding_model, **kwargs)  # type: ignore
        return self.validate_alignment(llm, test_dataset)  # type: ignore

    def align(
        self,
        train_dataset: "Dataset",
        embedding_model: "EmbeddingModelType",
        **kwargs: t.Dict[str, t.Any],
    ):
        """
        Args:
            train_dataset: train_dataset to align the metric with.
            embedding_model: The embedding model used for dynamic few-shot prompting.

        Align the metric with the specified experiments by different optimization methods.
        """

        # get prompt
        if not self.prompt:
            raise Exception("prompt not passed")
        from ragas.prompt.simple_prompt import Prompt

        self.prompt = (
            self.prompt if isinstance(self.prompt, Prompt) else Prompt(self.prompt)
        )
        # Extract specific parameters for from_prompt method
        max_similar_examples_val = kwargs.get("max_similar_examples", 3)
        similarity_threshold_val = kwargs.get("similarity_threshold", 0.7)
        max_similar_examples = (
            int(max_similar_examples_val)
            if isinstance(max_similar_examples_val, (int, str))
            else 3
        )
        similarity_threshold = (
            float(similarity_threshold_val)
            if isinstance(similarity_threshold_val, (int, float, str))
            else 0.7
        )
        # Convert BaseRagasEmbeddings to BaseRagasEmbedding if needed
        if hasattr(embedding_model, "embed_query"):
            # For legacy BaseRagasEmbeddings, we need to wrap it
            # Create a wrapper that implements BaseRagasEmbedding interface
            class EmbeddingWrapper:
                def __init__(self, legacy_embedding):
                    self.legacy_embedding = legacy_embedding

                def embed_text(self, text: str, **kwargs) -> t.List[float]:
                    return self.legacy_embedding.embed_query(text)

                async def aembed_text(self, text: str, **kwargs) -> t.List[float]:
                    return await self.legacy_embedding.aembed_query(text)

            actual_embedding_model = EmbeddingWrapper(embedding_model)
        else:
            # Already BaseRagasEmbedding
            actual_embedding_model = embedding_model

        from ragas.prompt.dynamic_few_shot import DynamicFewShotPrompt

        self.prompt = DynamicFewShotPrompt.from_prompt(
            self.prompt,
            actual_embedding_model,  # type: ignore[arg-type]
            max_similar_examples,
            similarity_threshold,
        )
        train_dataset.reload()
        total_items = len(train_dataset)
        input_vars = self.get_variables()
        output_vars = [self.name, f"{self.name}_reason"]

        from rich.progress import Progress

        with Progress() as progress:
            task = progress.add_task("Processing examples", total=total_items)
            for row in train_dataset:
                inputs = {
                    var: train_dataset.get_row_value(row, var) for var in input_vars
                }
                inputs = {k: v for k, v in inputs.items() if v is not None}
                output = {
                    var: train_dataset.get_row_value(row, var) for var in output_vars
                }
                output = {k: v for k, v in output.items() if v is not None}

                if output:
                    self.prompt.add_example(inputs, output)
                progress.update(task, advance=1)

    def validate_alignment(
        self,
        llm: "BaseRagasLLM",
        test_dataset: "Dataset",
        mapping: t.Dict[str, str] = {},
    ):
        """
        Args:
            llm: The LLM instance to use for scoring.
            test_dataset: An Dataset instance containing the gold standard scores.
            mapping: A dictionary mapping variable names expected by metrics to their corresponding names in the gold experiment.

        Validate the alignment of the metric by comparing the scores against a gold standard experiment.
        This method computes the Cohen's Kappa score and agreement rate between the gold standard scores and
        the predicted scores from the metric.
        """

        test_dataset.reload()
        gold_scores_raw = [
            test_dataset.get_row_value(row, self.name) for row in test_dataset
        ]
        pred_scores = []
        for row in test_dataset:
            values = {
                v: (
                    test_dataset.get_row_value(row, v)
                    if v not in mapping
                    else test_dataset.get_row_value(row, mapping.get(v, v))
                )
                for v in self.get_variables()
            }
            score = self.score(llm=llm, **values)
            pred_scores.append(score.value)

        # Convert to strings for correlation calculation, filtering out None values
        gold_scores = [str(score) for score in gold_scores_raw if score is not None]
        pred_scores_str = [str(score) for score in pred_scores if score is not None]

        df = test_dataset.to_pandas()
        df[f"{self.name}_pred"] = pred_scores
        correlation = self.get_correlation(gold_scores, pred_scores_str)
        agreement_rate = sum(
            x == y for x, y in zip(gold_scores, pred_scores_str)
        ) / len(gold_scores)
        return {
            "correlation": correlation,
            "agreement_rate": agreement_rate,
            "df": df,
        }

    def __repr__(self) -> str:
        """Return a clean string representation of the metric."""
        metric_type = self.__class__.__name__

        allowed_values = self.allowed_values
        if isinstance(allowed_values, range):
            allowed_values_str = (
                f", allowed_values=({allowed_values.start}, {allowed_values.stop})"
            )
        elif isinstance(allowed_values, (list, tuple, int)):
            allowed_values_str = f", allowed_values={allowed_values}"
        else:
            allowed_values_str = f", allowed_values={repr(allowed_values)}"

        prompt_str = ""
        if self.prompt:
            instruction = (
                self.prompt
                if isinstance(self.prompt, str)
                else (
                    self.prompt.instruction
                    if hasattr(self.prompt, "instruction")
                    else str(self.prompt)
                )
            )

            if instruction:
                max_len = 80
                if len(instruction) > max_len:
                    prompt_str = f", prompt='{instruction[: max_len - 3]}...'"
                else:
                    prompt_str = f", prompt='{instruction}'"

        return f"{metric_type}(name='{self.name}'{allowed_values_str}{prompt_str})"
