from __future__ import annotations

import typing as t
import warnings
from uuid import UUID

from datasets import Dataset
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.embeddings import Embeddings as LangchainEmbeddings
from langchain_core.language_models import BaseLanguageModel as LangchainLLM
from tqdm.auto import tqdm

from ragas._analytics import track_was_completed  # type: ignore
from ragas.callbacks import ChainType, RagasTracer, new_group
from ragas.dataset_schema import (
    EvaluationDataset,
    EvaluationResult,
    MultiTurnSample,
    SingleTurnSample,
)
from ragas.embeddings.base import (
    BaseRagasEmbedding,
    BaseRagasEmbeddings,
    LangchainEmbeddingsWrapper,
    _infer_embedding_provider_from_llm,
    embedding_factory,
)
from ragas.exceptions import ExceptionInRunner
from ragas.executor import Executor
from ragas.integrations.helicone import helicone_config
from ragas.llms import llm_factory
from ragas.llms.base import BaseRagasLLM, InstructorBaseRagasLLM, LangchainLLMWrapper
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._aspect_critic import AspectCritic
from ragas.metrics.base import (
    Metric,
    MetricWithEmbeddings,
    MetricWithLLM,
    ModeMetric,
    MultiTurnMetric,
    SingleTurnMetric,
)
from ragas.run_config import RunConfig
from ragas.utils import convert_v1_to_v2_dataset
from ragas.validation import (
    remap_column_names,
    validate_required_columns,
    validate_supported_metrics,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.cost import CostCallbackHandler, TokenUsageParser

RAGAS_EVALUATION_CHAIN_NAME = "ragas evaluation"


async def aevaluate(
    dataset: t.Union[Dataset, EvaluationDataset],
    metrics: t.Optional[t.Sequence[Metric]] = None,
    llm: t.Optional[BaseRagasLLM | InstructorBaseRagasLLM | LangchainLLM] = None,
    embeddings: t.Optional[
        BaseRagasEmbeddings | BaseRagasEmbedding | LangchainEmbeddings
    ] = None,
    experiment_name: t.Optional[str] = None,
    callbacks: Callbacks = None,
    run_config: t.Optional[RunConfig] = None,
    token_usage_parser: t.Optional[TokenUsageParser] = None,
    raise_exceptions: bool = False,
    column_map: t.Optional[t.Dict[str, str]] = None,
    show_progress: bool = True,
    batch_size: t.Optional[int] = None,
    _run_id: t.Optional[UUID] = None,
    _pbar: t.Optional[tqdm] = None,
    return_executor: bool = False,
) -> t.Union[EvaluationResult, Executor]:
    """
    Async version of evaluate that performs evaluation without applying nest_asyncio.

    This function is the async-first implementation that doesn't patch the event loop,
    making it safe to use in production async applications.

    Parameters are identical to evaluate() function.

    Returns
    -------
    EvaluationResult or Executor
        If return_executor is False, returns EvaluationResult object containing the scores of each metric.
        If return_executor is True, returns the Executor instance for cancellable execution.

    Examples
    --------
    ```python
    import asyncio
    from ragas import aevaluate

    async def main():
        result = await aevaluate(dataset, metrics)
        print(result)

    asyncio.run(main())
    ```
    """
    warnings.warn(
        "aevaluate() is deprecated and will be removed in a future version. "
        "Use the @experiment decorator instead. "
        "See https://docs.ragas.io/en/latest/concepts/experiment/ for more information.",
        DeprecationWarning,
        stacklevel=2,
    )

    column_map = column_map or {}
    callbacks = callbacks or []
    run_config = run_config or RunConfig()

    if helicone_config.is_enabled:
        import uuid

        helicone_config.session_name = "ragas-evaluation"
        helicone_config.session_id = str(uuid.uuid4())

    if dataset is None:
        raise ValueError("Provide dataset!")

    # Check metrics are correct type
    if not isinstance(metrics, (type(None), list)):
        raise TypeError(
            "Metrics should be provided in a list, e.g: metrics=[BleuScore()]"
        )

    if isinstance(metrics, list) and any(not isinstance(m, Metric) for m in metrics):
        raise TypeError(
            "All metrics must be initialised metric objects, e.g: metrics=[BleuScore(), AspectCritic()]"
        )

    # default metrics
    if metrics is None:
        from ragas.metrics._answer_relevance import answer_relevancy
        from ragas.metrics._context_precision import context_precision
        from ragas.metrics._context_recall import context_recall
        from ragas.metrics._faithfulness import faithfulness

        metrics = [answer_relevancy, context_precision, faithfulness, context_recall]

    if isinstance(dataset, Dataset):
        # remap column names from the dataset
        dataset = remap_column_names(dataset, column_map)
        dataset = convert_v1_to_v2_dataset(dataset)
        # validation
        dataset = EvaluationDataset.from_list(dataset.to_list())

    if isinstance(dataset, EvaluationDataset):
        validate_required_columns(dataset, metrics)
        validate_supported_metrics(dataset, metrics)

    # set the llm and embeddings
    if isinstance(llm, LangchainLLM):
        llm = LangchainLLMWrapper(llm, run_config=run_config)
    if isinstance(embeddings, LangchainEmbeddings):
        embeddings = LangchainEmbeddingsWrapper(embeddings)

    # init llms and embeddings
    binary_metrics = []
    llm_changed: t.List[int] = []
    embeddings_changed: t.List[int] = []
    answer_correctness_is_set = -1

    # loop through the metrics and perform initializations
    for i, metric in enumerate(metrics):
        # set llm and embeddings if not set
        if isinstance(metric, AspectCritic):
            binary_metrics.append(metric.name)
        if isinstance(metric, MetricWithLLM) and metric.llm is None:
            if llm is None:
                from openai import OpenAI

                client = OpenAI()
                llm = llm_factory("gpt-4o-mini", client=client)
            metric.llm = t.cast(t.Optional[BaseRagasLLM], llm)
            llm_changed.append(i)
        if isinstance(metric, MetricWithEmbeddings) and metric.embeddings is None:
            if embeddings is None:
                # Infer embedding provider from LLM if available
                inferred_provider = _infer_embedding_provider_from_llm(llm)
                # Extract client from LLM if available for modern embeddings
                embedding_client = None
                if hasattr(llm, "client"):
                    embedding_client = getattr(llm, "client")
                embeddings = embedding_factory(
                    provider=inferred_provider, client=embedding_client
                )
            metric.embeddings = embeddings
            embeddings_changed.append(i)
        if isinstance(metric, AnswerCorrectness):
            if metric.answer_similarity is None:
                answer_correctness_is_set = i

        # init all the models
        metric.init(run_config)

    executor = Executor(
        desc="Evaluating",
        keep_progress_bar=True,
        raise_exceptions=raise_exceptions,
        run_config=run_config,
        show_progress=show_progress,
        batch_size=batch_size,
        pbar=_pbar,
    )

    # Ragas Callbacks
    # init the callbacks we need for various tasks
    ragas_callbacks: t.Dict[str, BaseCallbackHandler] = {}

    # Ragas Tracer which traces the run
    tracer = RagasTracer()
    ragas_callbacks["tracer"] = tracer

    # check if cost needs to be calculated
    if token_usage_parser is not None:
        from ragas.cost import CostCallbackHandler

        cost_cb = CostCallbackHandler(token_usage_parser=token_usage_parser)
        ragas_callbacks["cost_cb"] = cost_cb

    # append all the ragas_callbacks to the callbacks
    for cb in ragas_callbacks.values():
        if isinstance(callbacks, BaseCallbackManager):
            callbacks.add_handler(cb)
        else:
            callbacks.append(cb)

    # new evaluation chain
    row_run_managers = []
    evaluation_rm, evaluation_group_cm = new_group(
        name=experiment_name or RAGAS_EVALUATION_CHAIN_NAME,
        inputs={},
        callbacks=callbacks,
        metadata={"type": ChainType.EVALUATION},
    )

    sample_type = dataset.get_sample_type()
    for i, sample in enumerate(dataset):
        row = t.cast(t.Dict[str, t.Any], sample.model_dump())
        row_rm, row_group_cm = new_group(
            name=f"row {i}",
            inputs=row,
            callbacks=evaluation_group_cm,
            metadata={"type": ChainType.ROW, "row_index": i},
        )
        row_run_managers.append((row_rm, row_group_cm))
        if sample_type == SingleTurnSample:
            _ = [
                executor.submit(
                    metric.single_turn_ascore,
                    sample,
                    row_group_cm,
                    name=f"{metric.name}-{i}",
                    timeout=run_config.timeout,
                )
                for metric in metrics
                if isinstance(metric, SingleTurnMetric)
            ]
        elif sample_type == MultiTurnSample:
            _ = [
                executor.submit(
                    metric.multi_turn_ascore,
                    sample,
                    row_group_cm,
                    name=f"{metric.name}-{i}",
                    timeout=run_config.timeout,
                )
                for metric in metrics
                if isinstance(metric, MultiTurnMetric)
            ]
        else:
            raise ValueError(f"Unsupported sample type {sample_type}")

    # Return executor for cancellable execution if requested
    if return_executor:
        return executor

    scores: t.List[t.Dict[str, t.Any]] = []
    try:
        # get the results using async method
        results = await executor.aresults()
        if results == []:
            raise ExceptionInRunner()

        # convert results to dataset_like
        for i, _ in enumerate(dataset):
            s = {}
            for j, m in enumerate(metrics):
                if isinstance(m, ModeMetric):  # type: ignore
                    key = f"{m.name}(mode={m.mode})"
                else:
                    key = m.name
                s[key] = results[len(metrics) * i + j]
            scores.append(s)
            # close the row chain
            row_rm, row_group_cm = row_run_managers[i]
            if not row_group_cm.ended:
                row_rm.on_chain_end(s)

    # run evaluation task
    except Exception as e:
        if not evaluation_group_cm.ended:
            evaluation_rm.on_chain_error(e)

        raise e
    else:
        # evalution run was successful
        # now lets process the results
        cost_cb = ragas_callbacks["cost_cb"] if "cost_cb" in ragas_callbacks else None
        result = EvaluationResult(
            scores=scores,
            dataset=dataset,
            binary_columns=binary_metrics,
            cost_cb=t.cast(
                t.Union["CostCallbackHandler", None],
                cost_cb,
            ),
            ragas_traces=tracer.traces,
            run_id=_run_id,
        )
        if not evaluation_group_cm.ended:
            evaluation_rm.on_chain_end({"scores": result.scores})
    finally:
        # reset llms and embeddings if changed
        for i in llm_changed:
            t.cast(MetricWithLLM, metrics[i]).llm = None
        for i in embeddings_changed:
            t.cast(MetricWithEmbeddings, metrics[i]).embeddings = None
        if answer_correctness_is_set != -1:
            t.cast(
                AnswerCorrectness, metrics[answer_correctness_is_set]
            ).answer_similarity = None

        # flush the analytics batcher
        from ragas._analytics import _analytics_batcher

        _analytics_batcher.flush()

    return result


@track_was_completed
def evaluate(
    dataset: t.Union[Dataset, EvaluationDataset],
    metrics: t.Optional[t.Sequence[Metric]] = None,
    llm: t.Optional[BaseRagasLLM | LangchainLLM] = None,
    embeddings: t.Optional[
        BaseRagasEmbeddings | BaseRagasEmbedding | LangchainEmbeddings
    ] = None,
    experiment_name: t.Optional[str] = None,
    callbacks: Callbacks = None,
    run_config: t.Optional[RunConfig] = None,
    token_usage_parser: t.Optional[TokenUsageParser] = None,
    raise_exceptions: bool = False,
    column_map: t.Optional[t.Dict[str, str]] = None,
    show_progress: bool = True,
    batch_size: t.Optional[int] = None,
    _run_id: t.Optional[UUID] = None,
    _pbar: t.Optional[tqdm] = None,
    return_executor: bool = False,
    allow_nest_asyncio: bool = True,
) -> t.Union[EvaluationResult, Executor]:
    """
    Perform the evaluation on the dataset with different metrics

    Parameters
    ----------
    dataset : Dataset, EvaluationDataset
        The dataset used by the metrics to evaluate the RAG pipeline.
    metrics : list[Metric], optional
        List of metrics to use for evaluation. If not provided, ragas will run
        the evaluation on the best set of metrics to give a complete view.
    llm : BaseRagasLLM, optional
        The language model (LLM) to use to generate the score for calculating the metrics.
        If not provided, ragas will use the default
        language model for metrics that require an LLM. This can be overridden by the LLM
        specified in the metric level with `metric.llm`.
    embeddings : BaseRagasEmbeddings, optional
        The embeddings model to use for the metrics.
        If not provided, ragas will use the default embeddings for metrics that require embeddings.
        This can be overridden by the embeddings specified in the metric level with `metric.embeddings`.
    experiment_name : str, optional
        The name of the experiment to track. This is used to track the evaluation in the tracing tool.
    callbacks : Callbacks, optional
        Lifecycle Langchain Callbacks to run during evaluation.
        Check the [Langchain documentation](https://python.langchain.com/docs/modules/callbacks/) for more information.
    run_config : RunConfig, optional
        Configuration for runtime settings like timeout and retries. If not provided, default values are used.
    token_usage_parser : TokenUsageParser, optional
        Parser to get the token usage from the LLM result.
        If not provided, the cost and total token count will not be calculated. Default is None.
    raise_exceptions : False
        Whether to raise exceptions or not. If set to True, the evaluation will raise an exception
        if any of the metrics fail. If set to False, the evaluation will return `np.nan` for the row that failed. Default is False.
    column_map : dict[str, str], optional
        The column names of the dataset to use for evaluation. If the column names of the dataset are different from the default ones,
        it is possible to provide the mapping as a dictionary here. Example: If the dataset column name is `contexts_v1`, it is possible to pass column_map as `{"contexts": "contexts_v1"}`.
    show_progress : bool, optional
        Whether to show the progress bar during evaluation. If set to False, the progress bar will be disabled. The default is True.
    batch_size : int, optional
        How large the batches should be. If set to None (default), no batching is done.
    return_executor : bool, optional
        If True, returns the Executor instance instead of running evaluation.
        The returned executor can be used to cancel execution by calling executor.cancel().
        To get results, call executor.results(). Default is False.
    allow_nest_asyncio : bool, optional
        Whether to allow nest_asyncio patching for Jupyter compatibility.
        Set to False in production async applications to avoid event loop conflicts. Default is True.

    Returns
    -------
    EvaluationResult or Executor
        If return_executor is False, returns EvaluationResult object containing the scores of each metric.
        If return_executor is True, returns the Executor instance for cancellable execution.

    Raises
    ------
    ValueError
        if validation fails because the columns required for the metrics are missing or
        if the columns are of the wrong format.

    Examples
    --------
    the basic usage is as follows:
    ```
    from ragas import evaluate

    >>> dataset
    Dataset({
        features: ['question', 'ground_truth', 'answer', 'contexts'],
        num_rows: 30
    })

    >>> result = evaluate(dataset)
    >>> print(result)
    {'context_precision': 0.817,
    'faithfulness': 0.892,
    'answer_relevancy': 0.874}
    ```
    """
    warnings.warn(
        "evaluate() is deprecated and will be removed in a future version. "
        "Use the @experiment decorator instead. "
        "See https://docs.ragas.io/en/latest/concepts/experiment/ for more information.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Create async wrapper for aevaluate
    async def _async_wrapper():
        return await aevaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            experiment_name=experiment_name,
            callbacks=callbacks,
            run_config=run_config,
            token_usage_parser=token_usage_parser,
            raise_exceptions=raise_exceptions,
            column_map=column_map,
            show_progress=show_progress,
            batch_size=batch_size,
            _run_id=_run_id,
            _pbar=_pbar,
            return_executor=return_executor,
        )

    if not allow_nest_asyncio:
        # Run without nest_asyncio - creates a new event loop
        import asyncio

        return asyncio.run(_async_wrapper())
    else:
        # Default behavior: use nest_asyncio for backward compatibility (Jupyter notebooks)
        from ragas.async_utils import run

        return run(_async_wrapper())
