from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ragas.integrations import langsmith as langsmith_integration


def test_langsmith_import_supports_current_langchain_layout():
    assert langsmith_integration.RunEvalConfig.__module__.startswith(
        ("langchain.smith", "langchain_classic.smith")
    )


def test_evaluate_delegates_to_run_on_dataset(monkeypatch: pytest.MonkeyPatch):
    fake_client = MagicMock()
    fake_client.read_dataset.return_value = object()

    monkeypatch.setattr(langsmith_integration, "Client", lambda: fake_client)

    wrapped_metrics = []

    def fake_evaluator_chain(metric):
        def evaluator(run, example=None):
            return {"metric": metric}

        wrapped_metrics.append(evaluator)
        return evaluator

    monkeypatch.setattr(
        langsmith_integration,
        "EvaluatorChain",
        fake_evaluator_chain,
    )

    captured: dict[str, object] = {}

    def fake_run_on_dataset(
        client,
        dataset_name,
        llm_or_chain_factory,
        *,
        evaluation=None,
        verbose=False,
        project_name=None,
        **kwargs,
    ):
        captured["client"] = client
        captured["dataset_name"] = dataset_name
        captured["llm_or_chain_factory"] = llm_or_chain_factory
        captured["evaluation"] = evaluation
        captured["verbose"] = verbose
        captured["project_name"] = project_name
        captured["kwargs"] = kwargs
        return {"project_name": project_name, "dataset_name": dataset_name}

    monkeypatch.setattr(
        langsmith_integration,
        "run_on_dataset",
        fake_run_on_dataset,
    )

    metrics = [object(), object()]

    def chain_factory():
        return None

    result = langsmith_integration.evaluate(
        dataset_name="demo-dataset",
        llm_or_chain_factory=chain_factory,
        experiment_name="demo-experiment",
        metrics=metrics,
        verbose=True,
    )

    assert result == {
        "project_name": "demo-experiment",
        "dataset_name": "demo-dataset",
    }
    assert captured["client"] is fake_client
    assert captured["dataset_name"] == "demo-dataset"
    assert captured["llm_or_chain_factory"] is chain_factory
    assert captured["verbose"] is True
    assert captured["project_name"] == "demo-experiment"

    eval_config = captured["evaluation"]
    assert isinstance(eval_config, langsmith_integration.RunEvalConfig)
    assert eval_config.custom_evaluators == wrapped_metrics


def test_evaluate_translates_missing_dataset_to_value_error(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_client = MagicMock()
    fake_client.read_dataset.side_effect = langsmith_integration.LangSmithNotFoundError(
        "missing"
    )

    monkeypatch.setattr(langsmith_integration, "Client", lambda: fake_client)

    with pytest.raises(ValueError, match="Dataset missing-dataset not found"):
        langsmith_integration.evaluate(
            dataset_name="missing-dataset",
            llm_or_chain_factory=lambda: None,
            metrics=[object()],
        )
