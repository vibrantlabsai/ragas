import typing as t

from ragas.callbacks import ChainRun, ChainType, parse_run_traces


def _chain_run(
    run_id: str,
    parent_run_id: t.Optional[str],
    name: str,
    *,
    inputs=None,
    outputs=None,
    children=None,
    chain_type: ChainType,
):
    return ChainRun(
        run_id=run_id,
        parent_run_id=parent_run_id,
        name=name,
        inputs=inputs or {},
        outputs=outputs or {},
        children=children or [],
        metadata={"type": chain_type},
    )


def _traces_for_prompts(metric_name, prompts):
    prompt_ids = [f"prompt-{i + 1}" for i in range(len(prompts))]
    traces = {
        "evaluation": _chain_run(
            "evaluation",
            None,
            "evaluation",
            children=["row"],
            chain_type=ChainType.EVALUATION,
        ),
        "row": _chain_run(
            "row",
            "evaluation",
            "row",
            children=["metric"],
            chain_type=ChainType.ROW,
        ),
        "metric": _chain_run(
            "metric",
            "row",
            metric_name,
            outputs={"output": 0.5},
            children=prompt_ids,
            chain_type=ChainType.METRIC,
        ),
    }
    for prompt_id, (name, input_data, output_data) in zip(prompt_ids, prompts):
        traces[prompt_id] = _chain_run(
            prompt_id,
            "metric",
            name,
            inputs={"data": input_data},
            outputs={"output": [output_data]},
            chain_type=ChainType.RAGAS_PROMPT,
        )
    return traces


def test_parse_run_traces_preserves_repeated_prompt_invocations():
    traces = _traces_for_prompts(
        "factual_correctness",
        [
            (
                "claim_decomposition_prompt",
                {"response": "answer"},
                {"claims": ["answer claim"]},
            ),
            ("n_l_i_statement_prompt", {"context": "reference"}, {"verdict": 1}),
            (
                "claim_decomposition_prompt",
                {"response": "reference"},
                {"claims": ["reference claim"]},
            ),
        ],
    )

    prompt_traces = parse_run_traces(traces)[0]["factual_correctness"]

    assert list(prompt_traces) == ["prompt_1", "prompt_2", "prompt_3"]
    assert [trace["name"] for trace in prompt_traces.values()] == [
        "claim_decomposition_prompt",
        "n_l_i_statement_prompt",
        "claim_decomposition_prompt",
    ]
    assert prompt_traces["prompt_1"]["input"] == {"response": "answer"}
    assert prompt_traces["prompt_3"]["output"] == {"claims": ["reference claim"]}


def test_parse_run_traces_keeps_legacy_shape_for_unique_prompt_names():
    traces = _traces_for_prompts(
        "faithfulness",
        [
            (
                "statement_generator_prompt",
                {"response": "answer"},
                {"statements": ["claim"]},
            ),
            ("n_l_i_statement_prompt", {"context": "context"}, {"verdict": 1}),
        ],
    )

    prompt_traces = parse_run_traces(traces)[0]["faithfulness"]

    assert list(prompt_traces) == [
        "statement_generator_prompt",
        "n_l_i_statement_prompt",
    ]
    assert prompt_traces["statement_generator_prompt"] == {
        "input": {"response": "answer"},
        "output": {"statements": ["claim"]},
    }
