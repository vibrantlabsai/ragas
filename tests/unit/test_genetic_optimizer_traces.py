from ragas.optimizers.genetic import GeneticOptimizer


def test_get_prompt_trace_supports_ordered_trace_shape():
    prompt_traces = {
        "prompt_1": {
            "name": "claim_decomposition_prompt",
            "input": "first-input",
            "output": "first-output",
        },
        "prompt_2": {
            "name": "n_l_i_statement_prompt",
            "input": "nli-input",
            "output": "nli-output",
        },
        "prompt_3": {
            "name": "claim_decomposition_prompt",
            "input": "last-input",
            "output": "last-output",
        },
    }

    trace = GeneticOptimizer._get_prompt_trace(
        prompt_traces, "claim_decomposition_prompt"
    )

    assert trace == prompt_traces["prompt_3"]


def test_get_prompt_trace_supports_legacy_trace_shape():
    prompt_traces = {
        "claim_decomposition_prompt": {
            "input": "legacy-input",
            "output": "legacy-output",
        }
    }

    trace = GeneticOptimizer._get_prompt_trace(
        prompt_traces, "claim_decomposition_prompt"
    )

    assert trace == prompt_traces["claim_decomposition_prompt"]
