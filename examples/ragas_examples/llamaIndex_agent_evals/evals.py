import json
import os
from collections import Counter
from typing import Any, Dict, List

import instructor

from ragas import Dataset, experiment
from ragas.llms import InstructorLLM
from ragas.metrics import DiscreteMetric, numeric_metric
from ragas.metrics.result import MetricResult

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context, JsonSerializer

from llamaindex_agent import add_item, list_items, llm, remove_item

evaluator_llm = InstructorLLM(
    client=instructor.from_provider(
        "google/gemini-2.0-flash",
        async_client=True,
        api_key=os.environ["GOOGLE_API_KEY"],
    ),
    model="gemini-2.0-flash",
    provider="google",
)


@numeric_metric(name="tool_call_accuracy")
def tool_call_accuracy_metric(
    predicted_calls: List[Dict], ground_truth_calls: List[Dict]
):
    def _normalize(d):
        """Recursively convert dicts/lists into hashable tuples."""
        if isinstance(d, dict):
            return tuple(sorted((k, _normalize(v)) for k, v in d.items()))
        elif isinstance(d, list):
            return tuple(_normalize(v) for v in d)
        else:
            return d

    try:
        if not predicted_calls and not ground_truth_calls:
            return MetricResult(
                value=1.0,
                reason="Both predicted and ground truth are empty (perfect match)",
            )

        gt_counter = Counter(_normalize(d) for d in ground_truth_calls)
        pred_counter = Counter(_normalize(d) for d in predicted_calls)

        tp = sum((gt_counter & pred_counter).values())
        fp = sum((pred_counter - gt_counter).values())
        fn = sum((gt_counter - pred_counter).values())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return MetricResult(
            value=f1,
            reason=(
                f"TP={tp}, FP={fp}, FN={fn}, "
                f"Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}"
            ),
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return MetricResult(value=0.0, reason=f"Error: {str(e)}")


@numeric_metric(name="goal_accuracy(without llm)")
def goal_accuracy_metric_without_llm(current_state: Dict, expected_state: Dict):
    try:
        if not current_state and not expected_state:
            return MetricResult(
                value=1.0,
                reason="Both current state and expected state are empty (perfect match)",
            )

        def normalize_state(state: Dict[str, Any]) -> Counter:
            flat = []
            for k, v in state.items():
                if isinstance(v, list):
                    flat.extend((k, item) for item in v)  # pair (key, item)
                else:
                    flat.append((k, v))
            return Counter(flat)

        gt_counter = normalize_state(expected_state)
        pred_counter = normalize_state(current_state)

        tp = sum((gt_counter & pred_counter).values())
        fp = sum((pred_counter - gt_counter).values())
        fn = sum((gt_counter - pred_counter).values())

        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return MetricResult(
            value=f1,
            reason=f"TP={tp}, FP={fp}, FN={fn}, Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}",
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return MetricResult(value=0.0, reason=f"Error: {str(e)}")


goal_accuracy_metric_with_llm = DiscreteMetric(
    name="goal_accuracy(with llm)",
    prompt="""You are evaluating whether the user’s action achieved the intended goal.

- Initial State: {initial_state}
- Final State: {final_state}
- User Input: {user_input}

Determine if the change from Initial State to Final State correctly reflects the User Input.

If yes, return 'pass'.
If no, return 'fail'.""",
    allowed_values=["pass", "fail"],
)


def load_dataset():
    # Create a dataset
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir=".",
    )

    test_cases = [
        {
            "test_case": "Coreference",
            "user_input": "add tomatoes and potatos. actually delete them",
            "context": None,
            "ground_truth_calls": [
                {"tool_name": "add_item", "tool_kwargs": {"item": "tomatoes"}},
                {"tool_name": "add_item", "tool_kwargs": {"item": "potatos"}},
                {"tool_name": "remove_item", "tool_kwargs": {"item": "tomatoes"}},
                {"tool_name": "remove_item", "tool_kwargs": {"item": "potatos"}},
            ],
            "expected_state": {"shopping_list": []},
        },
        {
            "test_case": "Correction/replace",
            "user_input": "add sugar… sorry, I meant brown sugar",
            "context": None,
            "ground_truth_calls": [
                {"tool_name": "add_item", "tool_kwargs": {"item": "brown sugar"}}
            ],
            "expected_state": {"shopping_list": ["brown sugar"]},
        },
        {
            "test_case": "Implicit intent",
            "user_input": "we’re out of milk",
            "context": None,
            "ground_truth_calls": [
                {"tool_name": "add_item", "tool_kwargs": {"item": "milk"}}
            ],
            "expected_state": {"shopping_list": ["milk"]},
        },
        {
            "test_case": "Mixed actions",
            "user_input": "Can you show me the list and also add butter?",
            "context": None,
            "ground_truth_calls": [
                {"tool_name": "list_items", "tool_kwargs": {}},
                {"tool_name": "add_item", "tool_kwargs": {"item": "butter"}},
            ],
            "expected_state": {"shopping_list": ["butter"]},
        },
        {
            "test_case": "Handle an ambiguous removal request",
            "user_input": "remove cheese",
            "context": json.load(open("./contexts/ambiguous_removal_request.json")),
            "ground_truth_calls": [],
            "expected_state": {"shopping_list": ["cheddar cheese", "provolone cheese"]},
        },
        {
            "test_case": "Adding duplicate item ",
            "user_input": "add bread",
            "context": json.load(open("./contexts/duplicate_addition.json")),
            "ground_truth_calls": [
                {"tool_name": "add_item", "tool_kwargs": {"item": "bread"}}
            ],
            "expected_state": {"shopping_list": ["milk", "eggs", "bread"]},
        },
        {
            "test_case": "Repeated removal",
            "user_input": "remove milk",
            "context": json.load(open("./contexts/repeated_removal.json")),
            "ground_truth_calls": [
                {"tool_name": "remove_item", "tool_kwargs": {"item": "milk"}}
            ],
            "expected_state": {"shopping_list": ["eggs", "bread"]},
        },
    ]

    # Add the data to the dataset
    for row in test_cases:
        dataset.append(row)

    dataset.save()  # Save the dataset
    return dataset


@experiment()
async def run_experiment(row):
    user_input = row["user_input"]
    ground_truth_calls = row["ground_truth_calls"]
    context = row["context"]
    # Get the model's prediction

    workflow = FunctionAgent(
        tools=[add_item, remove_item, list_items],
        llm=llm,
        system_prompt="""Your job is to manage a shopping list.
The shopping list starts empty. You can add items, remove items by name, and list all items.""",
        initial_state={"shopping_list": []},
    )

    if context:
        ctx = Context.from_dict(workflow, context, serializer=JsonSerializer())
        initial_state = await ctx.store.get("state")
    else:
        ctx = Context(workflow)
        initial_state = workflow.initial_state

    response = await workflow.run(user_msg=user_input, ctx=ctx)
    final_state = await ctx.store.get("state")

    predicted_calls = []

    if hasattr(response, "tool_calls") and response.tool_calls:
        for i in response.tool_calls:
            predicted_calls.append(
                {"tool_name": i.tool_name, "tool_kwargs": i.tool_kwargs}
            )

    # Calculate metrics
    tool_call_accuracy = tool_call_accuracy_metric.score(
        predicted_calls=predicted_calls, ground_truth_calls=ground_truth_calls
    )

    goal_accuracy_with_llm = goal_accuracy_metric_with_llm.score(
        llm=evaluator_llm,
        initial_state=initial_state,
        final_state=final_state,
        user_input=user_input,
    )

    goal_accuracy_without_llm = goal_accuracy_metric_without_llm.score(
        current_state=final_state,
        expected_state=row["expected_state"],
    )

    return {
        "user_input": user_input,
        "response": str(response),
        "tool_call_accuracy(f1)": tool_call_accuracy.value,
        "goal_accuracy(with llm)": goal_accuracy_with_llm.value,
        "goal_accuracy(without llm)": goal_accuracy_without_llm.value,
    }


async def main():
    dataset = load_dataset()
    experiment_result = await run_experiment.arun(dataset)
    print("Experiment_result: ", experiment_result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
