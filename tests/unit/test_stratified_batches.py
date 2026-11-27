import threading

from ragas.dataset_schema import (
    PromptAnnotation,
    SampleAnnotation,
    SingleMetricAnnotation,
)


def _annotation(label):
    return SampleAnnotation(
        metric_input={"user_input": "", "response": "", "reference": ""},
        metric_output=label,
        prompts={
            "prompt": PromptAnnotation(
                prompt_input={"user_input": ""},
                prompt_output={"verdict": 1},
                edited_output=None,
            )
        },
        is_accepted=True,
        target=None,
    )


def _dataset():
    # 5 samples across 2 classes
    labels = [0, 0, 0, 1, 1]
    return SingleMetricAnnotation(
        name="metric", samples=[_annotation(label) for label in labels]
    )


def _run_with_timeout(fn, timeout=5):
    result = {}

    def target():
        result["value"] = fn()

    # daemon thread so a hang on buggy code does not block the test process
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    return result, thread.is_alive()


def test_stratified_batches_drop_last_batch_terminates_and_drops_remainder():
    dataset = _dataset()

    result, hung = _run_with_timeout(
        lambda: dataset.stratified_batches(
            batch_size=2,
            stratify_key="metric_output",
            drop_last_batch=True,
            replace=False,
        )
    )

    assert not hung, "stratified_batches(drop_last_batch=True) did not terminate"
    batches = result["value"]
    assert len(batches) == 2
    assert all(len(batch) == 2 for batch in batches)


def test_stratified_batches_keep_last_batch_returns_remainder():
    dataset = _dataset()

    result, hung = _run_with_timeout(
        lambda: dataset.stratified_batches(
            batch_size=2,
            stratify_key="metric_output",
            drop_last_batch=False,
            replace=False,
        )
    )

    assert not hung, "stratified_batches(drop_last_batch=False) did not terminate"
    batches = result["value"]
    assert len(batches) == 3
    assert sorted(len(batch) for batch in batches) == [1, 2, 2]
