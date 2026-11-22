from __future__ import annotations

import math
import time
import typing as t

import numpy as np
import pytest
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import StringPromptValue as PromptValue

from ragas._analytics import EvaluationEvent
from ragas.llms.base import BaseRagasLLM


class EchoLLM(BaseRagasLLM):
    def generate_text(  # type: ignore
        self,
        prompt: PromptValue,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=prompt.to_string())]])

    async def agenerate_text(  # type: ignore
        self,
        prompt: PromptValue,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=prompt.to_string())]])

    def is_finished(self, response: LLMResult) -> bool:
        return True


def test_debug_tracking_flag(monkeypatch):
    import os

    from ragas._analytics import RAGAS_DEBUG_TRACKING

    monkeypatch.setenv(RAGAS_DEBUG_TRACKING, "true")
    assert os.environ.get(RAGAS_DEBUG_TRACKING, "").lower() == "true"


def test_base_event():
    from ragas._analytics import BaseEvent

    be = BaseEvent(event_type="evaluation")
    assert isinstance(be.model_dump().get("event_type"), str)
    assert isinstance(be.model_dump().get("user_id"), str)


def test_evaluation_event():
    from ragas._analytics import EvaluationEvent

    evaluation_event = EvaluationEvent(
        event_type="evaluation",
        metrics=["harmfulness"],
        num_rows=1,
        language="english",
        evaluation_type="SINGLE_TURN",
    )

    payload = evaluation_event.model_dump()
    assert isinstance(payload.get("user_id"), str)
    assert isinstance(payload.get("evaluation_type"), str)
    assert isinstance(payload.get("metrics"), list)


def setup_user_id_filepath(tmp_path, monkeypatch):
    # setup
    def user_data_dir_patch(appname, roaming=True) -> str:
        return str(tmp_path / appname)

    import ragas._analytics
    from ragas._analytics import USER_DATA_DIR_NAME

    monkeypatch.setattr(ragas._analytics, "user_data_dir", user_data_dir_patch)
    userid_filepath = tmp_path / USER_DATA_DIR_NAME / "uuid.json"

    return userid_filepath


def test_write_to_file(tmp_path, monkeypatch):
    userid_filepath = setup_user_id_filepath(tmp_path, monkeypatch)

    # check if file created if not existing
    assert not userid_filepath.exists()
    import json

    from ragas._analytics import get_userid

    # clear LRU cache since its created in setup for the above test
    get_userid.cache_clear()

    userid = get_userid()
    assert userid_filepath.exists()
    with open(userid_filepath, "r") as f:
        assert userid == json.load(f)["userid"]

    assert not (tmp_path / "uuid.json").exists()

    # del file and check if LRU cache is working
    userid_filepath.unlink()
    assert not userid_filepath.exists()
    userid_cached = get_userid()
    assert userid == userid_cached


def test_load_userid_from_json_file(tmp_path, monkeypatch):
    userid_filepath = setup_user_id_filepath(tmp_path, monkeypatch)
    assert not userid_filepath.exists()

    # create uuid.json file
    userid_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(userid_filepath, "w") as f:
        import json

        json.dump({"userid": "test-userid"}, f)

    from ragas._analytics import get_userid

    # clear LRU cache since its created in setup for the above test
    get_userid.cache_clear()

    assert get_userid() == "test-userid"


def test_testset_generation_tracking(monkeypatch):
    import ragas._analytics as analyticsmodule
    from ragas._analytics import TestsetGenerationEvent, track
    from ragas.testset.synthesizers import default_query_distribution

    distributions = default_query_distribution(llm=EchoLLM())

    testset_event_payload = TestsetGenerationEvent(
        event_type="testset_generation",
        evolution_names=[e.name for e, _ in distributions],
        evolution_percentages=[p for _, p in distributions],
        num_rows=10,
        language="english",
    )

    assert testset_event_payload.model_dump()["evolution_names"] == [
        "single_hop_specific_query_synthesizer",
        "multi_hop_abstract_query_synthesizer",
        "multi_hop_specific_query_synthesizer",
    ]

    assert all(
        np.isclose(
            testset_event_payload.model_dump()["evolution_percentages"],
            [
                0.33,
                0.33,
                0.33,
            ],
            atol=0.01,
        ).tolist()
    )

    # just in the case you actually want to check if tracking is working in the
    # dashboard
    if False:
        monkeypatch.setattr(analyticsmodule, "do_not_track", lambda: False)
        monkeypatch.setattr(analyticsmodule, "_usage_event_debugging", lambda: False)
        track(testset_event_payload)


def test_was_completed(monkeypatch):
    from ragas._analytics import IsCompleteEvent, track_was_completed

    event_properties_list: t.List[IsCompleteEvent] = []

    def echo_track(event_properties):
        event_properties_list.append(event_properties)

    monkeypatch.setattr("ragas._analytics.track", echo_track)

    @track_was_completed
    def test(raise_error=True):
        if raise_error:
            raise ValueError("test")
        else:
            pass

    with pytest.raises(ValueError):
        test(raise_error=True)

    assert event_properties_list[-1].event_type == "test"
    assert event_properties_list[-1].is_completed is False

    test(raise_error=False)

    assert event_properties_list[-1].event_type == "test"
    assert event_properties_list[-1].is_completed is True


evaluation_events_and_num_rows = [
    (  # 5 same events
        [
            EvaluationEvent(
                event_type="evaluation",
                metrics=["harmfulness"],
                num_rows=1,
                evaluation_type="SINGLE_TURN",
                language="english",
            )
            for _ in range(5)
        ],
        [5],
    ),
    (  # 5 different events with different metrics
        [
            EvaluationEvent(
                event_type="evaluation",
                metrics=[f"harmfulness_{i}"],
                num_rows=1,
                evaluation_type="SINGLE_TURN",
                language="english",
            )
            for i in range(5)
        ],
        [1, 1, 1, 1, 1],
    ),
    (  # 5 different events with different num_rows but 2 group of metrics
        [
            EvaluationEvent(
                metrics=["harmfulness"],
                num_rows=1,
                evaluation_type="SINGLE_TURN",
                language="english",
            )
            for i in range(10)
        ]
        + [
            EvaluationEvent(
                event_type="evaluation",
                metrics=["accuracy"],
                num_rows=1,
                evaluation_type="SINGLE_TURN",
                language="english",
            )
            for i in range(5)
        ],
        [10, 5],
    ),
]


@pytest.mark.parametrize(
    "evaluation_events, expected_num_rows_set", evaluation_events_and_num_rows
)
def test_analytics_batcher_join_evaluation_events(
    monkeypatch, evaluation_events, expected_num_rows_set
):
    """
    Test if the batcher joins the evaluation events correctly
    """
    from ragas._analytics import AnalyticsBatcher

    batcher = AnalyticsBatcher()

    joined_events = batcher._join_evaluation_events(evaluation_events)
    assert len(joined_events) == len(expected_num_rows_set)
    assert sorted(e.num_rows for e in joined_events) == sorted(expected_num_rows_set)


@pytest.mark.skip(reason="This test is flaky and needs to be fixed")
@pytest.mark.parametrize(
    "evaluation_events, expected_num_rows_set", evaluation_events_and_num_rows
)
def test_analytics_batcher_flush(monkeypatch, evaluation_events, expected_num_rows_set):
    """
    Test if the batcher flushes the events correctly
    """
    from ragas._analytics import AnalyticsBatcher

    FLUSH_INTERVAL = 0.3
    BATCH_SIZE = 5
    batcher = AnalyticsBatcher(batch_size=BATCH_SIZE, flush_interval=FLUSH_INTERVAL)

    # Use a list to hold the counter so it can be modified in the nested function
    flush_mock_call_count = [0]

    def flush_mock():
        # Access the list and modify its first element
        flush_mock_call_count[0] += 1
        batcher.buffer = []
        batcher.last_flush_time = time.time()

    monkeypatch.setattr(batcher, "flush", flush_mock)

    for event in evaluation_events[:-1]:
        batcher.add_evaluation(event)

    # Access the counter using flush_mock_call_count[0]
    time.sleep(FLUSH_INTERVAL + 0.1)
    batcher.add_evaluation(evaluation_events[-1])
    assert flush_mock_call_count[0] == math.ceil(
        sum(expected_num_rows_set) / BATCH_SIZE
    )

def test_do_not_track_standard_env_vars(monkeypatch):
    """Test that all standard do-not-track env vars are respected."""
    from ragas._analytics import do_not_track

    # Test DO_NOT_TRACK=1
    do_not_track.cache_clear()
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    assert do_not_track() is True
    monkeypatch.delenv("DO_NOT_TRACK")

    # Test DISABLE_TELEMETRY=1
    do_not_track.cache_clear()
    monkeypatch.setenv("DISABLE_TELEMETRY", "1")
    assert do_not_track() is True
    monkeypatch.delenv("DISABLE_TELEMETRY")

    # Test RAGAS_ANALYTICS_ENABLED=false
    do_not_track.cache_clear()
    monkeypatch.setenv("RAGAS_ANALYTICS_ENABLED", "false")
    assert do_not_track() is True
    monkeypatch.delenv("RAGAS_ANALYTICS_ENABLED")

    # Test RAGAS_TELEMETRY_ENABLED=false
    do_not_track.cache_clear()
    monkeypatch.setenv("RAGAS_TELEMETRY_ENABLED", "false")
    assert do_not_track() is True
    monkeypatch.delenv("RAGAS_TELEMETRY_ENABLED")

    # Test default — tracking enabled
    do_not_track.cache_clear()
    assert do_not_track() is False


def test_track_runs_in_daemon_thread(monkeypatch):
    """Test that track() fires in a daemon thread and doesn't block."""
    import threading
    import time
    from ragas._analytics import BaseEvent, track

    monkeypatch.setattr("ragas._analytics.do_not_track", lambda: False)
    monkeypatch.setattr("ragas._analytics._usage_event_debugging", lambda: False)

    threads_used = []

    original_thread_init = threading.Thread.__init__

    def patched_init(self, *args, **kwargs):
        original_thread_init(self, *args, **kwargs)
        threads_used.append(self)

    monkeypatch.setattr(threading.Thread, "__init__", patched_init)

    # Mock requests.post to avoid real network call
    monkeypatch.setattr(
        "ragas._analytics.requests.post",
        lambda *a, **kw: None,
    )

    event = BaseEvent(event_type="test_event")
    start = time.time()
    track(event)
    elapsed = time.time() - start

    # Should return almost instantly — not block for DNS timeout
    assert elapsed < 1.0, f"track() blocked for {elapsed:.2f}s — DNS fix not working"
    # A daemon thread should have been created
    assert any(t.daemon for t in threads_used)


def test_track_dns_failure_does_not_block(monkeypatch):
    """Test that a simulated DNS failure doesn't block the main thread."""
    import time
    from ragas._analytics import BaseEvent, track

    monkeypatch.setattr("ragas._analytics.do_not_track", lambda: False)
    monkeypatch.setattr("ragas._analytics._usage_event_debugging", lambda: False)

    def slow_post(*args, **kwargs):
        # Simulate a 10s DNS hang
        time.sleep(10)

    monkeypatch.setattr("ragas._analytics.requests.post", slow_post)

    event = BaseEvent(event_type="test_event")
    start = time.time()
    track(event)
    elapsed = time.time() - start

    # Main thread must return in under 1s even with 10s DNS hang
    assert elapsed < 1.0, f"track() blocked for {elapsed:.2f}s"
