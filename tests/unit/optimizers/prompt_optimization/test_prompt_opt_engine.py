"""R10-R12: the ragas -> DSPy Anthropic engine (breakpoints, params, budget, refusals)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("dspy")

from dspy.lm15 import Config, Message, Request  # noqa: E402

from ragas.optimizers.prompt_optimization.anthropic_engine import (  # noqa: E402
    RagasAnthropicEngine,
    make_dspy_lm,
    unwrap_anthropic_client,
)
from ragas.optimizers.prompt_optimization.budget import BudgetExceeded, BudgetTracker  # noqa: E402
from ragas.optimizers.prompt_optimization.bundle import FewShot, PromptBundle  # noqa: E402
from ragas.optimizers.prompt_optimization.config import BudgetConfig  # noqa: E402
from ragas.optimizers.prompt_optimization.signature import DSPyAnswerer  # noqa: E402
from ragas.optimizers.prompt_optimization.simulated import (  # noqa: E402
    SimulatedAnthropicClient,
    SimulatedStudent,
    format_fields,
)


def test_build_live_runtime_wiring_offline():
    """The live path (ragas llm_factory -> engine -> dspy.LM), with no request sent."""
    anthropic = pytest.importorskip("anthropic")
    from ragas.optimizers.prompt_optimization.config import ModelConfig
    from ragas.optimizers.prompt_optimization.runtime import build_live_runtime

    budget = BudgetTracker()
    runtime = build_live_runtime(
        ModelConfig(effort="low"),
        budget,
        client=anthropic.Anthropic(api_key="sk-test"),
        cache=False,
    )
    assert runtime.mode == "live"
    assert runtime.student_engine.model == "claude-haiku-5-5"
    assert runtime.proposer_engine.model == "claude-sonnet-5-5"
    assert isinstance(runtime.student_engine.client, anthropic.Anthropic)
    assert runtime.student_engine.budget is budget
    params = runtime.student_engine.build_params(_request())
    assert params["model"] == "claude-haiku-5-5"
    assert params["output_config"] == {"effort": "low"}
    assert not {"temperature", "top_p"} & set(params)
    assert runtime.student_lm.model == "anthropic/claude-haiku-5-5"


def _request(n_demos=2, temperature=0.7):
    msgs = []
    for i in range(n_demos):
        msgs += [Message.user(f"demo q{i}"), Message.assistant(f"demo a{i}")]
    msgs.append(Message.user("live question"))
    return Request(
        model="x",
        system="SYSTEM + INSTRUCTION",
        messages=tuple(msgs),
        config=Config(max_tokens=321, temperature=temperature, top_p=0.5),
    )


def _engine(responder=None, **kw):
    client = SimulatedAnthropicClient(
        responder or (lambda p: format_fields({"answer": "ok"}))
    )
    return RagasAnthropicEngine(client=client, model="claude-haiku-5-5", **kw), client


def test_cache_breakpoints_at_instruction_and_last_demo():
    engine, _ = _engine()
    params = engine.build_params(_request(n_demos=2))
    assert params["system"] == [
        {
            "type": "text",
            "text": "SYSTEM + INSTRUCTION",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    marked = [
        i
        for i, m in enumerate(params["messages"])
        if "cache_control" in m["content"][-1]
    ]
    assert marked == [3], "only the last few-shot assistant turn is breakpoint 2"
    assert params["messages"][-1]["content"][-1]["text"] == "live question"
    assert "cache_control" not in params["messages"][-1]["content"][-1]


def test_no_demo_breakpoint_without_demos_and_can_disable():
    engine, _ = _engine()
    params = engine.build_params(_request(n_demos=0))
    assert all("cache_control" not in m["content"][-1] for m in params["messages"])
    off, _ = _engine(cache_breakpoints=False)
    assert "cache_control" not in off.build_params(_request())["system"][0]


def test_no_sampling_params_and_effort_and_max_tokens():
    engine, _ = _engine(effort="low")
    params = engine.build_params(_request(temperature=1.3))
    assert not {"temperature", "top_p", "top_k"} & set(params)
    assert params["output_config"] == {"effort": "low"}
    assert params["max_tokens"] == 321
    no_effort, _ = _engine(effort=None)
    assert "output_config" not in no_effort.build_params(_request())


def test_developer_role_mapped_and_unsupported_role_rejected():
    engine, _ = _engine()
    req = Request(model="x", messages=(Message.developer("note"), Message.user("q")))
    assert engine.build_params(req)["messages"][0]["role"] == "user"
    bad = SimpleNamespace(
        system=None,
        messages=[SimpleNamespace(role="tool", parts=())],
        config=SimpleNamespace(max_tokens=None),
    )
    with pytest.raises(ValueError, match="tool"):
        engine.build_params(bad)


def test_complete_records_usage_and_returns_text():
    budget = BudgetTracker(BudgetConfig(max_usd=5.0))
    engine, client = _engine(budget=budget)
    response = engine.complete(_request())
    assert response.text.startswith("[[ ## answer ## ]]")
    assert response.finish_reason == "stop"
    assert engine.calls == 1 and len(client.requests) == 1
    assert budget.summary()["calls"] == 1 and budget.total_usd > 0
    engine.complete(_request())  # same system prefix -> simulated cache hit
    assert budget.summary()["tokens"]["cache_read_input_tokens"] > 0
    assert list(engine.stream(_request()))  # stream emits events
    engine.close()


def test_budget_cap_blocks_new_calls():
    budget = BudgetTracker(BudgetConfig(max_usd=0.0))
    engine, client = _engine(budget=budget)
    with pytest.raises(BudgetExceeded):
        engine.complete(_request())
    assert client.requests == []


def test_refusal_and_max_tokens_finish_reasons():
    engine, client = _engine()

    def refusal(**params):
        return SimpleNamespace(
            id="m1",
            model="claude-haiku-5-5",
            stop_reason="refusal",
            content=[SimpleNamespace(type="text", text="no")],
            usage=None,
        )

    client.messages.create = refusal
    response = engine.complete(_request())
    assert response.text == "" and response.finish_reason == "content_filter"

    client.messages.create = lambda **p: SimpleNamespace(
        id=None,
        model=None,
        stop_reason="max_tokens",
        content=[
            SimpleNamespace(type="thinking", text="hidden"),
            SimpleNamespace(type="text", text="cut"),
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )
    response = engine.complete(_request())
    assert response.finish_reason == "length" and response.text == "cut"


def test_unwrap_instructor_client_to_raw_client():
    raw = MagicMock(spec=["messages"])
    wrapper = SimpleNamespace(
        client=raw, messages=SimpleNamespace(create=lambda **k: None)
    )
    assert unwrap_anthropic_client(wrapper) is raw
    with pytest.raises(TypeError):
        unwrap_anthropic_client(object())


def test_engine_from_ragas_llm_keeps_explicit_sampling_only():
    anthropic = pytest.importorskip("anthropic")
    from ragas.llms import llm_factory

    client = anthropic.Anthropic(api_key="sk-test")
    default = RagasAnthropicEngine(
        llm_factory("claude-haiku-5-5", provider="anthropic", client=client)
    )
    assert default.model == "claude-haiku-5-5"
    assert isinstance(default.client, anthropic.Anthropic), "must unwrap instructor"
    assert "temperature" not in default.build_params(_request())
    explicit = RagasAnthropicEngine(
        llm_factory(
            "claude-haiku-5-5", provider="anthropic", client=client, temperature=1.0
        )
    )
    assert explicit.build_params(_request())["temperature"] == 1.0


def test_engine_argument_validation():
    with pytest.raises(ValueError):
        RagasAnthropicEngine()
    other = SimpleNamespace(provider="openai", model="gpt", client=None)
    with pytest.raises(ValueError, match="anthropic"):
        RagasAnthropicEngine(other)


def test_dspy_answerer_end_to_end_through_engine(golden_rows):
    engine, client = _engine(responder=SimulatedStudent(golden_rows))
    answerer = DSPyAnswerer(make_dspy_lm(engine, cache=False, num_retries=0))
    row = golden_rows[0]
    verbose = answerer(PromptBundle("Be thorough."), row)
    assert row.reference.rstrip(".") in verbose and verbose != row.reference
    with_demo = PromptBundle(
        "Be thorough.", few_shots=[FewShot.from_row(golden_rows[4])]
    )
    assert answerer(with_demo, row) == row.reference
    # The DSPy request reached the engine with a demo turn marked as breakpoint 2.
    last = client.requests[-1]
    assert last["messages"][1]["role"] == "assistant"
    assert "cache_control" in last["messages"][1]["content"][-1]
