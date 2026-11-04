"""R11 (budget math) and R13 (export markers and round trip)."""

import json
import threading

import pytest

from ragas.optimizers.prompt_optimization.budget import (
    BudgetExceeded,
    BudgetTracker,
    Usage,
)
from ragas.optimizers.prompt_optimization.bundle import FewShot, PromptBundle
from ragas.optimizers.prompt_optimization.config import BudgetConfig, ModelPrice
from ragas.optimizers.prompt_optimization.export import (
    FEW_SHOTS_END,
    FEW_SHOTS_START,
    INPUT_START,
    INSTRUCTION_END,
    INSTRUCTION_START,
    anthropic_request_template,
    export_bundle,
    load_prompt_bundle,
    parse_markdown,
    render_markdown,
)

# -- budget -----------------------------------------------------------------


def test_cost_math_uses_per_million_prices():
    tracker = BudgetTracker(
        BudgetConfig(
            max_usd=1.0,
            prices={
                "m": ModelPrice(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5)
            },
        )
    )
    cost = tracker.record("m", Usage(1_000_000, 100_000, 500_000, 200_000))
    # 2.00 + 1.00 + 0.10 + 0.50
    assert cost == pytest.approx(3.60)
    assert tracker.summary()["by_model"]["m"] == pytest.approx(3.60)


def test_default_cache_prices_and_unknown_model():
    price = ModelPrice(input=1.0, output=5.0)
    assert price.cache_read_price() == pytest.approx(0.1)
    assert price.cache_write_price() == pytest.approx(1.25)
    with pytest.raises(KeyError, match="nope"):
        BudgetTracker().price_for("nope")


def test_check_raises_once_cap_reached():
    tracker = BudgetTracker(BudgetConfig(max_usd=0.001))
    tracker.check()
    tracker.stage = "layer1"
    tracker.record("claude-haiku-5-5", Usage(input_tokens=20_000))
    with pytest.raises(BudgetExceeded, match="0.00"):
        tracker.check()
    assert tracker.remaining_usd == 0.0
    assert tracker.summary()["by_stage"]["layer1"] > 0


def test_tracker_is_thread_safe():
    tracker = BudgetTracker(BudgetConfig(max_usd=100.0))
    threads = [
        threading.Thread(
            target=lambda: [
                tracker.record("claude-haiku-5-5", Usage(10, 10)) for _ in range(200)
            ]
        )
        for _ in range(8)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert tracker.summary()["calls"] == 1600


# -- export -----------------------------------------------------------------


def _bundle(golden_rows):
    return PromptBundle(
        instruction="Answer from the documents only.\nBe concise.",
        few_shots=[FewShot.from_row(golden_rows[0]), FewShot.from_row(golden_rows[12])],
        source="layer2:instruction=1",
    )


def test_markdown_has_markers_in_order(golden_rows):
    text = render_markdown(_bundle(golden_rows), {"test": {"token_f1": 0.9}})
    positions = [
        text.index(m)
        for m in (
            INSTRUCTION_START,
            INSTRUCTION_END,
            FEW_SHOTS_START,
            FEW_SHOTS_END,
            INPUT_START,
        )
    ]
    assert positions == sorted(positions)
    assert "<!-- RAGAS:SHOT 1 id=f00 -->" in text
    assert "Building cache breakpoints" in text


def test_markdown_round_trip(golden_rows):
    bundle = _bundle(golden_rows)
    parsed = parse_markdown(render_markdown(bundle))
    assert parsed.instruction == bundle.instruction
    assert parsed.few_shots == bundle.few_shots


def test_parse_errors():
    with pytest.raises(ValueError):
        parse_markdown("no markers here")
    broken = f"{INSTRUCTION_START}x{INSTRUCTION_END}{FEW_SHOTS_START}<!-- RAGAS:SHOT 1 id=a -->\n### question\nq{FEW_SHOTS_END}"
    with pytest.raises(ValueError, match="missing"):
        parse_markdown(broken)


def test_request_template_breakpoints(golden_rows):
    bundle = _bundle(golden_rows)
    req = anthropic_request_template(bundle, "claude-haiku-5-5")
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    msgs = req["messages"]
    assert [m["role"] for m in msgs] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert msgs[-2]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in msgs[-1]["content"][-1], (
        "live input must not be cached"
    )
    assert sum("cache_control" in c for m in msgs for c in m["content"]) == 1
    assert "temperature" not in req and "top_p" not in req


def test_request_template_without_demos():
    req = anthropic_request_template(PromptBundle("Only an instruction."), "m")
    assert (
        len(req["messages"]) == 1
        and "cache_control" not in req["messages"][0]["content"][0]
    )


def test_export_files_and_load(tmp_path, golden_rows):
    bundle = _bundle(golden_rows)
    paths = export_bundle(
        bundle, tmp_path, model="claude-haiku-5-5", scores={"test": {"token_f1": 1.0}}
    )
    assert set(paths) == {"markdown", "json", "request_template"}
    assert load_prompt_bundle(paths["json"]).fingerprint() == bundle.fingerprint()
    assert load_prompt_bundle(paths["markdown"]).fingerprint() == bundle.fingerprint()
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["scores"]["test"]["token_f1"] == 1.0
    assert (
        json.loads(paths["request_template"].read_text(encoding="utf-8"))["model"]
        == "claude-haiku-5-5"
    )


def test_bundle_dict_round_trip_and_helpers(golden_rows):
    bundle = _bundle(golden_rows)
    again = PromptBundle.from_dict(bundle.to_dict())
    assert again.fingerprint() == bundle.fingerprint()
    assert bundle.with_instruction("new", "s").few_shots == bundle.few_shots
    assert bundle.prompt_length > len(bundle.instruction)
