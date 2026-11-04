"""R15: the prompt_optimization tool runs end to end offline; quickstart ships it."""

import importlib.util
import json

import pytest
from prompt_opt_testdata import TEMPLATE_DIR
from typer.testing import CliRunner

from ragas.cli import app


def _load_from(path):
    spec = importlib.util.spec_from_file_location(
        f"po_tool_{path.stem}_{abs(hash(path))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(name):
    return _load_from(TEMPLATE_DIR / f"{name}.py")


def test_quickstart_lists_template():
    result = CliRunner().invoke(app, ["quickstart"])
    assert result.exit_code == 0
    assert "prompt_optimization" in result.stdout


def test_quickstart_copies_template(tmp_path):
    result = CliRunner().invoke(
        app, ["quickstart", "prompt_optimization", "-o", str(tmp_path)]
    )
    assert result.exit_code == 0, result.stdout
    project = tmp_path / "prompt_optimization"
    # quickstart moves datasets/ to evals/datasets/
    for name in (
        "optimize.py",
        "evals.py",
        "rag.py",
        "README.md",
        "evals/datasets/hr_policy_qa_golden.jsonl",
        "prompts/baseline_system_prompt.md",
        "config/optimization_goal.json",
    ):
        assert (project / name).exists(), name
    # The copied tool still finds its dataset by default.
    copied = _load_from(project / "optimize.py")
    assert (
        copied.DEFAULT_DATASET
        == project / "evals" / "datasets" / "hr_policy_qa_golden.jsonl"
    )


def test_parser_defaults():
    args = _load("optimize").build_parser().parse_args([])
    assert args.student == "claude-haiku-5-5" and args.proposer == "claude-sonnet-5-5"
    assert args.max_usd == 5.0 and not args.dry_run


def test_dry_run_end_to_end(tmp_path, capsys):
    pytest.importorskip("dspy")
    pytest.importorskip("optuna")
    code = _load("optimize").main(
        [
            "--dry-run",
            "--rounds",
            "1",
            "--demo-sets",
            "4",
            "--instructions",
            "3",
            "--trials",
            "3",
            "--threads",
            "2",
            "--out-dir",
            str(tmp_path / "run"),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "Before vs after" in printed and "Spend:" in printed
    comparison = json.loads(
        (tmp_path / "run" / "comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["split"] == "holdout"
    assert (tmp_path / "run" / "final" / "final_prompt.md").exists()


def test_eval_tool_dry_run(tmp_path, capsys):
    pytest.importorskip("dspy")
    code = _load("evals").main(
        ["--dry-run", "--split", "test", "--out-dir", str(tmp_path / "e")]
    )
    assert code == 0
    assert "token_f1" in capsys.readouterr().out
    assert (tmp_path / "e" / "experiments" / "eval.csv").exists()


def test_rag_build_request_uses_breakpoints(tmp_path):
    from ragas.optimizers.prompt_optimization import PromptBundle, export_bundle

    paths = export_bundle(
        PromptBundle("Answer briefly."), tmp_path, model="claude-haiku-5-5"
    )
    request = _load("rag").build_request(
        paths["json"], "When are salaries paid?", ["Paid monthly."]
    )
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "When are salaries paid?" in request["messages"][-1]["content"][0]["text"]
    assert "[1] Paid monthly." in request["messages"][-1]["content"][0]["text"]


def test_load_env_file_reads_only_anthropic_keys_and_never_overrides(
    tmp_path, monkeypatch
):
    tool = _load("optimize")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "already-set")
    monkeypatch.delenv("OTHER_SECRET", raising=False)
    (tmp_path / ".env").write_text(
        '# comment\nexport ANTHROPIC_API_KEY="sk-ant-from-file"\n'
        "ANTHROPIC_AUTH_TOKEN=from-file\nOTHER_SECRET=x\nnot a pair\n",
        encoding="utf-8",
    )
    assert tool.load_env_file(tmp_path / "missing", tmp_path) == tmp_path / ".env"
    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-file"
    assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "already-set"
    assert "OTHER_SECRET" not in os.environ
    assert tool.load_env_file(tmp_path / "missing") is None
