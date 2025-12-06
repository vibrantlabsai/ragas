"""Tests for the Ragas CLI module."""

from typer.testing import CliRunner

from ragas.cli import app


def test_cli_help():
    """Test that the CLI help command works."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Ragas CLI for running LLM evaluations" in result.stdout


def test_hello_world_help():
    """Test that the hello-world help command works."""
    runner = CliRunner()
    result = runner.invoke(app, ["hello-world", "--help"])
    assert result.exit_code == 0
    assert "Directory to run the hello world example in" in result.stdout


def test_evals_help():
    """Test that the evals help command works."""
    runner = CliRunner()
    result = runner.invoke(app, ["evals", "--help"])
    assert result.exit_code == 0
    assert "Run evaluations on a dataset" in result.stdout


def test_quickstart_help():
    """Test that the quickstart help command works."""
    runner = CliRunner()
    result = runner.invoke(app, ["quickstart", "--help"])
    assert result.exit_code == 0
    assert "Clone a complete example project" in result.stdout


def test_quickstart_list_templates():
    """Test that quickstart lists available templates when no template is specified."""
    runner = CliRunner()
    result = runner.invoke(app, ["quickstart"])
    assert result.exit_code == 0
    assert "Available Ragas Quickstart Templates" in result.stdout
    assert "rag_eval" in result.stdout
    # Note: Other templates (agent_evals, benchmark_llm, etc.) are currently hidden
    # as they are not yet fully implemented. Only rag_eval is available.


def test_quickstart_invalid_template():
    """Test that quickstart fails gracefully with an invalid template."""
    runner = CliRunner()
    result = runner.invoke(app, ["quickstart", "invalid_template"])
    assert result.exit_code == 1
    assert "Unknown template" in result.stdout


def test_quickstart_creates_project(tmp_path):
    """Test that quickstart creates a project structure."""
    runner = CliRunner()
    result = runner.invoke(app, ["quickstart", "rag_eval", "-o", str(tmp_path)])

    # Check exit code
    assert result.exit_code == 0, f"Command failed with output: {result.stdout}"

    # Check success message
    assert "Created RAG Evaluation project" in result.stdout

    # Check that the directory was created
    project_dir = tmp_path / "rag_eval"
    assert project_dir.exists()

    # Check that README exists
    assert (project_dir / "README.md").exists()

    # Check that evals directory structure was created
    evals_dir = project_dir / "evals"
    assert evals_dir.exists(), "evals/ directory should exist"
    assert (evals_dir / "datasets").exists(), "evals/datasets/ should exist"
    assert (evals_dir / "experiments").exists(), "evals/experiments/ should exist"
    assert (evals_dir / "logs").exists(), "evals/logs/ should exist"


if __name__ == "__main__":
    print("Running CLI tests...")
    test_cli_help()
    print("✓ CLI help test passed")
    test_hello_world_help()
    print("✓ Hello world help test passed")
    test_evals_help()
    print("✓ Evals help test passed")
    test_quickstart_help()
    print("✓ Quickstart help test passed")
    test_quickstart_list_templates()
    print("✓ Quickstart list templates test passed")
    test_quickstart_invalid_template()
    print("✓ Quickstart invalid template test passed")
    print("All CLI tests passed!")
