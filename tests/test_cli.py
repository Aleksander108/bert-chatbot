"""Tests for the main CLI interface."""

from typer.testing import CliRunner

from bert_chatbot.cli import app


def test_cli_app() -> None:
    """Test that the CLI app runs without errors and shows help text."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "BERT Semantic ChatBot." in result.stdout
    assert "chat" in result.stdout


def test_cli_no_args() -> None:
    """Test CLI with no arguments shows help."""
    runner = CliRunner()
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "BERT Semantic ChatBot." in result.stdout
