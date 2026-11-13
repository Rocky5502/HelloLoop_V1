from __future__ import annotations

import subprocess
import sys

import helloloop
from helloloop import cli


def test_import_has_version() -> None:
    assert helloloop.__version__ == "1.0.0"


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "helloloop", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "HelloLoop uncertainty-aware loop engineering harness" in result.stdout


def test_model_cli_import_does_not_require_analysis_extra() -> None:
    script = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "matplotlib" or name.startswith("matplotlib."):
        raise ModuleNotFoundError("analysis extra intentionally unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from helloloop import cli
assert "model-catalog" in cli.build_parser().format_help()
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_synthetic_placeholder_is_labeled() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "helloloop", "synthetic"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Synthetic engineering validation only" in result.stdout


def test_cli_signals_blocked_runner(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_run_selected_experiment", lambda *_: {"status": "blocked"})
    assert cli.main(["run-experiment", "--experiment", "pilot"]) == 2


def test_cli_signals_official_evaluation_failures(monkeypatch) -> None:
    monkeypatch.setattr(cli, "evaluate_experiment_runs", lambda *args, **kwargs: {"failed": 1})
    assert cli.main(["evaluate-runs", "--experiment", "pilot"]) == 3


def test_cli_does_not_treat_merge_report_as_evaluator_failure(monkeypatch) -> None:
    monkeypatch.setattr(cli, "merge_shard_artifacts", lambda **kwargs: {"failed": 7})
    assert cli.main(["merge-run-artifacts", "--artifacts-dir", "artifacts"]) == 0
