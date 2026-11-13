from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from helloloop.experiments import runner


def test_minisweagent_model_mapping_is_passed_as_config_keyword() -> None:
    calls = []
    expected = {"model_name": "openai/model", "model_kwargs": {"api_base": "https://api"}}

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return "model"

    config = SimpleNamespace(to_minisweagent_config=lambda: expected)

    assert runner._initialize_agent_model(factory, config) == "model"
    assert calls == [((), {"config": expected})]


def test_runner_refuses_blocked_experiment_without_mutating_registry(
    tmp_path: Path, monkeypatch
) -> None:
    registry = tmp_path / "registry.csv"
    registry.write_text("sentinel\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        runner,
        "experiment_readiness",
        lambda **kwargs: {"ready": False, "blockers": ["model unresolved", "Docker unavailable"]},
    )

    result = runner.run_experiment(
        "swebench_lite_dev_pilot",
        registry_path=registry,
        audit_path=tmp_path / "audit.json",
        cost_cap_usd=None,
    )

    assert result["status"] == "blocked"
    assert result["executed_runs"] == 0
    assert registry.read_text(encoding="utf-8") == "sentinel\n"
    blocked = json.loads(
        (tmp_path / "results/runs/swebench_lite_dev_pilot/BLOCKED.json").read_text(encoding="utf-8")
    )
    assert blocked["blockers"] == ["model unresolved", "Docker unavailable"]


def test_safe_manifest_rejects_evaluator_only_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"tasks": [{"task_id": "task", "problem_statement": "x", "patch": "gold"}]}),
        encoding="utf-8",
    )

    try:
        runner._safe_manifest_task(manifest, "task")
    except ValueError as error:
        assert "evaluator-only" in str(error)
    else:
        raise AssertionError("unsafe manifest was accepted")


def test_infrastructure_failure_cost_is_written_to_event_not_registry(monkeypatch) -> None:
    registry_calls = []
    event_calls = []
    monkeypatch.setattr(
        runner,
        "update_run_status",
        lambda *args, **kwargs: registry_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        runner,
        "write_status_event",
        lambda *args, **kwargs: event_calls.append((args, kwargs)),
    )

    runner._record_infrastructure_failure(
        SimpleNamespace(run_id="run-a"),
        registry_path="registry.csv",
        run_dir=Path("run-a"),
        reason="transport failed",
        model_cost_usd=1.25,
    )

    assert "model_cost_usd" not in registry_calls[0][1]
    assert event_calls[0][1]["model_cost_usd"] == 1.25


def test_global_budget_limits_final_run_and_accounts_for_unstarted_cells(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    stale_blocker = tmp_path / "results/runs/swebench_lite_dev_pilot/BLOCKED.json"
    stale_blocker.parent.mkdir(parents=True)
    stale_blocker.write_text("{}\n", encoding="utf-8")
    cells = [
        SimpleNamespace(
            run_id=f"run-{index}",
            task_id=f"task-{index}",
            artifact_path=tmp_path / f"run-{index}",
        )
        for index in range(3)
    ]
    limits = []
    status_updates = []
    events = []
    monkeypatch.setattr(
        runner,
        "experiment_readiness",
        lambda **kwargs: {"ready": True, "blockers": [], "observed_cost_usd": 1.0},
    )
    monkeypatch.setattr(runner, "load_registry", lambda path: cells)
    monkeypatch.setattr(runner, "select_run_cells", lambda *args, **kwargs: cells)
    monkeypatch.setattr(
        runner,
        "_run_cell",
        lambda cell, **kwargs: (
            limits.append(kwargs["cost_limit_usd"]) or {"model_cost_usd": kwargs["cost_limit_usd"]}
        ),
    )
    monkeypatch.setattr(
        runner,
        "update_run_status",
        lambda *args, **kwargs: status_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        runner,
        "write_status_event",
        lambda *args, **kwargs: events.append((args, kwargs)),
    )
    monkeypatch.setattr(runner, "_refresh_audit", lambda *args, **kwargs: {"status": "pass"})

    result = runner.run_experiment(
        "pilot",
        registry_path=tmp_path / "registry.csv",
        audit_path=tmp_path / "audit.json",
        cost_cap_usd=3.0,
        per_run_cost_cap_usd=1.5,
        mark_budget_exhausted_skipped=True,
    )

    assert limits == [1.5, 0.5]
    assert result["executed_runs"] == 2
    assert result["skipped_runs"] == 1
    assert result["budget_exhausted"] is True
    assert status_updates[0][0][1:] == ("run-2", "skipped")
    assert status_updates[0][1]["failure_type"] == "budget"
    assert events[0][1]["failure_type"] == "budget"
    assert not stale_blocker.exists()
