from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from helloloop.experiments.registry import RunCell, load_registry, write_registry
from helloloop.experiments.shards import (
    merge_shard_artifacts,
    read_run_ids_file,
    select_run_cells,
    write_status_event,
)


def _cell(order: int, task: str, loop: str = "linear") -> RunCell:
    run_id = f"experiment__{task}__{loop}__primary_model__r0"
    return RunCell(
        phase="phase12",
        experiment="swebench_lite_dev_pilot",
        order=order,
        run_id=run_id,
        benchmark="swebench_lite_dev",
        manifest_path="manifest.json",
        task_id=task,
        loop_variant=loop,
        model_config="model.yaml",
        model_name="model",
        repetition_id=0,
        seed=20260711,
        created_at="2026-07-19T00:00:00+00:00",
        updated_at="2026-07-19T00:00:00+00:00",
        artifact_path=f"results/runs/experiment/{run_id}",
    )


def _cells() -> list[RunCell]:
    cells = []
    for task in ("task-a", "task-b", "task-c", "task-d"):
        for loop in ("linear", "verification"):
            cells.append(_cell(len(cells), task, loop))
    return cells


def test_task_stable_shards_keep_paired_loops_together() -> None:
    cells = _cells()
    shard_zero = select_run_cells(cells, "pilot", shard_index=0, shard_count=2)
    shard_one = select_run_cells(cells, "pilot", shard_index=1, shard_count=2)

    assert {cell.task_id for cell in shard_zero} == {"task-a", "task-c"}
    assert {cell.task_id for cell in shard_one} == {"task-b", "task-d"}
    assert {cell.run_id for cell in shard_zero}.isdisjoint(cell.run_id for cell in shard_one)


def test_run_selection_validates_ids_and_applies_limit_before_sharding() -> None:
    cells = _cells()
    selected = select_run_cells(cells, "pilot", limit=4, shard_index=1, shard_count=2)
    assert {cell.task_id for cell in selected} == {"task-b"}

    with pytest.raises(ValueError, match="do not belong"):
        select_run_cells(cells, "pilot", run_ids={"missing"})


def test_resumed_limit_counts_only_eligible_cells_without_changing_shard_ownership() -> None:
    cells = [
        replace(
            cell,
            status="complete",
            terminal_classification="submitted_patch",
        )
        if cell.task_id == "task-a"
        else cell
        for cell in _cells()
    ]

    shard_zero = select_run_cells(cells, "pilot", limit=4, shard_index=0, shard_count=2)
    shard_one = select_run_cells(cells, "pilot", limit=4, shard_index=1, shard_count=2)

    assert {cell.task_id for cell in shard_zero} == {"task-c"}
    assert {cell.task_id for cell in shard_one} == {"task-b"}
    assert len(shard_zero) + len(shard_one) == 4


def test_shard_ownership_does_not_change_when_only_one_shard_is_complete() -> None:
    cells = [
        replace(
            cell,
            status="complete",
            terminal_classification="submitted_patch",
        )
        if cell.task_id in {"task-b", "task-d"}
        else cell
        for cell in _cells()
    ]
    selected = select_run_cells(
        cells,
        "pilot",
        shard_index=1,
        shard_count=2,
        statuses=frozenset({"complete"}),
    )
    assert {cell.task_id for cell in selected} == {"task-b", "task-d"}


def test_run_id_files_support_json_and_newline_formats(tmp_path: Path) -> None:
    json_path = tmp_path / "runs.json"
    json_path.write_text(json.dumps(["run-a", "run-b"]), encoding="utf-8")
    text_path = tmp_path / "runs.txt"
    text_path.write_text("run-a\n\nrun-b\n", encoding="utf-8")

    assert read_run_ids_file(json_path) == {"run-a", "run-b"}
    assert read_run_ids_file(text_path) == {"run-a", "run-b"}


def test_merge_updates_registry_and_requires_complete_artifacts(tmp_path: Path) -> None:
    cell = _cell(0, "task-a")
    registry = tmp_path / "registry.csv"
    write_registry([cell], registry)
    run_dir = tmp_path / "artifacts" / cell.run_id
    run_dir.mkdir(parents=True)
    for name in ("trajectory.jsonl", "summary.json", "patch.diff", "prediction.jsonl"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    write_status_event(run_dir, cell, status="complete", model_cost_usd=0.25)

    report = merge_shard_artifacts(
        registry_path=registry,
        artifact_roots=[tmp_path / "artifacts"],
        audit_path=tmp_path / "audit.json",
        report_path=tmp_path / "merge.json",
    )

    assert report["runs_merged"] == 1
    assert load_registry(registry)[0].status == "complete"


def test_merge_advances_official_evaluation_for_terminal_run(tmp_path: Path) -> None:
    cell = replace(
        _cell(0, "task-a"),
        status="complete",
        terminal_classification="submitted_patch",
    )
    registry = tmp_path / "registry.csv"
    write_registry([cell], registry)
    run_dir = tmp_path / "artifacts" / cell.run_id
    run_dir.mkdir(parents=True)
    for name in ("trajectory.jsonl", "summary.json", "patch.diff", "prediction.jsonl"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    evaluated = replace(
        cell,
        official_evaluation_status="complete",
        official_resolved="true",
    )
    write_status_event(run_dir, evaluated, status="complete", model_cost_usd=0.25)

    report = merge_shard_artifacts(
        registry_path=registry,
        artifact_roots=[tmp_path / "artifacts"],
        audit_path=tmp_path / "audit.json",
        report_path=tmp_path / "merge.json",
    )

    merged = load_registry(registry)[0]
    assert report["runs_merged"] == 0
    assert report["official_evaluations_updated"] == 1
    assert merged.official_evaluation_status == "complete"
    assert merged.official_resolved == "true"


def test_merge_accepts_successful_official_evaluation_retry(tmp_path: Path) -> None:
    failed = replace(
        _cell(0, "task-a"),
        status="complete",
        terminal_classification="submitted_patch",
        official_evaluation_status="failed",
        official_evaluation_reason="official evaluator preflight failed",
    )
    registry = tmp_path / "registry.csv"
    write_registry([failed], registry)
    first = tmp_path / "first" / failed.run_id
    second = tmp_path / "second" / failed.run_id
    for run_dir in (first, second):
        run_dir.mkdir(parents=True)
        for name in ("trajectory.jsonl", "summary.json", "patch.diff", "prediction.jsonl"):
            (run_dir / name).write_text("{}\n", encoding="utf-8")
    write_status_event(first, failed, status="complete", model_cost_usd=0.25)
    evaluated = replace(
        failed,
        official_evaluation_status="complete",
        official_resolved="false",
        official_evaluation_reason="",
    )
    write_status_event(second, evaluated, status="complete", model_cost_usd=0.25)

    report = merge_shard_artifacts(
        registry_path=registry,
        artifact_roots=[first.parent, second.parent],
        audit_path=tmp_path / "audit.json",
        report_path=tmp_path / "merge.json",
    )

    merged = load_registry(registry)[0]
    assert report["official_evaluations_updated"] == 1
    assert merged.official_evaluation_status == "complete"
    assert merged.official_resolved == "false"


def test_merge_ignores_and_synchronizes_stale_official_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "canonical" / "run-a"
    current = replace(
        _cell(0, "task-a"),
        status="complete",
        artifact_path=str(run_dir),
        terminal_classification="submitted_patch",
        official_evaluation_status="complete",
        official_resolved="false",
    )
    registry = tmp_path / "registry.csv"
    write_registry([current], registry)
    run_dir.mkdir(parents=True)
    for name in ("trajectory.jsonl", "summary.json", "patch.diff", "prediction.jsonl"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    stale = replace(
        current,
        official_evaluation_status="failed",
        official_resolved="",
        official_evaluation_reason="official evaluator preflight failed",
    )
    write_status_event(run_dir, stale, status="complete", model_cost_usd=0.25)

    report = merge_shard_artifacts(
        registry_path=registry,
        artifact_roots=[run_dir.parent],
        audit_path=tmp_path / "audit.json",
        report_path=tmp_path / "merge.json",
    )

    merged = load_registry(registry)[0]
    event = json.loads((run_dir / "status_event.json").read_text())
    assert report["official_evaluations_updated"] == 0
    assert report["canonical_status_events_synced"] == 1
    assert merged.official_evaluation_status == "complete"
    assert merged.official_resolved == "false"
    assert event["official_evaluation_status"] == "complete"
    assert event["official_resolved"] == "false"


def test_merge_rejects_conflicting_terminal_events(tmp_path: Path) -> None:
    cell = _cell(0, "task-a")
    registry = tmp_path / "registry.csv"
    write_registry([cell], registry)
    first = tmp_path / "first" / cell.run_id
    second = tmp_path / "second" / cell.run_id
    write_status_event(
        first,
        cell,
        status="failed",
        reason="agent failure",
        failure_type="agent",
    )
    write_status_event(
        second,
        cell,
        status="skipped",
        reason="budget exhausted",
        failure_type="budget",
    )

    with pytest.raises(ValueError, match="conflicting status events"):
        merge_shard_artifacts(
            registry_path=registry,
            artifact_roots=[first.parent, second.parent],
            audit_path=tmp_path / "audit.json",
            report_path=tmp_path / "merge.json",
        )


def test_merge_can_explicitly_account_for_unreported_budget_cells(tmp_path: Path) -> None:
    cells = _cells()
    registry = tmp_path / "registry.csv"
    write_registry(cells, registry)
    first = cells[0]
    event_dir = tmp_path / "artifacts" / first.run_id
    write_status_event(
        event_dir,
        first,
        status="failed",
        reason="agent failure",
        failure_type="agent",
    )

    report = merge_shard_artifacts(
        registry_path=registry,
        artifact_roots=[tmp_path / "artifacts"],
        audit_path=tmp_path / "audit.json",
        report_path=tmp_path / "merge.json",
        experiment="pilot",
        skip_unreported=True,
        skip_reason="approved experiment budget exhausted",
    )

    merged = load_registry(registry)
    assert report["runs_marked_skipped"] == len(cells) - 1
    assert merged[0].status == "failed"
    assert all(cell.status == "skipped" for cell in merged[1:])
    assert all(cell.failure_type == "budget" for cell in merged[1:])


def test_merge_refuses_to_replace_terminal_registry_state(tmp_path: Path) -> None:
    cell = replace(
        _cell(0, "task-a"),
        status="failed",
        reason="original failure",
        failure_type="agent",
        official_evaluation_status="not_applicable",
        terminal_classification="agent_failure",
    )
    registry = tmp_path / "registry.csv"
    write_registry([cell], registry)
    event_dir = tmp_path / "artifacts" / cell.run_id
    write_status_event(
        event_dir,
        cell,
        status="skipped",
        reason="new state",
        failure_type="budget",
    )

    with pytest.raises(ValueError, match="replace terminal registry state"):
        merge_shard_artifacts(
            registry_path=registry,
            artifact_roots=[tmp_path / "artifacts"],
            audit_path=tmp_path / "audit.json",
            report_path=tmp_path / "merge.json",
        )


def test_merge_accepts_audited_export_only_infrastructure_recovery(tmp_path: Path) -> None:
    reason = "ValueError: prediction contains secret-like material"
    failed = replace(
        _cell(0, "task-a"),
        status="failed",
        reason=reason,
        failure_type="infrastructure",
        official_evaluation_status="not_applicable",
        terminal_classification="infrastructure_failure",
    )
    registry = tmp_path / "registry.csv"
    write_registry([failed], registry)
    run_dir = tmp_path / "artifacts" / failed.run_id
    run_dir.mkdir(parents=True)
    for name in ("trajectory.jsonl", "summary.json", "patch.diff", "prediction.jsonl"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    patch_sha256 = hashlib.sha256((run_dir / "patch.diff").read_bytes()).hexdigest()
    recovered = replace(
        failed,
        status="complete",
        reason="",
        failure_type="",
        official_evaluation_status="pending",
        terminal_classification="submitted_patch",
    )
    write_status_event(
        run_dir,
        recovered,
        status="complete",
        recovery={
            "method": "submission_export_without_model_call",
            "patch_sha256": patch_sha256,
            "prior_status": "failed",
            "prior_reason": reason,
            "prior_failure_type": "infrastructure",
        },
    )

    report = merge_shard_artifacts(
        registry_path=registry,
        artifact_roots=[tmp_path / "artifacts"],
        audit_path=tmp_path / "audit.json",
        report_path=tmp_path / "merge.json",
    )

    assert report["runs_recovered_without_model_call"] == 1
    assert load_registry(registry)[0].status == "complete"
