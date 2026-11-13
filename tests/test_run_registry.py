from __future__ import annotations

import json
from pathlib import Path

import pytest

from helloloop.enums import LoopVariant
from helloloop.experiments.registry import (
    ExperimentSpec,
    append_replication_plan,
    audit_registry,
    load_registry,
    plan_experiment,
    update_official_evaluation,
    update_run_status,
    write_registry,
)


def _fixture_files(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"tasks": [{"task_id": f"task-{index}"} for index in range(4)]}),
        encoding="utf-8",
    )
    model = tmp_path / "model.yaml"
    model.write_text("model: fixture-model\n", encoding="utf-8")
    return manifest, model


def test_plan_is_deterministic_paired_and_task_major(tmp_path: Path) -> None:
    manifest, model = _fixture_files(tmp_path)
    spec = ExperimentSpec(
        phase="phase-test",
        name="fixture",
        benchmark="fixture",
        manifest_path=str(manifest),
        loops=(LoopVariant.LINEAR, LoopVariant.VERIFICATION, LoopVariant.UNCERTAINTY_AWARE),
        model_configs=(str(model),),
    )
    first = plan_experiment(spec)
    second = plan_experiment(spec)
    assert [cell.run_id for cell in first] == [cell.run_id for cell in second]
    assert len(first) == 12
    assert audit_registry(first, (spec,))["status"] == "pass"
    for task_id in {cell.task_id for cell in first}:
        assert {cell.loop_variant for cell in first if cell.task_id == task_id} == {
            "linear",
            "verification",
            "uncertainty_aware",
        }


def test_registry_round_trip_and_status_validation(tmp_path: Path) -> None:
    manifest, model = _fixture_files(tmp_path)
    spec = ExperimentSpec(
        phase="phase-test",
        name="fixture",
        benchmark="fixture",
        manifest_path=str(manifest),
        loops=(LoopVariant.LINEAR,),
        model_configs=(str(model),),
    )
    path = write_registry(plan_experiment(spec), tmp_path / "registry.csv")
    cells = load_registry(path)
    updated = update_run_status(
        path, cells[0].run_id, "failed", reason="timeout", failure_type="infrastructure"
    )
    assert updated.status == "failed"
    with pytest.raises(ValueError, match="requires a reason"):
        update_run_status(path, cells[1].run_id, "skipped")


def test_registry_audit_detects_duplicate_run_ids(tmp_path: Path) -> None:
    manifest, model = _fixture_files(tmp_path)
    spec = ExperimentSpec(
        phase="phase-test",
        name="fixture",
        benchmark="fixture",
        manifest_path=str(manifest),
        loops=(LoopVariant.LINEAR,),
        model_configs=(str(model),),
    )
    cells = plan_experiment(spec)
    audit = audit_registry([cells[0], cells[0]])
    assert audit["status"] == "fail"
    assert "duplicate run IDs" in audit["violations"]


def test_official_evaluation_requires_boolean_and_completed_agent_run(tmp_path: Path) -> None:
    manifest, model = _fixture_files(tmp_path)
    spec = ExperimentSpec(
        phase="phase-test",
        name="fixture",
        benchmark="fixture",
        manifest_path=str(manifest),
        loops=(LoopVariant.LINEAR,),
        model_configs=(str(model),),
    )
    path = write_registry(plan_experiment(spec), tmp_path / "registry.csv")
    run_id = load_registry(path)[0].run_id

    with pytest.raises(ValueError, match="completed agent runs"):
        update_official_evaluation(path, run_id, "complete", resolved=True)

    update_run_status(path, run_id, "complete")
    with pytest.raises(ValueError, match="resolved boolean"):
        update_official_evaluation(path, run_id, "complete")

    updated = update_official_evaluation(path, run_id, "complete", resolved=False)
    assert updated.official_evaluation_status == "complete"
    assert updated.official_resolved == "false"


def test_replication_extension_is_paired_and_idempotent(tmp_path: Path) -> None:
    manifest, model = _fixture_files(tmp_path)
    base_spec = ExperimentSpec(
        phase="phase-test",
        name="base",
        benchmark="fixture",
        manifest_path=str(manifest),
        loops=(LoopVariant.LINEAR,),
        model_configs=(str(model),),
    )
    registry = write_registry(plan_experiment(base_spec), tmp_path / "registry.csv")
    arguments = {
        "registry_path": registry,
        "plan_path": tmp_path / "replication.json",
        "audit_path": tmp_path / "audit.json",
        "manifest_path": manifest,
        "model_config": model,
    }

    first = append_replication_plan(**arguments)
    second = append_replication_plan(**arguments)

    replication = [
        cell for cell in load_registry(registry) if cell.experiment == "swebench_lite_replication"
    ]
    assert first["added_runs"] == 8
    assert second["added_runs"] == 0
    assert len(replication) == 8
    assert {cell.repetition_id for cell in replication} == {1}
    with pytest.raises(ValueError, match="frozen design"):
        append_replication_plan(**arguments, seed=20260721)
    for task_id in {cell.task_id for cell in replication}:
        assert {cell.loop_variant for cell in replication if cell.task_id == task_id} == {
            "linear",
            "uncertainty_aware",
        }
