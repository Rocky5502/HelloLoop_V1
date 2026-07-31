from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

from helloloop.benchmarks.swebench_eval import SWEbenchPrediction, write_predictions_jsonl
from helloloop.experiments import evaluation
from helloloop.experiments.registry import RunCell, load_registry, write_registry
from helloloop.experiments.shards import write_status_event


def _completed_cell(tmp_path: Path) -> RunCell:
    run_dir = tmp_path / "runs" / "run-a"
    cell = RunCell(
        phase="phase12",
        experiment="swebench_lite_dev_pilot",
        order=0,
        run_id="run-a",
        benchmark="swebench_lite_dev",
        manifest_path="manifest.json",
        task_id="org__repo-1",
        loop_variant="linear",
        model_config="model.yaml",
        model_name="model/x",
        repetition_id=0,
        seed=20260711,
        status="complete",
        created_at="2026-07-19T00:00:00+00:00",
        updated_at="2026-07-19T00:00:00+00:00",
        artifact_path=str(run_dir),
        terminal_classification="submitted_patch",
    )
    run_dir.mkdir(parents=True)
    for filename in ("trajectory.jsonl", "summary.json", "patch.diff"):
        (run_dir / filename).write_text("{}\n", encoding="utf-8")
    write_predictions_jsonl(
        [SWEbenchPrediction(cell.task_id, "diff --git a/a b/a", cell.model_name)],
        run_dir / "prediction.jsonl",
    )
    write_status_event(run_dir, cell, status="complete", model_cost_usd=0.25)
    return cell


def test_official_evaluation_joins_report_and_resumes(tmp_path: Path, monkeypatch) -> None:
    cell = _completed_cell(tmp_path)
    registry = write_registry([cell], tmp_path / "registry.csv")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    def fake_copy(*, destination, instance_id, **kwargs):
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        (target / "report.json").write_text(
            json.dumps({instance_id: {"resolved": True}}),
            encoding="utf-8",
        )
        return ["report.json"]

    monkeypatch.setattr(evaluation, "run_official_evaluation", fake_run)
    monkeypatch.setattr(evaluation, "copy_evaluator_artifacts", fake_copy)
    result = evaluation.evaluate_experiment_runs(
        "pilot",
        registry_path=registry,
        audit_path=tmp_path / "audit.json",
        output_root=tmp_path / "evaluation",
        harness_root=tmp_path,
    )

    updated = load_registry(registry)[0]
    assert result["evaluated"] == 1
    assert result["resolved"] == 1
    assert updated.official_evaluation_status == "complete"
    assert updated.official_resolved == "true"
    event = json.loads((Path(cell.artifact_path) / "status_event.json").read_text())
    assert event["official_resolved"] == "true"
    batch_id = hashlib.sha256(cell.run_id.encode("utf-8")).hexdigest()[:12]
    group_dir = tmp_path / "evaluation/swebench_lite_dev_pilot/shard-0-of-1"
    assert (group_dir / f"group-linear-r0-{batch_id}/evaluation_group.json").exists()

    resumed = evaluation.evaluate_experiment_runs(
        "pilot",
        registry_path=registry,
        audit_path=tmp_path / "audit.json",
        output_root=tmp_path / "evaluation",
        harness_root=tmp_path,
    )
    assert resumed["selected_runs"] == 0
    assert resumed["evaluated"] == 0


def test_official_evaluation_records_missing_report_and_can_retry(
    tmp_path: Path, monkeypatch
) -> None:
    cell = _completed_cell(tmp_path)
    registry = write_registry([cell], tmp_path / "registry.csv")
    monkeypatch.setattr(
        evaluation,
        "run_official_evaluation",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=""
        ),
    )
    monkeypatch.setattr(evaluation, "copy_evaluator_artifacts", lambda **kwargs: [])

    first = evaluation.evaluate_experiment_runs(
        "pilot",
        registry_path=registry,
        audit_path=tmp_path / "audit.json",
        output_root=tmp_path / "evaluation",
        harness_root=tmp_path,
    )
    assert first["failed"] == 1
    assert load_registry(registry)[0].official_evaluation_status == "failed"

    def resolved_copy(*, destination, instance_id, **kwargs):
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        (target / "report.json").write_text(
            json.dumps({instance_id: {"resolved": False}}),
            encoding="utf-8",
        )
        return ["report.json"]

    monkeypatch.setattr(evaluation, "copy_evaluator_artifacts", resolved_copy)
    second = evaluation.evaluate_experiment_runs(
        "pilot",
        registry_path=registry,
        audit_path=tmp_path / "audit.json",
        output_root=tmp_path / "evaluation",
        harness_root=tmp_path,
    )
    assert second["unresolved"] == 1
    assert load_registry(registry)[0].official_evaluation_status == "complete"


def test_official_evaluation_records_empty_patch_as_unresolved(tmp_path: Path, monkeypatch) -> None:
    cell = replace(_completed_cell(tmp_path), terminal_classification="abstention")
    write_predictions_jsonl(
        [SWEbenchPrediction(cell.task_id, "", cell.model_name)],
        Path(cell.artifact_path) / "prediction.jsonl",
    )
    write_status_event(
        cell.artifact_path,
        cell,
        status="complete",
        model_cost_usd=0.25,
        terminal_classification="abstention",
    )
    registry = write_registry([cell], tmp_path / "registry.csv")
    monkeypatch.setattr(
        evaluation,
        "run_official_evaluation",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        ),
    )
    monkeypatch.setattr(evaluation, "copy_evaluator_artifacts", lambda **kwargs: [])

    result = evaluation.evaluate_experiment_runs(
        "pilot",
        registry_path=registry,
        audit_path=tmp_path / "audit.json",
        output_root=tmp_path / "evaluation",
        harness_root=tmp_path,
    )

    updated = load_registry(registry)[0]
    official = json.loads(
        (Path(cell.artifact_path) / "official_evaluation/official_evaluation.json").read_text()
    )
    assert result["failed"] == 0
    assert result["evaluated"] == 1
    assert result["unresolved"] == 1
    assert updated.official_evaluation_status == "complete"
    assert updated.official_resolved == "false"
    assert official["empty_patch"] is True
    assert official["resolved"] is False


def test_registry_rejects_completed_run_without_submission_classification(tmp_path: Path) -> None:
    cell = replace(_completed_cell(tmp_path), terminal_classification="")
    try:
        write_registry([cell], tmp_path / "registry.csv")
    except ValueError as error:
        assert "terminal classification" in str(error)
    else:
        raise AssertionError("unclassified terminal run was accepted")
