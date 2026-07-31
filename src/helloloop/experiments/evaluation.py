from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from helloloop.benchmarks.swebench_eval import (
    HarnessConfig,
    SWEbenchPrediction,
    copy_evaluator_artifacts,
    parse_instance_report,
    read_predictions_jsonl,
    run_official_evaluation,
    write_predictions_jsonl,
)
from helloloop.schema import redact_text

from .registry import audit_registry, load_registry, update_official_evaluation
from .runner import EXPERIMENT_DATASETS, _prune_docker
from .shards import resolve_experiment_name, select_run_cells, write_status_event


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_model_cost(run_dir: Path) -> float:
    event_path = run_dir / "status_event.json"
    if event_path.exists():
        return float(json.loads(event_path.read_text(encoding="utf-8")).get("model_cost_usd", 0.0))
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return float(
            json.loads(summary_path.read_text(encoding="utf-8")).get("estimated_cost_usd", 0.0)
        )
    return 0.0


def _prediction_for_cell(cell: Any) -> SWEbenchPrediction:
    path = Path(cell.artifact_path) / "prediction.jsonl"
    predictions = read_predictions_jsonl(path)
    if len(predictions) != 1 or predictions[0].instance_id != cell.task_id:
        raise ValueError(f"{cell.run_id} must contain exactly its own SWE-bench prediction")
    if predictions[0].model_name_or_path != cell.model_name:
        raise ValueError(f"{cell.run_id} prediction model does not match the registry")
    return predictions[0]


def _record_evaluation_state(
    registry_path: str | Path,
    cell: Any,
    *,
    status: str,
    resolved: bool | None = None,
    reason: str = "",
) -> Any:
    updated = update_official_evaluation(
        registry_path,
        cell.run_id,
        status,
        resolved=resolved,
        reason=reason,
    )
    write_status_event(
        updated.artifact_path,
        updated,
        status="complete",
        model_cost_usd=_read_model_cost(Path(updated.artifact_path)),
        terminal_classification=updated.terminal_classification,
    )
    return updated


def evaluate_experiment_runs(
    experiment: str,
    *,
    registry_path: str | Path = "results/run_registry.csv",
    audit_path: str | Path = "results/run_audit.json",
    output_root: str | Path = "results/official_evaluation",
    harness_root: str | Path = ".",
    run_ids: set[str] | None = None,
    limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    max_workers: int = 1,
    timeout_seconds: int = 1_800,
    cache_level: str = "env",
    required_memory_gib: float = 16.0,
    required_disk_gib: float = 120.0,
    batch_size: int | None = None,
    prune_docker_after_batch: bool = False,
) -> dict[str, Any]:
    experiment_name = resolve_experiment_name(experiment)
    if experiment_name not in EXPERIMENT_DATASETS:
        raise ValueError(f"unknown experiment: {experiment}")
    if batch_size is not None and batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    cells = select_run_cells(
        load_registry(registry_path),
        experiment_name,
        run_ids=run_ids,
        limit=limit,
        shard_index=shard_index,
        shard_count=shard_count,
        statuses=frozenset({"complete"}),
    )
    selected = [cell for cell in cells if cell.official_evaluation_status != "complete"]
    groups: dict[tuple[str, str, int], list[Any]] = defaultdict(list)
    for cell in selected:
        groups[(cell.loop_variant, cell.model_name, cell.repetition_id)].append(cell)

    dataset_name, split = EXPERIMENT_DATASETS[experiment_name]
    output = Path(output_root) / experiment_name / f"shard-{shard_index}-of-{shard_count}"
    totals = {"evaluated": 0, "resolved": 0, "unresolved": 0, "failed": 0}
    group_reports = []
    work_groups = []
    for key, group in sorted(groups.items()):
        size = batch_size or len(group)
        work_groups.extend(
            (key, group[start : start + size]) for start in range(0, len(group), size)
        )
    for (loop_variant, model_name, repetition_id), group in work_groups:
        predictions = []
        predictions_by_run: dict[str, SWEbenchPrediction] = {}
        prediction_errors: dict[str, str] = {}
        for cell in group:
            try:
                prediction = _prediction_for_cell(cell)
                predictions.append(prediction)
                predictions_by_run[cell.run_id] = prediction
            except (OSError, ValueError, json.JSONDecodeError) as error:
                safe_reason, _ = redact_text(str(error), max_chars=4_000)
                prediction_errors[cell.run_id] = f"prediction validation failed: {safe_reason}"
        for cell in group:
            if cell.run_id in prediction_errors:
                _record_evaluation_state(
                    registry_path,
                    cell,
                    status="failed",
                    reason=prediction_errors[cell.run_id],
                )
                totals["failed"] += 1
        evaluable = [cell for cell in group if cell.run_id not in prediction_errors]
        if not evaluable:
            continue

        batch_identity = "\n".join(sorted(cell.run_id for cell in evaluable))
        batch_id = hashlib.sha256(batch_identity.encode("utf-8")).hexdigest()[:12]
        group_dir = output / f"group-{loop_variant}-r{repetition_id}-{batch_id}"
        predictions_path = write_predictions_jsonl(predictions, group_dir / "predictions.jsonl")
        prediction_sha256 = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
        evaluator_run_id = (
            f"helloloop-{experiment_name}-{loop_variant}-r{repetition_id}-"
            f"s{shard_index}-{prediction_sha256[:12]}"
        )
        config = HarnessConfig(
            dataset_name=dataset_name,
            split=split,
            predictions_path=str(predictions_path.resolve()),
            run_id=evaluator_run_id,
            instance_ids=tuple(sorted(cell.task_id for cell in evaluable)),
            max_workers=max_workers,
            timeout_seconds=timeout_seconds,
            cache_level=cache_level,
            report_dir=str(group_dir.resolve()),
        )
        evaluator_error = ""
        exit_code: int | None = None
        try:
            process = run_official_evaluation(
                config,
                harness_root=harness_root,
                output_dir=group_dir,
                required_memory_gib=required_memory_gib,
                required_disk_gib=required_disk_gib,
            )
            exit_code = process.returncode
            if exit_code != 0:
                evaluator_error = f"official evaluator exited with code {exit_code}"
        except (OSError, RuntimeError, ValueError) as error:
            evaluator_error, _ = redact_text(str(error), max_chars=4_000)

        group_result = {
            "run_id": evaluator_run_id,
            "loop_variant": loop_variant,
            "model_name": model_name,
            "repetition_id": repetition_id,
            "prediction_sha256": prediction_sha256,
            "instance_ids": sorted(cell.task_id for cell in evaluable),
            "exit_code": exit_code,
            "evaluator_error": evaluator_error,
        }
        group_reports.append(group_result)
        _write_json(group_dir / "evaluation_group.json", group_result)

        for cell in evaluable:
            destination = Path(cell.artifact_path) / "official_evaluation"
            copied = copy_evaluator_artifacts(
                harness_root=harness_root,
                run_id=evaluator_run_id,
                model_name=model_name,
                instance_id=cell.task_id,
                destination=destination,
            )
            report_path = destination / "report.json"
            prediction = predictions_by_run[cell.run_id]
            if (
                not report_path.exists()
                and exit_code == 0
                and cell.terminal_classification == "abstention"
                and not prediction.model_patch.strip()
            ):
                _write_json(
                    destination / "official_evaluation.json",
                    {
                        "schema_id": "helloloop.official_evaluation.v1",
                        "run_id": cell.run_id,
                        "task_id": cell.task_id,
                        "evaluator_run_id": evaluator_run_id,
                        "prediction_sha256": prediction_sha256,
                        "resolved": False,
                        "empty_patch": True,
                        "copied_artifacts": copied,
                    },
                )
                _record_evaluation_state(
                    registry_path,
                    cell,
                    status="complete",
                    resolved=False,
                )
                totals["evaluated"] += 1
                totals["unresolved"] += 1
                continue
            try:
                if not report_path.exists():
                    raise ValueError(evaluator_error or "official instance report was not produced")
                resolved = parse_instance_report(report_path, cell.task_id)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                safe_reason, _ = redact_text(str(error), max_chars=4_000)
                _record_evaluation_state(
                    registry_path,
                    cell,
                    status="failed",
                    reason=safe_reason,
                )
                totals["failed"] += 1
                continue
            _write_json(
                destination / "official_evaluation.json",
                {
                    "schema_id": "helloloop.official_evaluation.v1",
                    "run_id": cell.run_id,
                    "task_id": cell.task_id,
                    "evaluator_run_id": evaluator_run_id,
                    "prediction_sha256": prediction_sha256,
                    "resolved": resolved,
                    "copied_artifacts": copied,
                },
            )
            _record_evaluation_state(
                registry_path,
                cell,
                status="complete",
                resolved=resolved,
            )
            totals["evaluated"] += 1
            totals["resolved" if resolved else "unresolved"] += 1
        if prune_docker_after_batch:
            _prune_docker(group_dir / "docker_prune.json")

    audit = audit_registry(load_registry(registry_path))
    _write_json(Path(audit_path), audit)
    result = {
        "schema_id": "helloloop.official_evaluation_batch.v1",
        "experiment": experiment_name,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "eligible_runs": len(cells),
        "selected_runs": len(selected),
        **totals,
        "groups": group_reports,
        "audit": audit,
    }
    _write_json(output / "evaluation_report.json", result)
    return result
