from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .registry import TERMINAL_STATUSES, RunCell, audit_registry, load_registry, write_registry

EXPERIMENT_ALIASES = {
    "pilot": "swebench_lite_dev_pilot",
    "main": "swebench_lite_main",
    "holdout": "swebench_verified_holdout",
    "replication": "swebench_lite_replication",
}
STATUS_EVENT_SCHEMA_ID = "helloloop.run_status_event.v1"


def resolve_experiment_name(value: str) -> str:
    return EXPERIMENT_ALIASES.get(value, value)


def read_run_ids_file(path: str | Path) -> set[str]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError("run ID JSON must be an array of strings")
        values = data
    else:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError("run ID selection contains duplicates")
    return set(values)


def select_run_cells(
    cells: list[RunCell],
    experiment: str,
    *,
    run_ids: set[str] | None = None,
    limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    statuses: frozenset[str] = frozenset({"planned"}),
) -> list[RunCell]:
    experiment_name = resolve_experiment_name(experiment)
    if shard_count < 1:
        raise ValueError("shard count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard index must be in [0, shard_count)")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    experiment_cells = [cell for cell in cells if cell.experiment == experiment_name]
    if not experiment_cells:
        raise ValueError(f"unknown experiment: {experiment}")
    if run_ids is not None:
        available = {cell.run_id for cell in experiment_cells}
        unknown = sorted(run_ids - available)
        if unknown:
            raise ValueError(f"run IDs do not belong to {experiment_name}: {unknown}")
    candidates = [cell for cell in experiment_cells if run_ids is None or cell.run_id in run_ids]
    eligible = [cell for cell in candidates if cell.status in statuses]
    if limit is not None:
        eligible = eligible[:limit]

    task_order = list(dict.fromkeys(cell.task_id for cell in candidates))
    shard_tasks = {
        task_id
        for task_position, task_id in enumerate(task_order)
        if task_position % shard_count == shard_index
    }
    return [cell for cell in eligible if cell.task_id in shard_tasks]


def write_status_event(
    run_dir: str | Path,
    cell: RunCell,
    *,
    status: str,
    reason: str = "",
    failure_type: str = "",
    model_cost_usd: float = 0.0,
    terminal_classification: str = "",
    recovery: dict[str, str] | None = None,
) -> Path:
    completed_at = datetime.now(UTC).isoformat()
    official_status = cell.official_evaluation_status
    official_resolved = cell.official_resolved
    official_reason = cell.official_evaluation_reason
    if status in {"failed", "skipped"}:
        official_status = "not_applicable"
        official_resolved = ""
        official_reason = ""
    if not terminal_classification:
        if status == "complete":
            terminal_classification = cell.terminal_classification or "submitted_patch"
        else:
            terminal_classification = {
                "agent": "agent_failure",
                "infrastructure": "infrastructure_failure",
                "budget": "budget_stop",
                "policy": "policy_stop",
            }.get(failure_type, "")
    validated = replace(
        cell,
        status=status,
        reason=reason,
        failure_type=failure_type,
        updated_at=completed_at,
        official_evaluation_status=official_status,
        official_resolved=official_resolved,
        official_evaluation_reason=official_reason,
        terminal_classification=terminal_classification,
    )
    validated.validate()
    target = Path(run_dir) / "status_event.json"
    if recovery is None and target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        recovery = previous.get("recovery")
    payload: dict[str, Any] = {
        "schema_id": STATUS_EVENT_SCHEMA_ID,
        "run_id": cell.run_id,
        "experiment": cell.experiment,
        "task_id": cell.task_id,
        "status": status,
        "reason": reason,
        "failure_type": failure_type,
        "model_cost_usd": round(float(model_cost_usd), 8),
        "completed_at": completed_at,
        "official_evaluation_status": validated.official_evaluation_status,
        "official_resolved": validated.official_resolved,
        "official_evaluation_reason": validated.official_evaluation_reason,
        "terminal_classification": validated.terminal_classification,
    }
    if recovery is not None:
        payload["recovery"] = recovery
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _load_events(artifact_roots: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    for root in sorted(Path(path) for path in artifact_roots):
        for path in sorted(root.glob("**/status_event.json")):
            event = json.loads(path.read_text(encoding="utf-8"))
            if event.get("schema_id") != STATUS_EVENT_SCHEMA_ID:
                raise ValueError(f"unsupported status event schema in {path}")
            run_id = str(event.get("run_id", ""))
            if not run_id:
                raise ValueError(f"status event has no run ID: {path}")
            if event.get("status") not in TERMINAL_STATUSES:
                raise ValueError(f"status event is not terminal: {path}")
            base_state = (
                event.get("status"),
                event.get("reason", ""),
                event.get("failure_type", ""),
                event.get("model_cost_usd", 0.0),
                event.get("terminal_classification", ""),
            )
            official_state = (
                event.get("official_evaluation_status", "pending"),
                event.get("official_resolved", ""),
                event.get("official_evaluation_reason", ""),
            )
            if run_id in events:
                existing = events[run_id]
                existing_base = (
                    existing.get("status"),
                    existing.get("reason", ""),
                    existing.get("failure_type", ""),
                    existing.get("model_cost_usd", 0.0),
                    existing.get("terminal_classification", ""),
                )
                existing_official = (
                    existing.get("official_evaluation_status", "pending"),
                    existing.get("official_resolved", ""),
                    existing.get("official_evaluation_reason", ""),
                )
                if existing_base != base_state:
                    raise ValueError(f"conflicting status events for {run_id}")
                if existing_official != official_state:
                    official_rank = {"pending": 0, "failed": 1, "complete": 2}
                    existing_rank = official_rank[str(existing_official[0])]
                    incoming_rank = official_rank[str(official_state[0])]
                    if existing_rank == incoming_rank:
                        raise ValueError(f"conflicting status events for {run_id}")
                    if incoming_rank > existing_rank:
                        events[run_id] = event
                    continue
            if run_id not in events or str(event["completed_at"]) > str(
                events[run_id]["completed_at"]
            ):
                events[run_id] = event
            if event["status"] == "complete":
                missing = [
                    name
                    for name in (
                        "trajectory.jsonl",
                        "summary.json",
                        "patch.diff",
                        "prediction.jsonl",
                    )
                    if not (path.parent / name).exists()
                ]
                if missing:
                    raise ValueError(f"complete event for {run_id} is missing artifacts: {missing}")
            recovery = event.get("recovery")
            if recovery is not None:
                patch_path = path.parent / "patch.diff"
                actual_patch_sha256 = hashlib.sha256(patch_path.read_bytes()).hexdigest()
                if not (
                    event["status"] == "complete"
                    and recovery.get("method") == "submission_export_without_model_call"
                    and recovery.get("patch_sha256") == actual_patch_sha256
                ):
                    raise ValueError(f"invalid recovery evidence for {run_id}")
    return events


def merge_shard_artifacts(
    *,
    registry_path: str | Path,
    artifact_roots: Iterable[str | Path],
    output_registry_path: str | Path | None = None,
    audit_path: str | Path = "results/run_audit.json",
    report_path: str | Path = "results/shard_merge_report.json",
    experiment: str | None = None,
    skip_unreported: bool = False,
    skip_reason: str = "",
) -> dict[str, Any]:
    cells = load_registry(registry_path)
    by_run_id = {cell.run_id: index for index, cell in enumerate(cells)}
    events = _load_events(artifact_roots)
    unknown = sorted(set(events) - set(by_run_id))
    if unknown:
        raise ValueError(f"status events are not present in the registry: {unknown}")

    merged = 0
    official_updates = 0
    recovered = 0
    for run_id, event in sorted(events.items(), key=lambda item: cells[by_run_id[item[0]]].order):
        index = by_run_id[run_id]
        current = cells[index]
        if current.status in TERMINAL_STATUSES:
            current_state = (current.status, current.reason, current.failure_type)
            event_state = (event["status"], event.get("reason", ""), event.get("failure_type", ""))
            if current_state != event_state:
                recovery = event.get("recovery", {})
                valid_recovery = (
                    current.status == "failed"
                    and current.failure_type == "infrastructure"
                    and event["status"] == "complete"
                    and recovery.get("method") == "submission_export_without_model_call"
                    and recovery.get("prior_status") == current.status
                    and recovery.get("prior_reason") == current.reason
                    and recovery.get("prior_failure_type") == current.failure_type
                )
                if not valid_recovery:
                    raise ValueError(f"refusing to replace terminal registry state for {run_id}")
                cells[index] = replace(
                    current,
                    status="complete",
                    reason="",
                    failure_type="",
                    updated_at=str(event["completed_at"]),
                    official_evaluation_status=str(
                        event.get("official_evaluation_status", "pending")
                    ),
                    official_resolved=str(event.get("official_resolved", "")),
                    official_evaluation_reason=str(event.get("official_evaluation_reason", "")),
                    terminal_classification=str(event["terminal_classification"]),
                )
                cells[index].validate()
                recovered += 1
                continue
            current_official = (
                current.official_evaluation_status,
                current.official_resolved,
                current.official_evaluation_reason,
            )
            event_official = (
                str(event.get("official_evaluation_status", "pending")),
                str(event.get("official_resolved", "")),
                str(event.get("official_evaluation_reason", "")),
            )
            if current_official == event_official:
                continue
            official_rank = {"pending": 0, "failed": 1, "complete": 2}
            current_rank = official_rank[current_official[0]]
            event_rank = official_rank[event_official[0]]
            if event_rank < current_rank:
                continue
            if event_rank == current_rank:
                raise ValueError(f"conflicting official evaluation states for {run_id}")
            cells[index] = replace(
                current,
                updated_at=str(event["completed_at"]),
                official_evaluation_status=event_official[0],
                official_resolved=event_official[1],
                official_evaluation_reason=event_official[2],
            )
            cells[index].validate()
            official_updates += 1
            continue
        cells[index] = replace(
            current,
            status=str(event["status"]),
            reason=str(event.get("reason", "")),
            failure_type=str(event.get("failure_type", "")),
            updated_at=str(event["completed_at"]),
            official_evaluation_status=str(event.get("official_evaluation_status", "pending")),
            official_resolved=str(event.get("official_resolved", "")),
            official_evaluation_reason=str(event.get("official_evaluation_reason", "")),
            terminal_classification=str(event.get("terminal_classification", "")),
        )
        cells[index].validate()
        merged += 1

    skipped = 0
    if skip_unreported:
        if not skip_reason.strip():
            raise ValueError("skip reason is required when unreported runs are marked skipped")
        experiment_name = resolve_experiment_name(experiment) if experiment else None
        now = datetime.now(UTC).isoformat()
        for index, cell in enumerate(cells):
            if cell.status != "planned" or cell.run_id in events:
                continue
            if experiment_name and cell.experiment != experiment_name:
                continue
            cells[index] = replace(
                cell,
                status="skipped",
                reason=skip_reason,
                failure_type="budget",
                updated_at=now,
                official_evaluation_status="not_applicable",
                official_resolved="",
                official_evaluation_reason="",
                terminal_classification="budget_stop",
            )
            skipped += 1

    output_registry = Path(output_registry_path or registry_path)
    write_registry(cells, output_registry)
    canonical_events_synced = 0
    for run_id, event in events.items():
        cell = cells[by_run_id[run_id]]
        run_dir = Path(cell.artifact_path)
        target = run_dir / "status_event.json"
        if not run_dir.is_dir() or not target.exists():
            continue
        current_event = json.loads(target.read_text(encoding="utf-8"))
        current_state = (
            current_event.get("status"),
            current_event.get("reason", ""),
            current_event.get("failure_type", ""),
            current_event.get("official_evaluation_status", "pending"),
            current_event.get("official_resolved", ""),
            current_event.get("official_evaluation_reason", ""),
            current_event.get("terminal_classification", ""),
        )
        canonical_state = (
            cell.status,
            cell.reason,
            cell.failure_type,
            cell.official_evaluation_status,
            cell.official_resolved,
            cell.official_evaluation_reason,
            cell.terminal_classification,
        )
        if current_state == canonical_state:
            continue
        write_status_event(
            run_dir,
            cell,
            status=cell.status,
            reason=cell.reason,
            failure_type=cell.failure_type,
            model_cost_usd=float(event.get("model_cost_usd", 0.0)),
            terminal_classification=cell.terminal_classification,
            recovery=event.get("recovery"),
        )
        canonical_events_synced += 1
    audit = audit_registry(cells)
    audit_target = Path(audit_path)
    audit_target.parent.mkdir(parents=True, exist_ok=True)
    audit_target.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "schema_id": "helloloop.shard_merge.v1",
        "registry": str(output_registry),
        "events_found": len(events),
        "runs_merged": merged,
        "official_evaluations_updated": official_updates,
        "runs_recovered_without_model_call": recovered,
        "runs_marked_skipped": skipped,
        "canonical_status_events_synced": canonical_events_synced,
        "audit": audit,
    }
    report_target = Path(report_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
