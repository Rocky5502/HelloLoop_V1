from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from helloloop.benchmarks import SWEbenchPrediction, write_predictions_jsonl

from .registry import audit_registry, load_registry, write_registry
from .shards import resolve_experiment_name, write_status_event

RECOVERABLE_REASON_SUFFIX = "contains secret-like material"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _returned_model_ids(trajectory: dict[str, Any]) -> list[str]:
    model_ids = set()
    for message in trajectory.get("messages", []):
        response = message.get("extra", {}).get("response", {})
        if isinstance(response, dict) and response.get("model"):
            model_ids.add(str(response["model"]))
    return sorted(model_ids)


def recover_generated_submissions(
    experiment: str,
    *,
    registry_path: str | Path = "results/run_registry.csv",
    audit_path: str | Path = "results/run_audit.json",
) -> dict[str, Any]:
    experiment_name = resolve_experiment_name(experiment)
    cells = load_registry(registry_path)
    recovered_ids: list[str] = []
    for index, cell in enumerate(cells):
        if not (
            cell.experiment == experiment_name
            and cell.status == "failed"
            and cell.failure_type == "infrastructure"
            and cell.reason.endswith(RECOVERABLE_REASON_SUFFIX)
        ):
            continue

        run_dir = Path(cell.artifact_path)
        summary_path = run_dir / "summary.json"
        upstream_path = run_dir / "upstream_trajectory.json"
        if not summary_path.is_file() or not upstream_path.is_file():
            raise ValueError(f"recoverable run {cell.run_id} is missing generation evidence")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
        info = upstream.get("info", {})
        if info.get("exit_status") != "Submitted" or summary.get("status") != "complete":
            raise ValueError(f"recoverable run {cell.run_id} has no completed submission")
        patch = str(info.get("submission", ""))
        if not patch.strip():
            raise ValueError(f"recoverable run {cell.run_id} has an empty submission")
        prediction = SWEbenchPrediction(cell.task_id, patch, cell.model_name)
        prediction.validate()
        returned_model_ids = _returned_model_ids(upstream)
        if returned_model_ids != [cell.model_name]:
            raise ValueError(
                f"recoverable run {cell.run_id} returned unexpected models: {returned_model_ids}"
            )

        (run_dir / "patch.diff").write_text(patch, encoding="utf-8")
        prediction_path = write_predictions_jsonl([prediction], run_dir / "prediction.jsonl")
        model_cost = float(summary.get("estimated_cost_usd", 0.0))
        patch_sha256 = hashlib.sha256(patch.encode("utf-8")).hexdigest()
        recovery = {
            "method": "submission_export_without_model_call",
            "patch_sha256": patch_sha256,
            "prior_failure_type": cell.failure_type,
            "prior_reason": cell.reason,
            "prior_status": cell.status,
        }
        _write_json(
            run_dir / "run_result.json",
            {
                "configured_model": cell.model_name,
                "exit_status": "Submitted",
                "loop_variant": cell.loop_variant,
                "model_cost_usd": model_cost,
                "official_evaluation_status": "pending",
                "prediction_path": str(prediction_path),
                "recovered_without_model_call": True,
                "recovery_patch_sha256": patch_sha256,
                "returned_model_ids": returned_model_ids,
                "run_id": cell.run_id,
                "schema_id": "helloloop.agent_run.v1",
                "task_id": cell.task_id,
            },
        )
        updated = replace(
            cell,
            status="complete",
            reason="",
            failure_type="",
            official_evaluation_status="pending",
            official_resolved="",
            official_evaluation_reason="",
            terminal_classification="submitted_patch",
        )
        updated.validate()
        cells[index] = updated
        write_status_event(
            run_dir,
            updated,
            status="complete",
            model_cost_usd=model_cost,
            terminal_classification="submitted_patch",
            recovery=recovery,
        )
        recovered_ids.append(cell.run_id)

    write_registry(cells, registry_path)
    audit = audit_registry(cells)
    _write_json(Path(audit_path), audit)
    return {
        "schema_id": "helloloop.submission_recovery.v1",
        "experiment": experiment_name,
        "recovered_runs": len(recovered_ids),
        "recovered_run_ids": recovered_ids,
        "model_calls": 0,
    }
