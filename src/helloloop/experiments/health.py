from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from helloloop.benchmarks.swebench_eval import docker_preflight
from helloloop.models import load_model_config, model_preflight
from helloloop.schema import read_trajectory

from .registry import TERMINAL_STATUSES, audit_registry, load_registry


def experiment_readiness(
    *,
    registry_path: str | Path = "results/run_registry.csv",
    cost_cap_usd: float | None = None,
    required_memory_gib: float = 16.0,
    required_disk_gib: float = 120.0,
) -> dict[str, Any]:
    cells = load_registry(registry_path)
    registry_audit = audit_registry(cells)
    model_configs = sorted({cell.model_config for cell in cells})
    model_checks = {path: model_preflight(load_model_config(path)) for path in model_configs}
    evaluator_check = docker_preflight(
        Path(registry_path).parent,
        required_memory_gib=required_memory_gib,
        required_disk_gib=required_disk_gib,
    )

    missing_artifacts = []
    corrupted_trajectories = []
    observed_cost = 0.0
    cost_known = True
    for cell in cells:
        if cell.status not in TERMINAL_STATUSES:
            continue
        run_dir = Path(cell.artifact_path)
        if cell.status == "complete":
            required = ("trajectory.jsonl", "summary.json", "patch.diff", "prediction.jsonl")
            for filename in required:
                if not (run_dir / filename).exists():
                    missing_artifacts.append(f"{cell.run_id}/{filename}")
        trajectory = run_dir / "trajectory.jsonl"
        if trajectory.exists():
            try:
                read_trajectory(trajectory)
            except (OSError, ValueError, json.JSONDecodeError):
                corrupted_trajectories.append(cell.run_id)
        summary = run_dir / "summary.json"
        if summary.exists():
            data = json.loads(summary.read_text(encoding="utf-8"))
            cost = data.get("estimated_cost_usd")
            if cost is None:
                cost_known = False
            else:
                observed_cost += float(cost)
        else:
            event = run_dir / "status_event.json"
            if event.exists():
                observed_cost += float(
                    json.loads(event.read_text(encoding="utf-8")).get("model_cost_usd", 0.0)
                )

    blockers = []
    if registry_audit["status"] != "pass":
        blockers.append("run registry audit failed")
    for path, check in model_checks.items():
        blockers.extend(f"{path}: {blocker}" for blocker in check["blockers"])
        config = load_model_config(path)
        if (
            config.input_price_per_million_usd is None
            or config.output_price_per_million_usd is None
        ):
            blockers.append(f"{path}: verified input and output prices are required")
    blockers.extend(f"official evaluator: {blocker}" for blocker in evaluator_check.blockers)
    if missing_artifacts:
        blockers.append(f"{len(missing_artifacts)} required artifacts missing")
    if corrupted_trajectories:
        blockers.append(f"{len(corrupted_trajectories)} trajectories corrupted")
    if cost_cap_usd is None:
        blockers.append("maximum cost cap unresolved")
    if cost_cap_usd is not None and not cost_known:
        blockers.append("observed cost is unknown for one or more completed runs")
    if cost_cap_usd is not None and cost_known and observed_cost >= cost_cap_usd:
        blockers.append(f"cost cap reached: {observed_cost:.4f} >= {cost_cap_usd:.4f}")

    return {
        "schema_id": "helloloop.experiment_readiness.v1",
        "ready": not blockers,
        "blockers": blockers,
        "api_quota": "unchecked_without_authenticated_probe",
        "registry": registry_audit,
        "model_checks": model_checks,
        "official_evaluator": evaluator_check.to_dict(),
        "status_counts": dict(sorted(Counter(cell.status for cell in cells).items())),
        "missing_artifacts": missing_artifacts,
        "corrupted_trajectories": corrupted_trajectories,
        "observed_cost_usd": round(observed_cost, 8) if cost_known else None,
        "cost_cap_usd": cost_cap_usd,
    }


def write_experiment_readiness(
    path: str | Path,
    *,
    registry_path: str | Path = "results/run_registry.csv",
    cost_cap_usd: float | None = None,
    required_memory_gib: float = 16.0,
    required_disk_gib: float = 120.0,
) -> dict[str, Any]:
    result = experiment_readiness(
        registry_path=registry_path,
        cost_cap_usd=cost_cap_usd,
        required_memory_gib=required_memory_gib,
        required_disk_gib=required_disk_gib,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
