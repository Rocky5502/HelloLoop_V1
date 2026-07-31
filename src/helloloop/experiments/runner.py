from __future__ import annotations

import json
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any

from helloloop.adapters import HelloLoopAgent
from helloloop.benchmarks import SWEbenchPrediction, write_predictions_jsonl
from helloloop.enums import LoopVariant
from helloloop.models import load_model_config
from helloloop.schema import TaskSpec, redact_text

from .health import experiment_readiness
from .registry import audit_registry, load_registry, update_run_status
from .shards import resolve_experiment_name, select_run_cells, write_status_event

EXPERIMENT_DATASETS = {
    "swebench_lite_dev_pilot": ("princeton-nlp/SWE-bench_Lite", "dev"),
    "swebench_lite_main": ("princeton-nlp/SWE-bench_Lite", "test"),
    "swebench_verified_holdout": ("princeton-nlp/SWE-bench_Verified", "test"),
    "swebench_lite_replication": ("princeton-nlp/SWE-bench_Lite", "test"),
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_manifest_task(manifest_path: str | Path, task_id: str) -> dict[str, Any]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    matches = [task for task in data.get("tasks", []) if task.get("task_id") == task_id]
    if len(matches) != 1:
        raise ValueError(f"expected one safe manifest record for {task_id}, found {len(matches)}")
    task = matches[0]
    forbidden = {"hints_text", "FAIL_TO_PASS", "PASS_TO_PASS", "patch", "test_patch"}
    exposed = forbidden.intersection(task)
    if exposed:
        raise ValueError(f"safe manifest exposes evaluator-only fields: {sorted(exposed)}")
    return task


def _measured_or_reserved_failure_cost(
    run_dir: Path,
    model_config: Any,
    agent: Any,
    reserved_cost_usd: float,
) -> tuple[float, str]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_cost = (
            int(summary.get("total_input_tokens", 0))
            * float(model_config.input_price_per_million_usd)
            + int(summary.get("total_output_tokens", 0))
            * float(model_config.output_price_per_million_usd)
        ) / 1_000_000
        summary["estimated_cost_usd"] = model_cost
        _write_json(summary_path, summary)
        return model_cost, "measured_from_tokens"
    if agent is not None and getattr(agent, "cost", None) is not None:
        return float(agent.cost), "measured_by_agent_runtime"
    return reserved_cost_usd, "reserved_after_unknown_failure"


def _record_infrastructure_failure(
    cell: Any,
    *,
    registry_path: str | Path,
    run_dir: Path,
    reason: str,
    model_cost_usd: float,
) -> None:
    update_run_status(
        registry_path,
        cell.run_id,
        "failed",
        reason=reason,
        failure_type="infrastructure",
    )
    write_status_event(
        run_dir,
        cell,
        status="failed",
        reason=reason,
        failure_type="infrastructure",
        model_cost_usd=model_cost_usd,
    )


def _prune_docker(path: Path) -> dict[str, Any]:
    docker = shutil.which("docker")
    if docker is None:
        result = {"attempted": False, "reason": "docker executable unavailable"}
    else:
        process = subprocess.run(
            [docker, "system", "prune", "--all", "--force"],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output, _ = redact_text(process.stdout + process.stderr, max_chars=20_000)
        result = {
            "attempted": True,
            "exit_code": process.returncode,
            "output": output,
        }
    _write_json(path, result)
    return result


def _refresh_audit(registry_path: str | Path, audit_path: str | Path) -> dict[str, Any]:
    audit = audit_registry(load_registry(registry_path))
    _write_json(Path(audit_path), audit)
    return audit


def _initialize_agent_model(factory: Any, model_config: Any) -> Any:
    return factory(config=model_config.to_minisweagent_config())


def _run_cell(
    cell: Any,
    *,
    registry_path: str | Path,
    cost_limit_usd: float,
) -> dict[str, Any]:
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    run_dir = Path(cell.artifact_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    task = _safe_manifest_task(cell.manifest_path, cell.task_id)
    model_config = load_model_config(cell.model_config)
    model_config.validate(require_runtime=True)
    base_config = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml"))
    instance = {**task, "instance_id": task["task_id"]}
    update_run_status(registry_path, cell.run_id, "running")
    env = None
    agent = None
    try:
        model = _initialize_agent_model(get_model, model_config)
        env = get_sb_environment(base_config, instance)
        agent_config = {**base_config["agent"], "cost_limit": cost_limit_usd}
        agent = HelloLoopAgent(
            model=model,
            env=env,
            task_spec=TaskSpec(
                task_id=cell.task_id,
                benchmark=cell.benchmark,
                problem_statement=str(task["problem_statement"]),
                repo=task.get("repo"),
                base_commit=task.get("base_commit"),
                metadata={"version": task.get("version")},
            ),
            run_id=cell.run_id,
            run_dir=run_dir,
            loop_variant=LoopVariant(cell.loop_variant),
            seed=cell.seed,
            repetition=cell.repetition_id,
            **agent_config,
        )
        result = agent.run()
        patch = str(result.get("submission", ""))
        write_predictions_jsonl(
            [SWEbenchPrediction(cell.task_id, patch, model_config.model)],
            run_dir / "prediction.jsonl",
        )
        exit_status = str(result.get("exit_status", "Unknown"))
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_cost = (
            int(summary.get("total_input_tokens", 0))
            * float(model_config.input_price_per_million_usd)
            + int(summary.get("total_output_tokens", 0))
            * float(model_config.output_price_per_million_usd)
        ) / 1_000_000
        summary["estimated_cost_usd"] = model_cost
        _write_json(summary_path, summary)
        payload = {
            "schema_id": "helloloop.agent_run.v1",
            "run_id": cell.run_id,
            "task_id": cell.task_id,
            "loop_variant": cell.loop_variant,
            "configured_model": model_config.model,
            "returned_model_ids": sorted(set(agent._returned_model_ids)),
            "exit_status": exit_status,
            "model_cost_usd": model_cost,
            "prediction_path": str(run_dir / "prediction.jsonl"),
            "official_evaluation_status": "pending",
        }
        _write_json(run_dir / "run_result.json", payload)
        if exit_status == "Submitted":
            classification = "submitted_patch" if patch.strip() else "abstention"
            update_run_status(
                registry_path,
                cell.run_id,
                "complete",
                terminal_classification=classification,
            )
            write_status_event(
                run_dir,
                cell,
                status="complete",
                model_cost_usd=model_cost,
                terminal_classification=classification,
            )
        else:
            failure_type = "budget" if "Limit" in exit_status or "Time" in exit_status else "agent"
            reason = f"agent exit status: {exit_status}"
            update_run_status(
                registry_path,
                cell.run_id,
                "failed",
                reason=reason,
                failure_type=failure_type,
            )
            write_status_event(
                run_dir,
                cell,
                status="failed",
                reason=reason,
                failure_type=failure_type,
                model_cost_usd=model_cost,
            )
        return payload
    except Exception as error:
        safe_message, _ = redact_text(str(error), max_chars=8_000)
        safe_traceback, _ = redact_text(traceback.format_exc(), max_chars=100_000)
        reason = f"{type(error).__name__}: {safe_message}"
        model_cost, cost_accounting = _measured_or_reserved_failure_cost(
            run_dir,
            model_config,
            agent,
            cost_limit_usd,
        )
        failure = {
            "schema_id": "helloloop.agent_run_failure.v1",
            "run_id": cell.run_id,
            "error_type": type(error).__name__,
            "message": safe_message,
            "traceback": safe_traceback,
            "model_cost_usd": model_cost,
            "cost_accounting": cost_accounting,
        }
        _write_json(run_dir / "failure.json", failure)
        _record_infrastructure_failure(
            cell,
            registry_path=registry_path,
            run_dir=run_dir,
            reason=reason,
            model_cost_usd=model_cost,
        )
        return failure
    finally:
        if env is not None:
            if hasattr(env, "close"):
                env.close()
            elif hasattr(env, "cleanup"):
                env.cleanup()


def run_experiment(
    experiment: str,
    *,
    registry_path: str | Path = "results/run_registry.csv",
    audit_path: str | Path = "results/run_audit.json",
    cost_cap_usd: float | None = None,
    per_run_cost_cap_usd: float | None = None,
    limit: int | None = None,
    run_ids: set[str] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    required_memory_gib: float = 16.0,
    required_disk_gib: float = 120.0,
    mark_budget_exhausted_skipped: bool = False,
    prune_docker_after_task: bool = False,
) -> dict[str, Any]:
    experiment = resolve_experiment_name(experiment)
    if experiment not in EXPERIMENT_DATASETS:
        raise ValueError(f"unknown experiment: {experiment}")
    if cost_cap_usd is not None and cost_cap_usd <= 0:
        raise ValueError("cost cap must be positive")
    if per_run_cost_cap_usd is not None and per_run_cost_cap_usd <= 0:
        raise ValueError("per-run cost cap must be positive")
    readiness = experiment_readiness(
        registry_path=registry_path,
        cost_cap_usd=cost_cap_usd,
        required_memory_gib=required_memory_gib,
        required_disk_gib=required_disk_gib,
    )
    output_dir = Path("results/runs") / experiment
    blocked_path = output_dir / "BLOCKED.json"
    if not readiness["ready"]:
        payload = {
            "schema_id": "helloloop.experiment_execution.v1",
            "experiment": experiment,
            "status": "blocked",
            "executed_runs": 0,
            "blockers": readiness["blockers"],
        }
        _write_json(blocked_path, payload)
        return payload
    blocked_path.unlink(missing_ok=True)

    cells = select_run_cells(
        load_registry(registry_path),
        experiment,
        run_ids=run_ids,
        limit=limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    remaining_cost = float(cost_cap_usd) - float(readiness["observed_cost_usd"] or 0.0)
    results = []
    skipped = 0
    for index, cell in enumerate(cells):
        if remaining_cost <= 0.0:
            if mark_budget_exhausted_skipped:
                for unexecuted in cells[index:]:
                    reason = "allocated experiment cost cap exhausted before execution"
                    update_run_status(
                        registry_path,
                        unexecuted.run_id,
                        "skipped",
                        reason=reason,
                        failure_type="budget",
                    )
                    write_status_event(
                        unexecuted.artifact_path,
                        unexecuted,
                        status="skipped",
                        reason=reason,
                        failure_type="budget",
                    )
                    skipped += 1
            break
        cell_cost_limit = remaining_cost
        if per_run_cost_cap_usd is not None:
            cell_cost_limit = min(cell_cost_limit, per_run_cost_cap_usd)
        result = _run_cell(
            cell,
            registry_path=registry_path,
            cost_limit_usd=cell_cost_limit,
        )
        results.append(result)
        remaining_cost -= float(result.get("model_cost_usd", 0.0))
        task_boundary = index == len(cells) - 1 or cells[index + 1].task_id != cell.task_id
        if prune_docker_after_task and task_boundary:
            _prune_docker(Path(cell.artifact_path) / "docker_prune.json")
    audit = _refresh_audit(registry_path, audit_path)
    return {
        "schema_id": "helloloop.experiment_execution.v1",
        "experiment": experiment,
        "status": "executed",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_runs": len(cells),
        "executed_runs": len(results),
        "skipped_runs": skipped,
        "remaining_cost_usd": round(max(0.0, remaining_cost), 8),
        "budget_exhausted": remaining_cost <= 0.0,
        "audit": audit,
    }
