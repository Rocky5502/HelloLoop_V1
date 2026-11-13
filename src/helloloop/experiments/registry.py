from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from helloloop.enums import LoopVariant

RUN_STATUSES = frozenset({"planned", "running", "complete", "failed", "skipped"})
TERMINAL_STATUSES = frozenset({"complete", "failed", "skipped"})
FAILURE_TYPES = frozenset({"", "agent", "infrastructure", "budget", "policy"})
OFFICIAL_EVALUATION_STATUSES = frozenset({"pending", "complete", "failed", "not_applicable"})
TERMINAL_CLASSIFICATIONS = frozenset(
    {
        "",
        "submitted_patch",
        "abstention",
        "agent_failure",
        "infrastructure_failure",
        "budget_stop",
        "policy_stop",
    }
)
RUN_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ExperimentSpec:
    phase: str
    name: str
    benchmark: str
    manifest_path: str
    loops: tuple[LoopVariant, ...]
    model_configs: tuple[str, ...]
    repetitions: int = 1
    seed: int = 20260711

    def validate(self) -> None:
        if not self.name or not self.phase or not self.benchmark:
            raise ValueError("phase, experiment name, and benchmark are required")
        if not self.loops or not self.model_configs or self.repetitions < 1:
            raise ValueError("experiment matrix dimensions must be positive")


@dataclass(frozen=True)
class RunCell:
    phase: str
    experiment: str
    order: int
    run_id: str
    benchmark: str
    manifest_path: str
    task_id: str
    loop_variant: str
    model_config: str
    model_name: str
    repetition_id: int
    seed: int
    status: str = "planned"
    reason: str = ""
    failure_type: str = ""
    created_at: str = ""
    updated_at: str = ""
    artifact_path: str = ""
    official_evaluation_status: str = "pending"
    official_resolved: str = ""
    official_evaluation_reason: str = ""
    terminal_classification: str = ""

    def validate(self) -> None:
        for field_name in (
            "phase",
            "experiment",
            "run_id",
            "benchmark",
            "manifest_path",
            "task_id",
            "loop_variant",
            "model_config",
            "model_name",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} is required")
        LoopVariant(self.loop_variant)
        if self.order < 0 or self.repetition_id < 0:
            raise ValueError("order and repetition must be non-negative")
        if self.status not in RUN_STATUSES:
            raise ValueError(f"invalid run status: {self.status}")
        if self.failure_type not in FAILURE_TYPES:
            raise ValueError(f"invalid failure type: {self.failure_type}")
        if self.status in {"failed", "skipped"} and not self.reason.strip():
            raise ValueError(f"{self.status} run requires a reason")
        if self.status == "failed" and not self.failure_type:
            raise ValueError("failed run requires a failure type")
        if self.terminal_classification not in TERMINAL_CLASSIFICATIONS:
            raise ValueError(f"invalid terminal classification: {self.terminal_classification}")
        if self.status not in TERMINAL_STATUSES and self.terminal_classification:
            raise ValueError("non-terminal run cannot have a terminal classification")
        if self.status in TERMINAL_STATUSES and not self.terminal_classification:
            raise ValueError("terminal run requires a terminal classification")
        if self.status == "complete" and self.terminal_classification not in {
            "submitted_patch",
            "abstention",
        }:
            raise ValueError("completed run classification must describe its submission")
        if self.official_evaluation_status not in OFFICIAL_EVALUATION_STATUSES:
            raise ValueError(
                f"invalid official evaluation status: {self.official_evaluation_status}"
            )
        if self.status in {"planned", "running"} and self.official_evaluation_status != "pending":
            raise ValueError(f"{self.status} run must have pending official evaluation")
        if (
            self.status in {"failed", "skipped"}
            and self.official_evaluation_status != "not_applicable"
        ):
            raise ValueError(f"{self.status} run must mark official evaluation not applicable")
        if self.status == "complete" and self.official_evaluation_status == "not_applicable":
            raise ValueError("completed agent run requires an official evaluation")
        if self.official_evaluation_status == "complete" and self.official_resolved not in {
            "true",
            "false",
        }:
            raise ValueError("completed official evaluation requires a resolved boolean")
        if self.official_evaluation_status != "complete" and self.official_resolved:
            raise ValueError("official resolved value requires a completed evaluation")
        if self.official_evaluation_status == "failed" and not self.official_evaluation_reason:
            raise ValueError("failed official evaluation requires a reason")


def _load_manifest_tasks(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    task_ids = [str(task["task_id"]) for task in tasks]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError(f"manifest {path} is empty or contains duplicate task IDs")
    return task_ids


def _load_model_name(path: str | Path) -> str:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyYAML is required for experiment planning") from error
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return str(data.get("model", "UNRESOLVED"))


def _slug(value: str) -> str:
    return RUN_ID_RE.sub("-", value).strip("-")


def _stable_seed(base_seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{value}".encode()).hexdigest()
    return int(digest[:16], 16)


def plan_experiment(spec: ExperimentSpec) -> list[RunCell]:
    spec.validate()
    task_ids = _load_manifest_tasks(spec.manifest_path)
    task_rng = random.Random(spec.seed)
    task_rng.shuffle(task_ids)
    model_names = {path: _load_model_name(path) for path in spec.model_configs}
    cells = []
    now = datetime.now(UTC).isoformat()
    for task_id in task_ids:
        combinations = [
            (loop, model_config, repetition)
            for repetition in range(spec.repetitions)
            for model_config in spec.model_configs
            for loop in spec.loops
        ]
        random.Random(_stable_seed(spec.seed, task_id)).shuffle(combinations)
        for loop, model_config, repetition in combinations:
            run_id = "__".join(
                (
                    _slug(spec.name),
                    _slug(task_id),
                    loop.value,
                    _slug(Path(model_config).stem),
                    f"r{repetition}",
                )
            )
            cells.append(
                RunCell(
                    phase=spec.phase,
                    experiment=spec.name,
                    order=len(cells),
                    run_id=run_id,
                    benchmark=spec.benchmark,
                    manifest_path=spec.manifest_path,
                    task_id=task_id,
                    loop_variant=loop.value,
                    model_config=model_config,
                    model_name=model_names[model_config],
                    repetition_id=repetition,
                    seed=spec.seed,
                    created_at=now,
                    updated_at=now,
                    artifact_path=f"results/runs/{spec.name}/{run_id}",
                )
            )
    return cells


def default_experiment_specs() -> tuple[ExperimentSpec, ...]:
    primary = ("configs/models/primary_model.yaml",)
    return (
        ExperimentSpec(
            phase="phase12",
            name="swebench_lite_dev_pilot",
            benchmark="swebench_lite_dev",
            manifest_path="manifests/swebench_lite_dev_12.json",
            loops=(
                LoopVariant.LINEAR,
                LoopVariant.VERIFICATION,
                LoopVariant.UNCERTAINTY_AWARE,
            ),
            model_configs=primary,
        ),
        ExperimentSpec(
            phase="phase13",
            name="swebench_lite_main",
            benchmark="swebench_lite_test",
            manifest_path="manifests/swebench_lite_test_40.json",
            loops=tuple(LoopVariant),
            model_configs=primary,
        ),
        ExperimentSpec(
            phase="phase14",
            name="swebench_verified_holdout",
            benchmark="swebench_verified",
            manifest_path="manifests/swebench_verified_holdout_20.json",
            loops=(LoopVariant.VERIFICATION, LoopVariant.UNCERTAINTY_AWARE),
            model_configs=primary,
        ),
    )


def build_default_experiment_plan(
    *,
    registry_path: str | Path = "results/run_registry.csv",
    plan_path: str | Path = "results/experiment_plan.json",
    audit_path: str | Path = "results/run_audit.json",
) -> dict[str, Any]:
    specs = default_experiment_specs()
    cells = []
    for spec in specs:
        experiment_cells = plan_experiment(spec)
        offset = len(cells)
        cells.extend(replace(cell, order=cell.order + offset) for cell in experiment_cells)
    write_registry(cells, registry_path)
    plan = {
        "schema_id": "helloloop.experiment_plan.v1",
        "seed": 20260711,
        "matrix": [
            {
                "phase": spec.phase,
                "experiment": spec.name,
                "benchmark": spec.benchmark,
                "manifest_path": spec.manifest_path,
                "task_count": len(_load_manifest_tasks(spec.manifest_path)),
                "loops": [loop.value for loop in spec.loops],
                "model_configs": list(spec.model_configs),
                "repetitions": spec.repetitions,
                "planned_runs": len([cell for cell in cells if cell.experiment == spec.name]),
            }
            for spec in specs
        ],
        "planned_runs": len(cells),
        "run_order": "task-major; combinations deterministically shuffled within task",
    }
    plan_target = Path(plan_path)
    plan_target.parent.mkdir(parents=True, exist_ok=True)
    plan_target.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = audit_registry(cells, specs)
    audit_target = Path(audit_path)
    audit_target.parent.mkdir(parents=True, exist_ok=True)
    audit_target.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["status"] != "pass":
        raise ValueError(f"experiment plan audit failed; see {audit_target}")
    return {"plan": plan, "audit": audit, "registry": str(registry_path)}


def append_replication_plan(
    *,
    registry_path: str | Path = "results/run_registry.csv",
    plan_path: str | Path = "results/replication_plan.json",
    audit_path: str | Path = "results/run_audit.json",
    manifest_path: str | Path = "manifests/swebench_lite_test_40.json",
    model_config: str | Path = "configs/models/primary_model.yaml",
    seed: int = 20260720,
) -> dict[str, Any]:
    cells = load_registry(registry_path)
    experiment_name = "swebench_lite_replication"
    spec = ExperimentSpec(
        phase="phase15",
        name=experiment_name,
        benchmark="swebench_lite_test",
        manifest_path=str(manifest_path),
        loops=(LoopVariant.LINEAR, LoopVariant.UNCERTAINTY_AWARE),
        model_configs=(str(model_config),),
        seed=seed,
    )
    base_cells = [cell for cell in cells if cell.experiment != experiment_name]
    expected_extension = [
        replace(
            cell,
            order=len(base_cells) + index,
            run_id=cell.run_id.removesuffix("__r0") + "__r1",
            repetition_id=1,
            artifact_path=cell.artifact_path.removesuffix("__r0") + "__r1",
        )
        for index, cell in enumerate(plan_experiment(spec))
    ]
    existing = [cell for cell in cells if cell.experiment == experiment_name]
    if existing:
        design_fields = (
            "phase",
            "experiment",
            "order",
            "run_id",
            "benchmark",
            "manifest_path",
            "task_id",
            "loop_variant",
            "model_config",
            "model_name",
            "repetition_id",
            "seed",
            "artifact_path",
        )
        observed_design = [
            tuple(getattr(cell, field) for field in design_fields) for cell in existing
        ]
        expected_design = [
            tuple(getattr(cell, field) for field in design_fields) for cell in expected_extension
        ]
        if observed_design != expected_design:
            raise ValueError("existing replication extension does not match the frozen design")
        extension = existing
        added = 0
    else:
        extension = expected_extension
        cells.extend(extension)
        added = len(extension)
        write_registry(cells, registry_path)

    plan = {
        "schema_id": "helloloop.replication_plan.v1",
        "experiment": experiment_name,
        "analysis_status": "posthoc_replication_not_registered_primary",
        "benchmark": "swebench_lite_test",
        "manifest_path": str(manifest_path),
        "model_config": str(model_config),
        "model_id": extension[0].model_name,
        "seed": seed,
        "tasks": len({cell.task_id for cell in extension}),
        "loops": [LoopVariant.LINEAR.value, LoopVariant.UNCERTAINTY_AWARE.value],
        "planned_runs": len(extension),
        "repetition_id": 1,
        "purpose": (
            "test task-level repeatability of the primary linear versus uncertainty-aware "
            "comparison without changing its confirmatory status"
        ),
    }
    target = Path(plan_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = audit_registry(cells)
    Path(audit_path).write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_id": "helloloop.replication_plan_append.v1",
        "added_runs": added,
        "replication_runs": len(extension),
        "registry_runs": len(cells),
        "registry": str(registry_path),
        "plan": str(plan_path),
        "audit": audit,
    }


def write_registry(cells: list[RunCell], path: str | Path) -> Path:
    for cell in cells:
        cell.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    fields = list(RunCell.__dataclass_fields__)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(cell) for cell in cells)
    temporary.replace(target)
    return target


def load_registry(path: str | Path) -> list[RunCell]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cells = []
    for row in rows:
        row["order"] = int(row["order"])
        row["repetition_id"] = int(row["repetition_id"])
        row["seed"] = int(row["seed"])
        row.setdefault("official_evaluation_status", "pending")
        row.setdefault("official_resolved", "")
        row.setdefault("official_evaluation_reason", "")
        row.setdefault("terminal_classification", "")
        cell = RunCell(**row)
        cell.validate()
        cells.append(cell)
    return cells


def update_run_status(
    registry_path: str | Path,
    run_id: str,
    status: str,
    *,
    reason: str = "",
    failure_type: str = "",
    terminal_classification: str = "",
) -> RunCell:
    cells = load_registry(registry_path)
    matches = [index for index, cell in enumerate(cells) if cell.run_id == run_id]
    if len(matches) != 1:
        raise ValueError(f"expected one registry cell for {run_id}, found {len(matches)}")
    index = matches[0]
    official_status = cells[index].official_evaluation_status
    if status in {"failed", "skipped"}:
        official_status = "not_applicable"
    if status in TERMINAL_STATUSES and not terminal_classification:
        if status == "complete":
            terminal_classification = "submitted_patch"
        else:
            terminal_classification = {
                "agent": "agent_failure",
                "infrastructure": "infrastructure_failure",
                "budget": "budget_stop",
                "policy": "policy_stop",
            }.get(failure_type, "")
    updated = replace(
        cells[index],
        status=status,
        reason=reason,
        failure_type=failure_type,
        updated_at=datetime.now(UTC).isoformat(),
        official_evaluation_status=official_status,
        official_resolved="" if official_status != "complete" else cells[index].official_resolved,
        official_evaluation_reason=(
            "" if official_status != "failed" else cells[index].official_evaluation_reason
        ),
        terminal_classification=terminal_classification,
    )
    updated.validate()
    cells[index] = updated
    write_registry(cells, registry_path)
    return updated


def update_official_evaluation(
    registry_path: str | Path,
    run_id: str,
    status: str,
    *,
    resolved: bool | None = None,
    reason: str = "",
) -> RunCell:
    if status not in {"complete", "failed"}:
        raise ValueError("official evaluation update status must be complete or failed")
    if status == "complete" and not isinstance(resolved, bool):
        raise ValueError("completed official evaluation requires a resolved boolean")
    if status == "failed" and not reason.strip():
        raise ValueError("failed official evaluation requires a reason")
    cells = load_registry(registry_path)
    matches = [index for index, cell in enumerate(cells) if cell.run_id == run_id]
    if len(matches) != 1:
        raise ValueError(f"expected one registry cell for {run_id}, found {len(matches)}")
    index = matches[0]
    if cells[index].status != "complete":
        raise ValueError("only completed agent runs can be officially evaluated")
    updated = replace(
        cells[index],
        official_evaluation_status=status,
        official_resolved=(str(resolved).lower() if status == "complete" else ""),
        official_evaluation_reason=reason,
        updated_at=datetime.now(UTC).isoformat(),
    )
    updated.validate()
    cells[index] = updated
    write_registry(cells, registry_path)
    return updated


def audit_registry(
    cells: list[RunCell],
    specs: tuple[ExperimentSpec, ...] | None = None,
) -> dict[str, Any]:
    violations = []
    run_ids = [cell.run_id for cell in cells]
    if len(run_ids) != len(set(run_ids)):
        violations.append("duplicate run IDs")
    if [cell.order for cell in cells] != list(range(len(cells))):
        violations.append("run order is not contiguous")
    experiments = {}
    for experiment in sorted({cell.experiment for cell in cells}):
        group = [cell for cell in cells if cell.experiment == experiment]
        task_positions: dict[str, list[int]] = {}
        for cell in group:
            task_positions.setdefault(cell.task_id, []).append(cell.order)
        if any(max(values) - min(values) + 1 != len(values) for values in task_positions.values()):
            violations.append(f"{experiment}: task cells are not contiguous")
        experiments[experiment] = {
            "planned_runs": len(group),
            "tasks": len(task_positions),
            "status_counts": dict(
                sorted(
                    {
                        status: sum(cell.status == status for cell in group)
                        for status in RUN_STATUSES
                    }.items()
                )
            ),
        }
    if specs:
        for spec in specs:
            expected = (
                len(_load_manifest_tasks(spec.manifest_path))
                * len(spec.loops)
                * len(spec.model_configs)
                * spec.repetitions
            )
            observed = sum(cell.experiment == spec.name for cell in cells)
            if observed != expected:
                violations.append(f"{spec.name}: expected {expected} cells, found {observed}")
    terminal = sum(cell.status in TERMINAL_STATUSES for cell in cells)
    evaluation_required = sum(cell.status == "complete" for cell in cells)
    officially_evaluated = sum(
        cell.status == "complete" and cell.official_evaluation_status == "complete"
        for cell in cells
    )
    official_evaluation_failed = sum(
        cell.status == "complete" and cell.official_evaluation_status == "failed" for cell in cells
    )
    resolved_runs = sum(cell.official_resolved == "true" for cell in cells)
    unresolved_runs = sum(cell.official_resolved == "false" for cell in cells)
    return {
        "schema_id": "helloloop.run_registry_audit.v1",
        "status": "pass" if not violations else "fail",
        "planned_runs": len(cells),
        "terminal_runs": terminal,
        "complete_for_analysis": (
            terminal == len(cells)
            and officially_evaluated == evaluation_required
            and not violations
        ),
        "official_evaluation_required": evaluation_required,
        "officially_evaluated_runs": officially_evaluated,
        "official_evaluation_failed": official_evaluation_failed,
        "official_resolved_runs": resolved_runs,
        "official_unresolved_runs": unresolved_runs,
        "terminal_classification_counts": dict(
            sorted(
                {
                    classification: sum(
                        cell.terminal_classification == classification for cell in cells
                    )
                    for classification in TERMINAL_CLASSIFICATIONS
                    if classification
                }.items()
            )
        ),
        "experiments": experiments,
        "violations": violations,
    }
