from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

from helloloop.experiments.registry import RunCell, load_registry

from .sensitivity import generate_sensitivity_analysis
from .statistics import (
    calibration_metrics,
    compare_paired_binary,
    require_complete_empirical_registry,
    risk_coverage_curve,
    to_jsonable,
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def _artifact_dir(project: Path, cell: RunCell) -> Path:
    path = Path(cell.artifact_path)
    return path if path.is_absolute() else project / path


def _load_trajectory(project: Path, cell: RunCell) -> list[dict[str, Any]]:
    path = _artifact_dir(project, cell) / "trajectory.jsonl"
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"officially evaluated run has an empty trajectory: {cell.run_id}")
    return records


def _summary(project: Path, cell: RunCell) -> dict[str, Any]:
    path = _artifact_dir(project, cell) / "summary.json"
    return _read_json(path) if path.exists() else {}


def _estimated_cost(project: Path, cell: RunCell, summary: dict[str, Any]) -> float | str:
    summary_cost = summary.get("estimated_cost_usd")
    if summary_cost is not None:
        return float(summary_cost)
    event_path = _artifact_dir(project, cell) / "status_event.json"
    if not event_path.exists():
        return ""
    event_cost = _read_json(event_path).get("model_cost_usd")
    return "" if event_cost is None else float(event_cost)


def _verification_passed(records: list[dict[str, Any]]) -> bool:
    for record in records:
        if record.get("stage") != "verification":
            continue
        for evidence in record.get("evidence", []):
            if evidence.get("metadata", {}).get("returncode") == 0:
                return True
    return False


def _sustained_high_uncertainty(records: list[dict[str, Any]]) -> tuple[bool, bool]:
    values = [float(record["uncertainty"]["composite"]) for record in records]
    for index in range(2, len(values)):
        if all(value >= 0.7 for value in values[index - 2 : index + 1]):
            substantive_after = any(
                record.get("stage") != "termination" for record in records[index + 1 :]
            )
            return True, substantive_after
    return False, False


def _run_row(project: Path, cell: RunCell) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    officially_evaluated = cell.official_evaluation_status == "complete"
    resolved = cell.official_resolved == "true" if officially_evaluated else None
    summary = _summary(project, cell)
    estimated_cost = _estimated_cost(project, cell, summary)
    records = _load_trajectory(project, cell) if officially_evaluated else []
    residual = summary.get("residual_uncertainty")
    confidence = None if residual is None else 1.0 - float(residual)
    row = {
        "run_id": cell.run_id,
        "experiment": cell.experiment,
        "task_id": cell.task_id,
        "loop_variant": cell.loop_variant,
        "model_name": cell.model_name,
        "status": cell.status,
        "terminal_classification": cell.terminal_classification,
        "official_evaluation_status": cell.official_evaluation_status,
        "official_resolved": "" if resolved is None else str(resolved).lower(),
        "registered_success": str(resolved is True).lower(),
        "confidence": "" if confidence is None else confidence,
        "residual_uncertainty": "" if residual is None else residual,
        "steps": summary.get("steps", ""),
        "input_tokens": summary.get("total_input_tokens", ""),
        "output_tokens": summary.get("total_output_tokens", ""),
        "tool_calls": summary.get("total_tool_calls", ""),
        "estimated_cost_usd": estimated_cost,
        "artifact_path": cell.artifact_path,
    }
    return row, records


RUN_FIELDS = [
    "run_id",
    "experiment",
    "task_id",
    "loop_variant",
    "model_name",
    "status",
    "terminal_classification",
    "official_evaluation_status",
    "official_resolved",
    "registered_success",
    "confidence",
    "residual_uncertainty",
    "steps",
    "input_tokens",
    "output_tokens",
    "tool_calls",
    "estimated_cost_usd",
    "artifact_path",
]


def generate_empirical_analysis(
    root: str | Path = ".",
    *,
    registry_path: str | Path = "results/run_registry.csv",
    audit_path: str | Path = "results/run_audit.json",
    output_dir: str | Path = "results/empirical",
) -> dict[str, Any]:
    project = Path(root).resolve()
    registry = project / registry_path
    audit = require_complete_empirical_registry(project / audit_path)
    output = project / output_dir
    cells = load_registry(registry)
    run_rows = []
    trajectories: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        row, records = _run_row(project, cell)
        run_rows.append(row)
        if records:
            trajectories[cell.run_id] = records
    _write_csv(output / "run_level.csv", run_rows, RUN_FIELDS)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["experiment"]), str(row["loop_variant"]))].append(row)
    loop_rows = []
    for (experiment, loop), rows in sorted(grouped.items()):
        successes = sum(row["registered_success"] == "true" for row in rows)
        evaluated = [row for row in rows if row["official_evaluation_status"] == "complete"]
        evaluated_successes = sum(row["official_resolved"] == "true" for row in evaluated)
        low, high = _wilson_interval(successes, len(rows))
        loop_rows.append(
            {
                "experiment": experiment,
                "loop_variant": loop,
                "registered_runs": len(rows),
                "submitted_runs": sum(
                    row["terminal_classification"] in {"submitted_patch", "abstention"}
                    for row in rows
                ),
                "officially_evaluated_runs": len(evaluated),
                "resolved_runs": successes,
                "registered_success_rate": successes / len(rows),
                "registered_success_ci_low": low,
                "registered_success_ci_high": high,
                "evaluator_resolution_rate": (
                    evaluated_successes / len(evaluated) if evaluated else ""
                ),
            }
        )
    loop_fields = [
        "experiment",
        "loop_variant",
        "registered_runs",
        "submitted_runs",
        "officially_evaluated_runs",
        "resolved_runs",
        "registered_success_rate",
        "registered_success_ci_low",
        "registered_success_ci_high",
        "evaluator_resolution_rate",
    ]
    _write_csv(output / "loop_outcomes.csv", loop_rows, loop_fields)

    efficiency_rows = []
    for (experiment, loop), rows in sorted(grouped.items()):
        with_summaries = [row for row in rows if row["steps"] != ""]
        known_cost = [
            float(row["estimated_cost_usd"]) for row in rows if row["estimated_cost_usd"] != ""
        ]
        efficiency_rows.append(
            {
                "experiment": experiment,
                "loop_variant": loop,
                "runs": len(rows),
                "runs_with_summary": len(with_summaries),
                "runs_with_known_cost": len(known_cost),
                "mean_cost_usd": sum(known_cost) / len(known_cost) if known_cost else "",
                "total_cost_usd": sum(known_cost) if known_cost else "",
                "mean_steps": (
                    sum(float(row["steps"]) for row in with_summaries) / len(with_summaries)
                    if with_summaries
                    else ""
                ),
                "mean_input_tokens": (
                    sum(float(row["input_tokens"]) for row in with_summaries) / len(with_summaries)
                    if with_summaries
                    else ""
                ),
                "mean_output_tokens": (
                    sum(float(row["output_tokens"]) for row in with_summaries) / len(with_summaries)
                    if with_summaries
                    else ""
                ),
                "mean_tool_calls": (
                    sum(float(row["tool_calls"]) for row in with_summaries) / len(with_summaries)
                    if with_summaries
                    else ""
                ),
            }
        )
    efficiency_fields = [
        "experiment",
        "loop_variant",
        "runs",
        "runs_with_summary",
        "runs_with_known_cost",
        "mean_cost_usd",
        "total_cost_usd",
        "mean_steps",
        "mean_input_tokens",
        "mean_output_tokens",
        "mean_tool_calls",
    ]
    _write_csv(output / "efficiency.csv", efficiency_rows, efficiency_fields)

    calibrated = [
        row
        for row in run_rows
        if row["official_evaluation_status"] == "complete" and row["confidence"] != ""
    ]
    calibration_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    calibration_payload: dict[str, Any] = {"available": False, "n": 0}
    if calibrated:
        confidence = [float(row["confidence"]) for row in calibrated]
        outcomes = [row["official_resolved"] == "true" for row in calibrated]
        calibration = calibration_metrics(confidence, outcomes)
        calibration_payload = {"available": True, **to_jsonable(calibration)}
        calibration_rows = [asdict(item) for item in calibration.equal_width_bins]
        risk_rows = [asdict(item) for item in risk_coverage_curve(confidence, outcomes)]
    _write_json(output / "calibration.json", calibration_payload)
    _write_csv(
        output / "calibration_bins.csv",
        calibration_rows,
        ["bin_index", "lower", "upper", "n", "mean_confidence", "empirical_accuracy"],
    )
    _write_csv(
        output / "risk_coverage.csv",
        risk_rows,
        ["coverage", "risk", "selective_accuracy", "retained", "threshold"],
    )

    trajectory_rows = []
    stage_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for cell in cells:
        for record in trajectories.get(cell.run_id, []):
            value = float(record["uncertainty"]["composite"])
            trajectory_rows.append(
                {
                    "run_id": cell.run_id,
                    "experiment": cell.experiment,
                    "task_id": cell.task_id,
                    "loop_variant": cell.loop_variant,
                    "step_index": int(record["step_index"]),
                    "stage": str(record["stage"]),
                    "composite_uncertainty": value,
                    "official_resolved": cell.official_resolved,
                }
            )
            stage_values[(cell.loop_variant, str(record["stage"]))].append(value)
    _write_csv(
        output / "trajectory_points.csv",
        trajectory_rows,
        [
            "run_id",
            "experiment",
            "task_id",
            "loop_variant",
            "step_index",
            "stage",
            "composite_uncertainty",
            "official_resolved",
        ],
    )
    stage_rows = [
        {
            "loop_variant": loop,
            "stage": stage,
            "n": len(values),
            "mean_uncertainty": sum(values) / len(values),
        }
        for (loop, stage), values in sorted(stage_values.items())
    ]
    _write_csv(
        output / "stage_uncertainty.csv",
        stage_rows,
        ["loop_variant", "stage", "n", "mean_uncertainty"],
    )

    paired_results = []
    for experiment in sorted({cell.experiment for cell in cells}):
        experiment_cells = [cell for cell in cells if cell.experiment == experiment]
        loops = sorted({cell.loop_variant for cell in experiment_cells})
        by_key = {(cell.task_id, cell.loop_variant): cell for cell in experiment_cells}
        tasks = sorted({cell.task_id for cell in experiment_cells})
        for loop_a, loop_b in combinations(loops, 2):
            paired_tasks = [
                task for task in tasks if (task, loop_a) in by_key and (task, loop_b) in by_key
            ]
            values_a = [by_key[(task, loop_a)].official_resolved == "true" for task in paired_tasks]
            values_b = [by_key[(task, loop_b)].official_resolved == "true" for task in paired_tasks]
            result = compare_paired_binary(
                values_a,
                values_b,
                variant_a=loop_a,
                variant_b=loop_b,
                missing_pairs=len(tasks) - len(paired_tasks),
            )
            paired_results.append({"experiment": experiment, **to_jsonable(result)})
    _write_json(
        output / "paired_tests.json",
        {"schema_id": "helloloop.paired_tests.v1", "comparisons": paired_results},
    )

    ethics_rows = []
    evaluated_cells = [cell for cell in cells if cell.official_evaluation_status == "complete"]
    submitted = [
        cell for cell in evaluated_cells if cell.terminal_classification == "submitted_patch"
    ]
    ethics_rows.append(
        {
            "metric": "false_assurance_rate",
            "numerator": sum(cell.official_resolved == "false" for cell in submitted),
            "denominator": len(submitted),
            "rate": (
                sum(cell.official_resolved == "false" for cell in submitted) / len(submitted)
                if submitted
                else ""
            ),
            "excluded_unknown": 0,
            "definition": "submitted patches with official unresolved outcome",
        }
    )
    sustained = []
    premature = []
    for cell in evaluated_cells:
        records = trajectories[cell.run_id]
        high, continued = _sustained_high_uncertainty(records)
        if high:
            sustained.append(continued)
        premature.append(not _verification_passed(records))
    for metric, values, definition in (
        (
            "unsafe_persistence_rate",
            sustained,
            "officially evaluated runs continuing after three consecutive uncertainty values >= 0.7",
        ),
        (
            "premature_stop_proxy_rate",
            premature,
            "officially evaluated submissions without a recorded passing verification action",
        ),
    ):
        ethics_rows.append(
            {
                "metric": metric,
                "numerator": sum(values),
                "denominator": len(values),
                "rate": sum(values) / len(values) if values else "",
                "excluded_unknown": 0,
                "definition": definition,
            }
        )
    ethics_rows.extend(
        [
            {
                "metric": metric,
                "numerator": "",
                "denominator": 0,
                "rate": "",
                "excluded_unknown": count,
                "definition": definition,
            }
            for metric, count, definition in (
                (
                    "opaque_escalation_rate",
                    sum(cell.terminal_classification == "policy_stop" for cell in cells),
                    "requires structured human annotation",
                ),
                (
                    "inappropriate_abstention_rate",
                    sum(cell.terminal_classification == "abstention" for cell in cells),
                    "requires solvability adjudication beyond the abstaining patch",
                ),
                (
                    "computational_overuse_rate",
                    len(cells),
                    "requires action-level human annotation",
                ),
            )
        ]
    )
    _write_csv(
        output / "ethical_metrics.csv",
        ethics_rows,
        ["metric", "numerator", "denominator", "rate", "excluded_unknown", "definition"],
    )

    taxonomy_rows = [
        {
            "terminal_classification": classification,
            "count": sum(cell.terminal_classification == classification for cell in cells),
        }
        for classification in sorted({cell.terminal_classification for cell in cells})
    ]
    _write_csv(
        output / "error_taxonomy.csv",
        taxonomy_rows,
        ["terminal_classification", "count"],
    )

    sensitivity_records = [
        {
            "run_id": cell.run_id,
            "resolved": cell.official_resolved == "true",
            "uncertainty": trajectories[cell.run_id][-1]["uncertainty"],
        }
        for cell in evaluated_cells
    ]
    sensitivity = generate_sensitivity_analysis(sensitivity_records, output)

    result = {
        "schema_id": "helloloop.empirical_analysis.v1",
        "complete": True,
        "planned_runs": len(cells),
        "terminal_runs": int(audit["terminal_runs"]),
        "officially_evaluated_runs": int(audit["officially_evaluated_runs"]),
        "resolved_runs": sum(cell.official_resolved == "true" for cell in cells),
        "output_dir": str(output),
        "source_registry": str(registry),
        "synthetic_inputs_used": False,
        "sensitivity": sensitivity,
    }
    _write_json(output / "analysis_status.json", result)
    return result
