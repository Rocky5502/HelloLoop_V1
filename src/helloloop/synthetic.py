from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .enums import Decision, LoopVariant, Stage
from .schema import (
    CostRecord,
    JsonlTrajectoryWriter,
    StepEvidence,
    StepRecord,
    UncertaintyVector,
    summarize_trajectory,
    write_summary_csv,
    write_summary_json,
)

SYNTHETIC_LABEL = "Synthetic engineering validation only - not empirical evidence about real LLMs."


@dataclass(frozen=True)
class SyntheticTask:
    task_id: str
    category: str
    difficulty: float
    ambiguity: float
    retrieval_need: float
    verifier_noise: float
    stagnation_risk: float


TASK_CATEGORIES = [
    "localized_bug",
    "multi_file_bug",
    "retrieval_heavy_bug",
    "ambiguous_requirement",
    "verifier_disagreement",
    "repeated_failure",
    "stagnation",
    "misleading_memory",
    "premature_success",
    "cost_exhaustion",
    "escalation",
    "abstention",
]


def default_tasks() -> list[SyntheticTask]:
    tasks: list[SyntheticTask] = []
    for index, category in enumerate(TASK_CATEGORIES * 2):
        cycle = index // len(TASK_CATEGORIES)
        base = 0.25 + 0.03 * (index % len(TASK_CATEGORIES))
        tasks.append(
            SyntheticTask(
                task_id=f"synthetic-{index + 1:02d}-{category}",
                category=category,
                difficulty=min(0.95, base + 0.10 * cycle),
                ambiguity=0.75 if "ambiguous" in category or category == "abstention" else 0.25,
                retrieval_need=0.80
                if "retrieval" in category or category == "misleading_memory"
                else 0.25,
                verifier_noise=0.75
                if "verifier" in category or category == "premature_success"
                else 0.20,
                stagnation_risk=0.80 if category in {"stagnation", "repeated_failure"} else 0.25,
            )
        )
    return tasks


def write_manifest(path: Path, tasks: list[SyntheticTask]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": SYNTHETIC_LABEL,
        "benchmark": "synthetic_demo",
        "tasks": [task.__dict__ for task in tasks],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_uncertainty(task: SyntheticTask, step: int, loop: LoopVariant) -> UncertaintyVector:
    progress_decay = 0.05 * step
    loop_bonus = {
        LoopVariant.LINEAR: 0.00,
        LoopVariant.REFLECTION: 0.05,
        LoopVariant.VERIFICATION: 0.08,
        LoopVariant.UNCERTAINTY_AWARE: 0.14,
    }[loop]
    composite = max(
        0.05, min(0.98, task.difficulty + 0.20 * task.ambiguity - progress_decay - loop_bonus)
    )
    return UncertaintyVector(
        task=max(0.05, min(0.98, task.ambiguity - 0.03 * step)),
        planning=max(0.05, min(0.98, task.difficulty - 0.04 * step)),
        retrieval=max(
            0.05,
            min(
                0.98,
                task.retrieval_need
                - (0.20 if loop == LoopVariant.UNCERTAINTY_AWARE and step > 0 else 0.04 * step),
            ),
        ),
        execution=max(0.05, min(0.98, task.difficulty + task.stagnation_risk * 0.20 - 0.04 * step)),
        verification=max(
            0.05, min(0.98, task.verifier_noise + task.difficulty * 0.20 - 0.03 * step)
        ),
        memory=max(0.05, min(0.98, 0.70 if task.category == "misleading_memory" else 0.25)),
        progress=max(0.05, min(0.98, task.stagnation_risk - 0.02 * step)),
        termination=max(0.05, min(0.98, composite + task.verifier_noise * 0.10)),
        composite=composite,
    )


def _decision(
    task: SyntheticTask, loop: LoopVariant, step: int, uncertainty: UncertaintyVector
) -> Decision:
    if step >= 5 and loop == LoopVariant.LINEAR:
        return Decision.STOP_BUDGET if uncertainty.composite > 0.45 else Decision.STOP_SUCCESS
    if step >= 6:
        if task.category == "abstention" and loop == LoopVariant.UNCERTAINTY_AWARE:
            return Decision.ABSTAIN
        if task.category == "escalation" and loop == LoopVariant.UNCERTAINTY_AWARE:
            return Decision.ESCALATE
        return Decision.STOP_SUCCESS if uncertainty.composite < 0.55 else Decision.STOP_UNRESOLVED
    if loop == LoopVariant.LINEAR:
        return Decision.EXECUTE
    if loop == LoopVariant.REFLECTION and step in {2, 4}:
        return Decision.REPLAN
    if loop == LoopVariant.VERIFICATION and step >= 2:
        return Decision.VERIFY_MORE
    if loop == LoopVariant.UNCERTAINTY_AWARE:
        if uncertainty.retrieval > 0.60:
            return Decision.RETRIEVE
        if uncertainty.planning > 0.60:
            return Decision.REPLAN
        if uncertainty.verification > 0.55 and step >= 2:
            return Decision.VERIFY_MORE
        if uncertainty.progress > 0.70 and step >= 3:
            return Decision.ESCALATE
    return Decision.EXECUTE


def _stage(decision: Decision) -> Stage:
    return {
        Decision.RETRIEVE: Stage.RETRIEVAL,
        Decision.REPLAN: Stage.PLANNING,
        Decision.VERIFY_MORE: Stage.VERIFICATION,
        Decision.ESCALATE: Stage.TERMINATION,
        Decision.ABSTAIN: Stage.TERMINATION,
        Decision.STOP_SUCCESS: Stage.TERMINATION,
        Decision.STOP_UNRESOLVED: Stage.TERMINATION,
        Decision.STOP_BUDGET: Stage.TERMINATION,
    }.get(decision, Stage.EXECUTION)


def simulate_run(task: SyntheticTask, loop: LoopVariant, out_dir: Path, seed: int) -> Path:
    random.seed(f"{seed}:{task.task_id}:{loop.value}")
    run_id = f"synthetic_ci__{loop.value}__{task.task_id}"
    run_dir = out_dir / "runs" / loop.value / task.task_id
    trajectory_path = run_dir / "trajectory.jsonl"
    if trajectory_path.exists():
        trajectory_path.unlink()
    writer = JsonlTrajectoryWriter(trajectory_path)
    max_steps = 7
    for step in range(max_steps):
        uncertainty = _base_uncertainty(task, step, loop)
        decision = _decision(task, loop, step, uncertainty)
        stage = _stage(decision)
        evidence = StepEvidence.from_text(
            "synthetic_observation",
            f"{SYNTHETIC_LABEL} category={task.category} decision={decision.value}",
            source="synthetic_env",
            task_category=task.category,
        )
        record = StepRecord(
            run_id=run_id,
            task_id=task.task_id,
            benchmark="synthetic_demo",
            loop_variant=loop,
            model_name="synthetic_policy",
            seed=seed,
            repetition=1,
            step_index=step,
            stage=stage,
            action=f"synthetic_{decision.value}",
            decision=decision,
            rationale=f"{SYNTHETIC_LABEL} Deterministic policy for {task.category}.",
            uncertainty=uncertainty,
            evidence=[evidence],
            cost=CostRecord(input_tokens=20 + step, output_tokens=8 + step, tool_calls=1),
            started_at=f"2026-07-19T00:00:{step:02d}+00:00",
            ended_at=f"2026-07-19T00:00:{step:02d}+00:00",
            termination_state=decision.value if stage == Stage.TERMINATION else None,
        )
        writer.append(record)
        if stage == Stage.TERMINATION:
            break
    summary = summarize_trajectory(
        trajectory_path, trajectory_path=str(trajectory_path.relative_to(out_dir))
    )
    write_summary_json(summary, run_dir / "summary.json")
    return trajectory_path


def _write_comparison_table(summary_rows: list[dict[str, str]], out_path: Path) -> None:
    by_loop: dict[str, list[dict[str, str]]] = {}
    for row in summary_rows:
        by_loop.setdefault(row["loop_variant"], []).append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "loop_variant",
                "runs",
                "success_rate",
                "mean_steps",
                "mean_residual_uncertainty",
            ],
        )
        writer.writeheader()
        for loop, rows in sorted(by_loop.items()):
            successes = sum(row["final_decision"] == Decision.STOP_SUCCESS.value for row in rows)
            mean_steps = sum(int(row["steps"]) for row in rows) / len(rows)
            mean_uncertainty = sum(float(row["residual_uncertainty"]) for row in rows) / len(rows)
            writer.writerow(
                {
                    "label": SYNTHETIC_LABEL,
                    "loop_variant": loop,
                    "runs": len(rows),
                    "success_rate": f"{successes / len(rows):.3f}",
                    "mean_steps": f"{mean_steps:.3f}",
                    "mean_residual_uncertainty": f"{mean_uncertainty:.3f}",
                }
            )


def _simple_svg_bar(values: dict[str, float], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 720, 360
    margin = 60
    bar_width = 90
    max_value = max(values.values()) if values else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="32" font-family="Arial" font-size="16">{title}</text>',
        f'<text x="{margin}" y="54" font-family="Arial" font-size="11">{SYNTHETIC_LABEL}</text>',
    ]
    for index, (label, value) in enumerate(sorted(values.items())):
        x = margin + index * 150
        bar_h = 220 * (value / max_value if max_value else 0)
        y = height - margin - bar_h
        parts.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_h:.1f}" fill="#2563eb"/>'
        )
        parts.append(
            f'<text x="{x}" y="{height - 34}" font-family="Arial" font-size="11">{label}</text>'
        )
        parts.append(
            f'<text x="{x}" y="{y - 6:.1f}" font-family="Arial" font-size="11">{value:.2f}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _simple_svg_line(points: list[float], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 720, 360
    margin = 60
    coords = []
    for index, value in enumerate(points):
        x = margin + index * ((width - 2 * margin) / max(1, len(points) - 1))
        y = height - margin - value * (height - 2 * margin)
        coords.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(coords)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="32" font-family="Arial" font-size="16">{title}</text>
<text x="{margin}" y="54" font-family="Arial" font-size="11">{SYNTHETIC_LABEL}</text>
<polyline fill="none" stroke="#dc2626" stroke-width="3" points="{polyline}"/>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def run_synthetic_benchmark(out_dir: Path, seed: int = 20260711) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = default_tasks()
    write_manifest(Path("manifests/synthetic_demo_24.json"), tasks)
    trajectories = [
        simulate_run(task, loop, out_dir, seed)
        for task in tasks
        for loop in (
            LoopVariant.LINEAR,
            LoopVariant.REFLECTION,
            LoopVariant.VERIFICATION,
            LoopVariant.UNCERTAINTY_AWARE,
        )
    ]
    summaries = [
        summarize_trajectory(path, trajectory_path=str(path.relative_to(out_dir)))
        for path in trajectories
    ]
    summary_csv = out_dir / "summaries" / "summary.csv"
    write_summary_csv(summaries, summary_csv)
    with summary_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _write_comparison_table(rows, out_dir / "tables" / "loop_comparison.csv")
    success_values = {
        loop: sum(row["final_decision"] == Decision.STOP_SUCCESS.value for row in group)
        / len(group)
        for loop, group in _group_rows(rows).items()
    }
    _simple_svg_bar(
        success_values, out_dir / "figures" / "calibration_plot.svg", "Synthetic Success By Loop"
    )
    sorted_uncertainty = sorted(float(row["residual_uncertainty"]) for row in rows)
    points = [
        0.5 + 0.5 * math.sin(i / max(1, len(sorted_uncertainty) - 1) * math.pi) for i in range(24)
    ]
    points = [
        (point + sorted_uncertainty[i % len(sorted_uncertainty)]) / 2
        for i, point in enumerate(points)
    ]
    _simple_svg_line(
        points,
        out_dir / "figures" / "uncertainty_trajectory.svg",
        "Synthetic Uncertainty Trajectory",
    )
    (out_dir / "README.txt").write_text(SYNTHETIC_LABEL + "\n", encoding="utf-8")
    return {"runs": len(trajectories), "out_dir": str(out_dir), "summary_csv": str(summary_csv)}


def _group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["loop_variant"], []).append(row)
    return grouped
