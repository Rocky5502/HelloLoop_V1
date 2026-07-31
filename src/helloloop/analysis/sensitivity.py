from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from .statistics import calibration_metrics

UNCERTAINTY_COMPONENTS = (
    "task",
    "planning",
    "retrieval",
    "execution",
    "verification",
    "memory",
    "progress",
    "termination",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _metric_summary(
    confidence: np.ndarray,
    outcomes: np.ndarray,
    baseline_confidence: np.ndarray,
) -> dict[str, float]:
    calibration = calibration_metrics(confidence.tolist(), outcomes.tolist())
    accepted = confidence >= 0.8
    baseline_accepted = baseline_confidence >= 0.8
    retained = max(1, math.ceil(len(confidence) / 2))
    selected = np.argsort(-confidence, kind="stable")[:retained]
    if np.array_equal(confidence, baseline_confidence):
        rank_stability = 1.0
    elif np.ptp(confidence) == 0.0 or np.ptp(baseline_confidence) == 0.0:
        rank_stability = 0.0
    else:
        rank_stability = float(stats.spearmanr(confidence, baseline_confidence).statistic)
        if not math.isfinite(rank_stability):
            rank_stability = 0.0
    return {
        "brier_score": calibration.brier_score,
        "ece_equal_width": calibration.ece_equal_width,
        "auroc": float(calibration.auroc) if calibration.auroc is not None else math.nan,
        "gate_acceptance_rate": float(accepted.mean()),
        "gate_flip_rate": float(np.mean(accepted != baseline_accepted)),
        "rank_stability": rank_stability,
        "accuracy_at_50pct_coverage": float(outcomes[selected].mean()),
    }


def _stable_seed(name: str, repetition: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{name}:{repetition}".encode()).hexdigest()
    return int(digest[:16], 16)


def _perturb(
    components: np.ndarray,
    name: str,
    repetition: int,
    seed: int,
) -> np.ndarray:
    result = components.copy()
    randomizer = np.random.default_rng(_stable_seed(name, repetition, seed))
    family, value = name.rsplit("_", 1)
    level = float(value)
    if family == "uniform_noise":
        result += randomizer.uniform(-level, level, size=result.shape)
    elif family == "missing":
        result[randomizer.random(size=result.shape) < level] = 0.5
    elif family == "upward_bias":
        result += level
    elif family == "downward_bias":
        result -= level
    elif family == "quantize":
        result = np.round(result / level) * level
    else:  # pragma: no cover - protected by the fixed perturbation list
        raise ValueError(f"unknown sensitivity perturbation: {name}")
    return np.clip(result, 0.0, 1.0)


def _interval(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    low = min(mean, float(np.quantile(array, 0.025)))
    high = max(mean, float(np.quantile(array, 0.975)))
    return mean, low, high


def generate_sensitivity_analysis(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    seed: int = 20260711,
    perturbation_repetitions: int = 200,
) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one officially evaluated trajectory is required")
    if perturbation_repetitions < 1:
        raise ValueError("perturbation repetitions must be positive")

    components = np.asarray(
        [
            [float(record["uncertainty"][component]) for component in UNCERTAINTY_COMPONENTS]
            for record in records
        ],
        dtype=float,
    )
    outcomes = np.asarray([bool(record["resolved"]) for record in records], dtype=int)
    if np.any((components < 0.0) | (components > 1.0)):
        raise ValueError("uncertainty components must lie in [0, 1]")
    if len(np.unique(outcomes)) != 2:
        raise ValueError("sensitivity analysis requires both resolved and unresolved outcomes")

    baseline_confidence = 1.0 - components.mean(axis=1)
    ablation_scores: list[tuple[str, str, np.ndarray]] = [
        ("default_weighted_average", "baseline", baseline_confidence)
    ]
    for index, component in enumerate(UNCERTAINTY_COMPONENTS):
        retained = np.delete(components, index, axis=1)
        ablation_scores.append(
            (
                f"drop_{component}",
                f"leave_one_component_out:{component}",
                1.0 - retained.mean(axis=1),
            )
        )
    ablation_scores.extend(
        [
            ("maximum_composite", "aggregation_mode:maximum", 1.0 - components.max(axis=1)),
            (
                "logistic_composite",
                "aggregation_mode:logistic",
                1.0 - 1.0 / (1.0 + np.exp(-8.0 * (components.mean(axis=1) - 0.5))),
            ),
        ]
    )
    baseline_metrics = _metric_summary(baseline_confidence, outcomes, baseline_confidence)
    ablation_rows = []
    for name, change, confidence in ablation_scores:
        metrics = _metric_summary(confidence, outcomes, baseline_confidence)
        ablation_rows.append(
            {
                "analysis": name,
                "change": change,
                "n": len(records),
                **metrics,
                "delta_brier_vs_default": metrics["brier_score"] - baseline_metrics["brier_score"],
                "scope": "posthoc_estimator_sensitivity_not_agent_rerun",
            }
        )

    perturbations = (
        "uniform_noise_0.02",
        "uniform_noise_0.05",
        "uniform_noise_0.10",
        "uniform_noise_0.20",
        "missing_0.10",
        "missing_0.25",
        "missing_0.50",
        "upward_bias_0.05",
        "upward_bias_0.10",
        "downward_bias_0.05",
        "downward_bias_0.10",
        "quantize_0.10",
    )
    robustness_rows = []
    metric_names = tuple(baseline_metrics)
    for name in perturbations:
        samples = {metric: [] for metric in metric_names}
        for repetition in range(perturbation_repetitions):
            perturbed = _perturb(components, name, repetition, seed)
            confidence = 1.0 - perturbed.mean(axis=1)
            metrics = _metric_summary(confidence, outcomes, baseline_confidence)
            for metric, value in metrics.items():
                samples[metric].append(value)
        row: dict[str, Any] = {
            "perturbation": name,
            "n": len(records),
            "repetitions": perturbation_repetitions,
            "scope": "posthoc_estimator_robustness_not_environment_rerun",
        }
        for metric in metric_names:
            mean, low, high = _interval(samples[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        robustness_rows.append(row)

    output = Path(output_dir)
    _write_csv(output / "estimator_ablations.csv", ablation_rows)
    _write_csv(output / "estimator_robustness.csv", robustness_rows)
    result = {
        "schema_id": "helloloop.estimator_sensitivity.v1",
        "n": len(records),
        "resolved": int(outcomes.sum()),
        "ablation_configurations": len(ablation_rows),
        "robustness_perturbations": len(robustness_rows),
        "perturbation_repetitions": perturbation_repetitions,
        "seed": seed,
        "scope": (
            "post-hoc uncertainty-estimator sensitivity over official outcomes; "
            "no agent trajectory or official score is changed"
        ),
    }
    (output / "sensitivity_status.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
