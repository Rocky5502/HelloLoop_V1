from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from statsmodels.formula.api import mixedlm


@dataclass(frozen=True)
class Estimate:
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    n: int


@dataclass(frozen=True)
class PairedBinaryResult:
    variant_a: str
    variant_b: str
    n_pairs: int
    missing_pairs: int
    a_success_b_failure: int
    a_failure_b_success: int
    success_rate_a: float
    success_rate_b: float
    risk_difference: Estimate
    mcnemar_exact_p: float


@dataclass(frozen=True)
class PairedContinuousResult:
    variant_a: str
    variant_b: str
    n_pairs: int
    missing_pairs: int
    median_difference: float
    mean_difference: Estimate
    wilcoxon_statistic: float | None
    wilcoxon_p: float | None
    rank_biserial: float | None


@dataclass(frozen=True)
class CalibrationBin:
    bin_index: int
    lower: float
    upper: float
    n: int
    mean_confidence: float | None
    empirical_accuracy: float | None


@dataclass(frozen=True)
class CalibrationResult:
    n: int
    positives: int
    brier_score: float
    ece_equal_width: float
    ece_equal_mass: float
    auroc: float | None
    equal_width_bins: tuple[CalibrationBin, ...]
    equal_mass_bins: tuple[CalibrationBin, ...]


@dataclass(frozen=True)
class RiskCoveragePoint:
    coverage: float
    risk: float
    selective_accuracy: float
    retained: int
    threshold: float


@dataclass(frozen=True)
class StageSummary:
    stage: str
    n: int
    mean: float
    median: float
    standard_deviation: float
    q25: float
    q75: float


@dataclass(frozen=True)
class Transition:
    source: str
    target: str
    count: int
    source_total: int
    probability: float


@dataclass(frozen=True)
class ParetoPoint:
    label: str
    cost: float
    success: float


def _finite_float_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if array.size == 0:
        raise ValueError("at least one value is required")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    return array


def _binary_array(values: Iterable[int | bool]) -> np.ndarray:
    array = _finite_float_array(values)
    if not np.all(np.isin(array, [0.0, 1.0])):
        raise ValueError("outcomes must be binary")
    return array.astype(int)


def paired_bootstrap_mean_difference(
    values_a: Iterable[float],
    values_b: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 20260711,
) -> Estimate:
    a = _finite_float_array(values_a)
    b = _finite_float_array(values_b)
    if len(a) != len(b):
        raise ValueError("paired arrays must have equal lengths")
    if not 0.0 < confidence < 1.0 or resamples < 1:
        raise ValueError("invalid bootstrap configuration")
    differences = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    estimates = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return Estimate(
        estimate=float(differences.mean()),
        ci_low=float(low),
        ci_high=float(high),
        n=len(differences),
    )


def compare_paired_binary(
    outcomes_a: Sequence[int | bool],
    outcomes_b: Sequence[int | bool],
    *,
    variant_a: str,
    variant_b: str,
    missing_pairs: int = 0,
    resamples: int = 10_000,
    seed: int = 20260711,
) -> PairedBinaryResult:
    a = _binary_array(outcomes_a)
    b = _binary_array(outcomes_b)
    if len(a) != len(b):
        raise ValueError("paired arrays must have equal lengths")
    a_win = int(np.sum((a == 1) & (b == 0)))
    b_win = int(np.sum((a == 0) & (b == 1)))
    discordant = a_win + b_win
    p_value = (
        1.0
        if discordant == 0
        else float(stats.binomtest(min(a_win, b_win), discordant, 0.5).pvalue)
    )
    return PairedBinaryResult(
        variant_a=variant_a,
        variant_b=variant_b,
        n_pairs=len(a),
        missing_pairs=missing_pairs,
        a_success_b_failure=a_win,
        a_failure_b_success=b_win,
        success_rate_a=float(a.mean()),
        success_rate_b=float(b.mean()),
        risk_difference=paired_bootstrap_mean_difference(a, b, resamples=resamples, seed=seed),
        mcnemar_exact_p=p_value,
    )


def compare_paired_continuous(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    variant_a: str,
    variant_b: str,
    missing_pairs: int = 0,
    resamples: int = 10_000,
    seed: int = 20260711,
) -> PairedContinuousResult:
    a = _finite_float_array(values_a)
    b = _finite_float_array(values_b)
    if len(a) != len(b):
        raise ValueError("paired arrays must have equal lengths")
    differences = a - b
    nonzero = differences[differences != 0]
    if nonzero.size:
        test = stats.wilcoxon(nonzero, alternative="two-sided", method="auto")
        ranks = stats.rankdata(np.abs(nonzero))
        positive = float(ranks[nonzero > 0].sum())
        negative = float(ranks[nonzero < 0].sum())
        rank_biserial = (positive - negative) / (positive + negative)
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    else:
        statistic = None
        p_value = None
        rank_biserial = 0.0
    return PairedContinuousResult(
        variant_a=variant_a,
        variant_b=variant_b,
        n_pairs=len(a),
        missing_pairs=missing_pairs,
        median_difference=float(np.median(differences)),
        mean_difference=paired_bootstrap_mean_difference(a, b, resamples=resamples, seed=seed),
        wilcoxon_statistic=statistic,
        wilcoxon_p=p_value,
        rank_biserial=rank_biserial,
    )


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> dict[str, float | int]:
    x_values = _finite_float_array(x)
    y_values = _finite_float_array(y)
    if len(x_values) != len(y_values):
        raise ValueError("arrays must have equal lengths")
    result = stats.spearmanr(x_values, y_values)
    return {"rho": float(result.statistic), "p_value": float(result.pvalue), "n": len(x_values)}


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = _finite_float_array(p_values)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running_max = 0.0
    count = len(values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, float(values[original_index]) * (count - rank))
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted.tolist()


def _equal_width_bins(
    confidence: np.ndarray, outcomes: np.ndarray, n_bins: int
) -> tuple[CalibrationBin, ...]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.minimum(np.searchsorted(edges, confidence, side="right") - 1, n_bins - 1)
    assignments = np.maximum(assignments, 0)
    result = []
    for index in range(n_bins):
        mask = assignments == index
        count = int(mask.sum())
        result.append(
            CalibrationBin(
                bin_index=index,
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                n=count,
                mean_confidence=float(confidence[mask].mean()) if count else None,
                empirical_accuracy=float(outcomes[mask].mean()) if count else None,
            )
        )
    return tuple(result)


def _equal_mass_bins(
    confidence: np.ndarray, outcomes: np.ndarray, n_bins: int
) -> tuple[CalibrationBin, ...]:
    sorted_indices = np.argsort(confidence, kind="stable")
    groups = np.array_split(sorted_indices, min(n_bins, len(confidence)))
    result = []
    for index, group in enumerate(groups):
        scores = confidence[group]
        result.append(
            CalibrationBin(
                bin_index=index,
                lower=float(scores.min()),
                upper=float(scores.max()),
                n=len(group),
                mean_confidence=float(scores.mean()),
                empirical_accuracy=float(outcomes[group].mean()),
            )
        )
    return tuple(result)


def _ece(bins: Sequence[CalibrationBin], total: int) -> float:
    return float(
        sum(
            item.n / total * abs(item.mean_confidence - item.empirical_accuracy)
            for item in bins
            if item.n and item.mean_confidence is not None and item.empirical_accuracy is not None
        )
    )


def calibration_metrics(
    confidence: Sequence[float], outcomes: Sequence[int | bool], *, n_bins: int = 10
) -> CalibrationResult:
    scores = _finite_float_array(confidence)
    labels = _binary_array(outcomes)
    if len(scores) != len(labels):
        raise ValueError("confidence and outcome arrays must have equal lengths")
    if np.any((scores < 0.0) | (scores > 1.0)) or n_bins < 1:
        raise ValueError("confidence must lie in [0, 1] and n_bins must be positive")
    width = _equal_width_bins(scores, labels, n_bins)
    mass = _equal_mass_bins(scores, labels, n_bins)
    auroc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
    return CalibrationResult(
        n=len(scores),
        positives=int(labels.sum()),
        brier_score=float(np.mean((scores - labels) ** 2)),
        ece_equal_width=_ece(width, len(scores)),
        ece_equal_mass=_ece(mass, len(scores)),
        auroc=auroc,
        equal_width_bins=width,
        equal_mass_bins=mass,
    )


def risk_coverage_curve(
    confidence: Sequence[float], outcomes: Sequence[int | bool]
) -> list[RiskCoveragePoint]:
    scores = _finite_float_array(confidence)
    labels = _binary_array(outcomes)
    if len(scores) != len(labels):
        raise ValueError("confidence and outcome arrays must have equal lengths")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("confidence must lie in [0, 1]")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    return [
        RiskCoveragePoint(
            coverage=index / len(labels),
            risk=float(1.0 - sorted_labels[:index].mean()),
            selective_accuracy=float(sorted_labels[:index].mean()),
            retained=index,
            threshold=float(sorted_scores[index - 1]),
        )
        for index in range(1, len(labels) + 1)
    ]


def summarize_stages(records: Sequence[tuple[str, float]]) -> list[StageSummary]:
    grouped: dict[str, list[float]] = {}
    for stage, value in records:
        if not stage.strip() or not math.isfinite(value):
            raise ValueError("stages must be named and uncertainty values finite")
        grouped.setdefault(stage, []).append(value)
    return [
        StageSummary(
            stage=stage,
            n=len(values),
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            standard_deviation=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            q25=float(np.quantile(values, 0.25)),
            q75=float(np.quantile(values, 0.75)),
        )
        for stage, values in sorted(grouped.items())
    ]


def uncertainty_volatility(values: Sequence[float]) -> dict[str, float | int]:
    series = _finite_float_array(values)
    if len(series) < 2:
        return {"n": len(series), "mean_absolute_delta": 0.0, "max_absolute_delta": 0.0}
    deltas = np.abs(np.diff(series))
    return {
        "n": len(series),
        "mean_absolute_delta": float(deltas.mean()),
        "max_absolute_delta": float(deltas.max()),
    }


def transition_matrix(sequences: Sequence[Sequence[str]]) -> list[Transition]:
    counts: dict[tuple[str, str], int] = {}
    source_totals: dict[str, int] = {}
    for sequence in sequences:
        for source, target in zip(sequence, sequence[1:]):
            if not source.strip() or not target.strip():
                raise ValueError("transition states must be named")
            counts[(source, target)] = counts.get((source, target), 0) + 1
            source_totals[source] = source_totals.get(source, 0) + 1
    return [
        Transition(
            source=source,
            target=target,
            count=count,
            source_total=source_totals[source],
            probability=count / source_totals[source],
        )
        for (source, target), count in sorted(counts.items())
    ]


def event_centered_analysis(
    sequences: Sequence[Sequence[tuple[str, float]]],
    *,
    event: str,
    offset_before: int = 1,
    offset_after: int = 1,
) -> dict[str, float | int | None]:
    if offset_before < 1 or offset_after < 1:
        raise ValueError("event offsets must be positive")
    before_values: list[float] = []
    after_values: list[float] = []
    total_events = 0
    for sequence in sequences:
        for index, (stage, _) in enumerate(sequence):
            if stage != event:
                continue
            total_events += 1
            before = index - offset_before
            after = index + offset_after
            if before >= 0 and after < len(sequence):
                before_values.append(float(sequence[before][1]))
                after_values.append(float(sequence[after][1]))
    if not before_values:
        return {
            "event": event,
            "total_events": total_events,
            "complete_windows": 0,
            "mean_before": None,
            "mean_after": None,
            "mean_delta": None,
        }
    before_array = _finite_float_array(before_values)
    after_array = _finite_float_array(after_values)
    return {
        "event": event,
        "total_events": total_events,
        "complete_windows": len(before_values),
        "mean_before": float(before_array.mean()),
        "mean_after": float(after_array.mean()),
        "mean_delta": float((after_array - before_array).mean()),
    }


def cluster_trajectories(
    trajectories: dict[str, Sequence[float]],
    *,
    n_clusters: int = 3,
    seed: int = 20260711,
) -> list[dict[str, float | int | str]]:
    if not trajectories or n_clusters < 1 or n_clusters > len(trajectories):
        raise ValueError("n_clusters must be between 1 and the trajectory count")
    features: list[list[float]] = []
    run_ids: list[str] = []
    for run_id, raw_values in sorted(trajectories.items()):
        values = _finite_float_array(raw_values)
        slope = float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) > 1 else 0.0
        volatility = float(np.abs(np.diff(values)).mean()) if len(values) > 1 else 0.0
        features.append(
            [
                float(values[0]),
                float(values[-1]),
                float(values.mean()),
                float(values.std()),
                slope,
                volatility,
            ]
        )
        run_ids.append(run_id)
    matrix = np.asarray(features, dtype=float)
    scale = matrix.std(axis=0)
    scale[scale == 0.0] = 1.0
    normalized = (matrix - matrix.mean(axis=0)) / scale
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(normalized)
    return [
        {
            "run_id": run_id,
            "cluster": int(label),
            "start": row[0],
            "end": row[1],
            "mean": row[2],
            "standard_deviation": row[3],
            "slope": row[4],
            "volatility": row[5],
        }
        for run_id, label, row in zip(run_ids, labels, features)
    ]


def pareto_frontier(points: Sequence[tuple[str, float, float]]) -> list[ParetoPoint]:
    validated: list[ParetoPoint] = []
    for label, cost, success in points:
        if not label.strip() or cost < 0.0 or not 0.0 <= success <= 1.0:
            raise ValueError(
                "Pareto points require a label, non-negative cost, and success in [0, 1]"
            )
        validated.append(ParetoPoint(label=label, cost=float(cost), success=float(success)))
    return [
        point
        for point in sorted(validated, key=lambda item: (item.cost, -item.success, item.label))
        if not any(
            other.cost <= point.cost
            and other.success >= point.success
            and (other.cost < point.cost or other.success > point.success)
            for other in validated
        )
    ]


def paired_assumption_checks(differences: Sequence[float]) -> dict[str, object]:
    values = _finite_float_array(differences)
    warnings = []
    if len(values) < 10:
        warnings.append("small_sample")
    if len(values) >= 3 and len(values) <= 5_000 and not np.all(values == values[0]):
        shapiro = stats.shapiro(values)
        statistic = float(shapiro.statistic)
        p_value = float(shapiro.pvalue)
    else:
        statistic = None
        p_value = None
        warnings.append("normality_test_not_applicable")
    return {
        "n": len(values),
        "zero_differences": int(np.sum(values == 0.0)),
        "shapiro_statistic": statistic,
        "shapiro_p": p_value,
        "warnings": warnings,
    }


def fit_linear_mixed_effects(
    records: Sequence[dict[str, object]],
    *,
    outcome: str,
    fixed_effects: Sequence[str],
    group: str = "task_id",
) -> dict[str, object]:
    if not records or not fixed_effects:
        raise ValueError("records and fixed effects are required")
    frame = pd.DataFrame(records)
    required = {outcome, group, *fixed_effects}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing mixed-model columns: {', '.join(missing)}")
    complete = frame[list(required)].dropna()
    if complete[group].nunique() < 2:
        raise ValueError("mixed-effects models require at least two groups")
    formula = f"{outcome} ~ " + " + ".join(fixed_effects)
    fit = mixedlm(formula, complete, groups=complete[group]).fit(reml=False, method="lbfgs")
    return {
        "formula": formula,
        "group": group,
        "n": len(complete),
        "groups": int(complete[group].nunique()),
        "converged": bool(fit.converged),
        "parameters": {str(key): float(value) for key, value in fit.params.items()},
        "p_values": {str(key): float(value) for key, value in fit.pvalues.items()},
        "aic": float(fit.aic),
        "bic": float(fit.bic),
    }


def require_complete_empirical_registry(audit_path: str | Path) -> dict[str, object]:
    path = Path(audit_path)
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("complete_for_analysis") is not True:
        terminal = audit.get("terminal_runs", "unknown")
        planned = audit.get("planned_runs", "unknown")
        raise RuntimeError(
            "Confirmatory analysis refused: empirical registry is incomplete "
            f"({terminal}/{planned} terminal runs)."
        )
    return audit


def to_jsonable(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
