from __future__ import annotations

import json

import pytest

from helloloop.analysis.statistics import (
    calibration_metrics,
    cluster_trajectories,
    compare_paired_binary,
    compare_paired_continuous,
    event_centered_analysis,
    fit_linear_mixed_effects,
    holm_adjust,
    paired_assumption_checks,
    pareto_frontier,
    require_complete_empirical_registry,
    risk_coverage_curve,
    spearman_correlation,
    summarize_stages,
    transition_matrix,
    uncertainty_volatility,
)


def test_paired_binary_reports_exact_counts_and_effect() -> None:
    result = compare_paired_binary(
        [1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        variant_a="verification",
        variant_b="linear",
        missing_pairs=2,
        resamples=500,
        seed=3,
    )
    assert result.n_pairs == 5
    assert result.missing_pairs == 2
    assert result.a_success_b_failure == 2
    assert result.a_failure_b_success == 1
    assert result.risk_difference.estimate == pytest.approx(0.2)
    assert 0.0 <= result.mcnemar_exact_p <= 1.0


def test_continuous_and_multiple_testing_statistics() -> None:
    result = compare_paired_continuous(
        [2.0, 3.0, 5.0, 7.0],
        [3.0, 3.0, 4.0, 5.0],
        variant_a="a",
        variant_b="b",
        resamples=500,
    )
    assert result.n_pairs == 4
    assert result.mean_difference.estimate == pytest.approx(0.5)
    assert result.rank_biserial is not None
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert spearman_correlation([1, 2, 3], [3, 2, 1])["rho"] == pytest.approx(-1.0)


def test_calibration_and_risk_coverage_keep_source_counts() -> None:
    result = calibration_metrics([0.9, 0.8, 0.4, 0.1], [1, 1, 0, 0], n_bins=2)
    assert result.n == 4
    assert result.positives == 2
    assert result.auroc == pytest.approx(1.0)
    assert sum(item.n for item in result.equal_width_bins) == 4
    curve = risk_coverage_curve([0.9, 0.8, 0.4, 0.1], [1, 1, 0, 0])
    assert curve[0].coverage == pytest.approx(0.25)
    assert curve[0].selective_accuracy == pytest.approx(1.0)
    assert curve[-1].coverage == pytest.approx(1.0)


def test_stage_and_volatility_summaries() -> None:
    stages = summarize_stages([("retrieve", 0.8), ("retrieve", 0.6), ("verify", 0.2)])
    assert stages[0].n == 2
    assert stages[0].mean == pytest.approx(0.7)
    assert uncertainty_volatility([0.8, 0.5, 0.6])["mean_absolute_delta"] == pytest.approx(0.2)


def test_trajectory_and_event_analysis() -> None:
    transitions = transition_matrix([["plan", "retrieve", "execute"], ["plan", "execute"]])
    assert sum(item.count for item in transitions if item.source == "plan") == 2
    event = event_centered_analysis(
        [[("plan", 0.8), ("retrieve", 0.7), ("execute", 0.4)]], event="retrieve"
    )
    assert event["complete_windows"] == 1
    assert event["mean_delta"] == pytest.approx(-0.4)
    clusters = cluster_trajectories(
        {"down-a": [0.9, 0.5, 0.1], "down-b": [0.8, 0.4, 0.2], "up": [0.1, 0.4, 0.9]},
        n_clusters=2,
    )
    assert len(clusters) == 3
    assert all("volatility" in row for row in clusters)


def test_pareto_assumptions_and_mixed_effects() -> None:
    frontier = pareto_frontier([("a", 1.0, 0.5), ("b", 2.0, 0.4), ("c", 2.0, 0.8)])
    assert [item.label for item in frontier] == ["a", "c"]
    assert "small_sample" in paired_assumption_checks([1.0, 0.0, -1.0])["warnings"]
    records = []
    for task_index in range(8):
        for treatment in (0, 1):
            records.append(
                {
                    "task_id": f"task-{task_index}",
                    "outcome": task_index * 0.1 + treatment * 0.5,
                    "treatment": treatment,
                }
            )
    model = fit_linear_mixed_effects(
        records, outcome="outcome", fixed_effects=["treatment"], group="task_id"
    )
    assert model["n"] == 16
    assert model["parameters"]["treatment"] == pytest.approx(0.5, abs=0.05)


def test_confirmatory_gate_refuses_incomplete_registry(tmp_path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "complete_for_analysis": False,
                "terminal_runs": 0,
                "planned_runs": 4,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="0/4 terminal"):
        require_complete_empirical_registry(path)


def test_confirmatory_gate_accepts_complete_registry(tmp_path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"status": "pass", "complete_for_analysis": True}), encoding="utf-8")
    assert require_complete_empirical_registry(path)["complete_for_analysis"] is True
