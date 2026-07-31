from __future__ import annotations

import csv
import json
from pathlib import Path

from helloloop.analysis.sensitivity import (
    UNCERTAINTY_COMPONENTS,
    generate_sensitivity_analysis,
)


def _record(index: int, resolved: bool) -> dict[str, object]:
    center = 0.2 if resolved else 0.7
    return {
        "run_id": f"run-{index}",
        "resolved": resolved,
        "uncertainty": {
            component: min(1.0, center + component_index * 0.01)
            for component_index, component in enumerate(UNCERTAINTY_COMPONENTS)
        },
    }


def test_sensitivity_analysis_is_deterministic_and_scoped(tmp_path: Path) -> None:
    records = [_record(index, index % 2 == 0) for index in range(20)]

    first = generate_sensitivity_analysis(records, tmp_path / "first", perturbation_repetitions=20)
    second = generate_sensitivity_analysis(
        records, tmp_path / "second", perturbation_repetitions=20
    )

    assert first == second
    assert first["ablation_configurations"] == 11
    assert first["robustness_perturbations"] == 12
    assert (tmp_path / "first/estimator_ablations.csv").read_bytes() == (
        tmp_path / "second/estimator_ablations.csv"
    ).read_bytes()
    assert (tmp_path / "first/estimator_robustness.csv").read_bytes() == (
        tmp_path / "second/estimator_robustness.csv"
    ).read_bytes()
    with (tmp_path / "first/estimator_ablations.csv").open(encoding="utf-8", newline="") as handle:
        ablations = list(csv.DictReader(handle))
    with (tmp_path / "first/estimator_robustness.csv").open(encoding="utf-8", newline="") as handle:
        robustness = list(csv.DictReader(handle))
    assert {row["analysis"] for row in ablations} >= {
        "default_weighted_average",
        "drop_verification",
        "maximum_composite",
    }
    assert all("not_agent_rerun" in row["scope"] for row in ablations)
    assert all(0.0 <= float(row["gate_flip_rate_mean"]) <= 1.0 for row in robustness)
    status = json.loads((tmp_path / "first/sensitivity_status.json").read_text())
    assert status["n"] == 20


def test_sensitivity_analysis_rejects_single_class_outcomes(tmp_path: Path) -> None:
    records = [_record(index, True) for index in range(4)]

    try:
        generate_sensitivity_analysis(records, tmp_path)
    except ValueError as error:
        assert "both resolved and unresolved" in str(error)
    else:  # pragma: no cover
        raise AssertionError("single-class sensitivity input should fail")
