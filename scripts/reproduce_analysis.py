#!/usr/bin/env python3
"""Reproduce public metrics and figures from frozen, redacted CSV inputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data/processed/run_level.csv"
PREDICTIONS = ROOT / "results/processed/grouped_cv_predictions.csv"
EXPECTED = ROOT / "results/processed/expected_metrics.json"
OUTPUT = ROOT / "results/generated"
SEED = 20260711


def _close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0):
        raise AssertionError(f"metric mismatch: observed {actual:.12g}, expected {expected:.12g}")


def _policy_rows(runs: pd.DataFrame, experiment: str) -> list[dict[str, object]]:
    selected = runs[runs["experiment"] == experiment]
    rows = []
    for policy, group in selected.groupby("loop_variant", sort=True):
        resolved = int(group["registered_success"].sum())
        registered = len(group)
        rows.append(
            {
                "experiment": experiment,
                "policy": policy,
                "resolved": resolved,
                "registered": registered,
                "resolution_rate": resolved / registered,
            }
        )
    return rows


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    bin_index = np.minimum((probability * bins).astype(int), bins - 1)
    total = 0.0
    for index in range(bins):
        selected = bin_index == index
        if selected.any():
            total += selected.mean() * abs(y[selected].mean() - probability[selected].mean())
    return float(total)


def _save_resolution_figure(table: pd.DataFrame, output: Path) -> None:
    main = table[table["experiment"] == "swebench_lite_main"]
    order = ["linear", "reflection", "verification", "uncertainty_aware"]
    main = main.set_index("policy").loc[order]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = ["#2364AA", "#3DA35D", "#F2A541", "#C44536"]
    ax.bar(["L0", "L1", "L2", "L3"], main["resolution_rate"] * 100, color=colors)
    ax.set_ylabel("Registered resolution (%)")
    ax.set_ylim(0, 65)
    ax.set_title("Lite main policy comparison")
    for index, value in enumerate(main["resolution_rate"] * 100):
        ax.text(index, value + 1.2, f"{value:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(output, dpi=180, metadata={"Software": "HelloLoop"})
    plt.close(fig)


def _save_cost_figure(cost: dict[str, float], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    values = [cost["measured_total_usd"], cost["reserved_pre_model_usd"]]
    ax.bar(
        ["Token-derived measured", "Pre-model reservations"], values, color=["#2364AA", "#F2A541"]
    )
    ax.set_ylabel("USD")
    ax.set_title("Cost-accounting components")
    for index, value in enumerate(values):
        ax.text(index, value + 1, f"${value:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(output, dpi=180, metadata={"Software": "HelloLoop"})
    plt.close(fig)


def reproduce(clean: bool = False) -> dict[str, object]:
    for required in (RUNS, PREDICTIONS, EXPECTED):
        if not required.is_file():
            raise FileNotFoundError(
                f"required frozen input is missing: {required.relative_to(ROOT)}"
            )
    if OUTPUT.exists():
        if not clean:
            raise FileExistsError("results/generated already exists; pass --clean to replace it")
        shutil.rmtree(OUTPUT)
    (OUTPUT / "tables").mkdir(parents=True)
    (OUTPUT / "figures").mkdir(parents=True)

    np.random.seed(SEED)
    runs = pd.read_csv(RUNS)
    predictions = pd.read_csv(PREDICTIONS)
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))

    terminal = runs["terminal_classification"].value_counts()
    summary: dict[str, object] = {
        "seed": SEED,
        "registered_runs": len(runs),
        "evaluated_runs": int(runs["evaluator_reached"].sum()),
        "patches": int(runs["patch_submitted"].sum()),
        "evaluated_abstentions": int(runs["abstention"].sum()),
        "resolved": int(runs["registered_success"].sum()),
        "agent_failures": int(terminal.get("agent_failure", 0)),
        "infrastructure_failures": int(terminal.get("infrastructure_failure", 0)),
        "budget_stops": int(terminal.get("budget_stop", 0)),
    }
    for key in (
        "registered_runs",
        "evaluated_runs",
        "patches",
        "evaluated_abstentions",
        "resolved",
        "agent_failures",
        "infrastructure_failures",
        "budget_stops",
    ):
        if summary[key] != expected[key]:
            raise AssertionError(f"count mismatch for {key}: {summary[key]} != {expected[key]}")

    policy_rows = []
    for experiment in sorted(runs["experiment"].unique()):
        policy_rows.extend(_policy_rows(runs, experiment))
    policy_table = pd.DataFrame(policy_rows)
    policy_table.to_csv(OUTPUT / "tables/policy_resolution.csv", index=False)

    for experiment, expected_key in (
        ("swebench_lite_main", "main"),
        ("swebench_lite_replication", "replication"),
        ("swebench_verified_holdout", "holdout"),
    ):
        observed = policy_table[policy_table["experiment"] == experiment].set_index("policy")
        for policy, value in expected[expected_key].items():
            if isinstance(value, list):
                counts = [
                    int(observed.loc[policy, "resolved"]),
                    int(observed.loc[policy, "registered"]),
                ]
                if counts != value:
                    raise AssertionError(f"policy count mismatch for {experiment}/{policy}")

    y = predictions["registered_success"].to_numpy(dtype=float)
    frozen = predictions["frozen_unweighted_mean"].to_numpy(dtype=float)
    full = predictions["logistic_all_components"].to_numpy(dtype=float)
    signal = {
        "rows": len(predictions),
        "brier": float(np.mean((y - frozen) ** 2)),
        "ece": _ece(y, frozen),
        "frozen_auroc": float(roc_auc_score(y, frozen)),
        "full_vector_auroc": float(roc_auc_score(y, full)),
    }
    signal["difference"] = signal["full_vector_auroc"] - signal["frozen_auroc"]
    for key in ("brier", "ece", "frozen_auroc", "full_vector_auroc", "difference"):
        _close(signal[key], expected["signal"][key])
    summary["signal"] = signal

    cost = {
        "budget_total_usd": float(runs["budget_cost_usd"].sum()),
        "reserved_pre_model_usd": float(runs["reserved_pre_model_usd"].sum()),
        "measured_total_usd": float(runs["measured_full_rate_usd"].sum()),
    }
    for key, value in cost.items():
        _close(value, expected["cost"][key], tolerance=1e-7)
    summary["cost"] = cost
    summary["locked_inference"] = {
        "main_mcnemar_p": expected["main"]["mcnemar_p"],
        "replication_mcnemar_p": expected["replication"]["mcnemar_p"],
        "both_lite_clustered_ci": expected["both_lite"]["clustered_ci"],
        "both_lite_sign_flip_p": expected["both_lite"]["sign_flip_p"],
        "holdout_mcnemar_p": expected["holdout"]["mcnemar_p"],
        "signal_clustered_ci": expected["signal"]["clustered_ci"],
        "signal_permutation_p": expected["signal"]["permutation_p"],
    }

    _save_resolution_figure(policy_table, OUTPUT / "figures/policy_resolution.png")
    _save_cost_figure(cost, OUTPUT / "figures/cost_components.png")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="replace only results/generated")
    arguments = parser.parse_args()
    print(json.dumps(reproduce(clean=arguments.clean), indent=2, sort_keys=True))
