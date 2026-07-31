from __future__ import annotations

import csv

from helloloop.synthetic import SYNTHETIC_LABEL, default_tasks, run_synthetic_benchmark


def test_default_tasks_cover_24_cases() -> None:
    tasks = default_tasks()
    assert len(tasks) == 24
    assert len({task.category for task in tasks}) == 12


def test_synthetic_benchmark_writes_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = run_synthetic_benchmark(tmp_path / "synthetic", seed=20260711)
    assert result["runs"] == 96
    summary = tmp_path / "synthetic" / "summaries" / "summary.csv"
    with summary.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 96
    assert (tmp_path / "synthetic" / "figures" / "calibration_plot.svg").exists()
    assert SYNTHETIC_LABEL in (tmp_path / "synthetic" / "README.txt").read_text(encoding="utf-8")
