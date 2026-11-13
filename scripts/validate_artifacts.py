#!/usr/bin/env python3
"""Validate public files, schemas, checksums, and locked empirical counts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "data/DATA_CARD.md",
        "data/processed/run_level.csv",
        "results/processed/grouped_cv_predictions.csv",
        "results/processed/expected_metrics.json",
        "docs/figures/helloloop_framework.png",
        "docs/FIGURE_MANIFEST.md",
        "docs/TABLE_MANIFEST.md",
        "results/checksums/SHA256SUMS",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"required public files are missing: {', '.join(missing)}")

    with (ROOT / "data/processed/run_level.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 316:
        raise AssertionError(f"expected 316 registered rows, found {len(rows)}")
    expected = json.loads((ROOT / "results/processed/expected_metrics.json").read_text())
    checks = {
        "evaluated_runs": sum(int(row["evaluator_reached"]) for row in rows),
        "patches": sum(int(row["patch_submitted"]) for row in rows),
        "evaluated_abstentions": sum(int(row["abstention"]) for row in rows),
        "resolved": sum(int(row["registered_success"]) for row in rows),
    }
    for key, value in checks.items():
        if value != expected[key]:
            raise AssertionError(f"locked count mismatch for {key}: {value} != {expected[key]}")

    checksum_lines = (ROOT / "results/checksums/SHA256SUMS").read_text().splitlines()
    if not checksum_lines:
        raise AssertionError("checksum manifest is empty")
    for line in checksum_lines:
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or sha256(path) != digest:
            raise AssertionError(f"checksum mismatch: {relative}")
    print(f"artifact validation passed: {len(rows)} run rows, {len(checksum_lines)} checksums")


if __name__ == "__main__":
    main()
