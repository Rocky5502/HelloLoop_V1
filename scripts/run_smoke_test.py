#!/usr/bin/env python3
"""Run the deterministic synthetic integration target without network access."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from helloloop.synthetic import run_synthetic_benchmark


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="helloloop-smoke-") as directory:
        result = run_synthetic_benchmark(Path(directory), seed=20260711)
        if result["runs"] != 96:
            raise AssertionError(f"expected 96 synthetic runs, observed {result['runs']}")
        expected = Path(directory) / "tables/loop_comparison.csv"
        if not expected.is_file():
            raise AssertionError("synthetic policy comparison was not generated")
        print(json.dumps({"status": "passed", "synthetic_runs": 96}, sort_keys=True))


if __name__ == "__main__":
    main()
