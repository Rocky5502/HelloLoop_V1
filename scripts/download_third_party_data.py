#!/usr/bin/env python3
"""Explicitly download upstream benchmark data; no data is downloaded on import."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download upstream data under its source terms into ignored data/raw."
    )
    parser.add_argument("--cache-dir", default="data/raw/huggingface")
    parser.add_argument("--manifest-dir", default="manifests")
    parser.add_argument("--max-per-repo", type=int, default=5)
    args = parser.parse_args()
    from helloloop.benchmarks import prepare_swebench_manifests

    result = prepare_swebench_manifests(
        cache_dir=args.cache_dir,
        manifest_dir=args.manifest_dir,
        seed=20260711,
        max_per_repo=args.max_per_repo,
    )
    print(result)


if __name__ == "__main__":
    main()
