from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import run_deterministic_smoke
from .benchmarks import prepare_gold_smoke_plan, prepare_swebench_manifests
from .experiments import (
    append_replication_plan,
    build_default_experiment_plan,
    evaluate_experiment_runs,
    merge_shard_artifacts,
    read_run_ids_file,
    recover_generated_submissions,
    run_experiment,
    write_experiment_readiness,
)
from .models import load_model_config, write_model_catalog, write_model_preflight, write_model_smoke
from .synthetic import run_synthetic_benchmark
from .version import __version__


def _run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", default="results/run_registry.csv")
    parser.add_argument("--audit", default="results/run_audit.json")
    parser.add_argument("--cost-cap-usd", type=float, default=150.0)
    parser.add_argument("--per-run-cost-cap-usd", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--run-ids-file")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--required-memory-gib", type=float, default=16.0)
    parser.add_argument("--required-disk-gib", type=float, default=120.0)
    parser.add_argument("--mark-budget-exhausted-skipped", action="store_true")
    parser.add_argument("--prune-docker-after-task", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="helloloop", description="HelloLoop uncertainty-aware loop engineering harness"
    )
    parser.add_argument("--version", action="version", version=f"helloloop {__version__}")
    commands = parser.add_subparsers(dest="command")

    synthetic = commands.add_parser("synthetic", help="Run the deterministic synthetic target")
    synthetic.add_argument("--out", default="results/runs/synthetic_ci")
    synthetic.add_argument("--seed", type=int, default=20260711)
    smoke = commands.add_parser("agent-smoke", help="Run the local mini-SWE-agent integration")
    smoke.add_argument("--out", default="results/runs/mini_swe_agent_smoke")

    prepare = commands.add_parser("prepare-data", help="Download upstream data and make manifests")
    prepare.add_argument("--cache-dir", default="data/raw/huggingface")
    prepare.add_argument("--manifest-dir", default="manifests")
    prepare.add_argument("--seed", type=int, default=20260711)
    prepare.add_argument("--max-per-repo", type=int, default=5)
    prepare.add_argument("--public-trajectory-ids")
    prepare.add_argument("--public-trajectory-repo")

    gold = commands.add_parser("gold-smoke-plan", help="Prepare an upstream evaluator smoke plan")
    gold.add_argument("--out", default="results/runs/swebench_gold_smoke")
    gold.add_argument("--instance-id", default="sympy__sympy-20590")
    gold.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    gold.add_argument("--split", default="test")
    gold.add_argument("--harness-root", default="third_party/SWE-bench")

    for name, help_text in (
        ("model-preflight", "Validate model configuration without making a request"),
        ("model-catalog", "List model identifiers from an explicitly configured endpoint"),
        ("model-smoke", "Make one bounded model request"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", default="configs/models/example_model.yaml")
        command.add_argument("--out", default=f"results/runs/model_smoke/{name}.json")

    commands.add_parser("plan-experiments", help="Build the deterministic experiment registry")
    commands.add_parser("append-replication-plan", help="Append the replication cells")
    readiness = commands.add_parser("experiment-readiness", help="Check local experiment resources")
    readiness.add_argument("--registry", default="results/run_registry.csv")
    readiness.add_argument("--out", default="results/run_health.json")
    readiness.add_argument("--cost-cap-usd", type=float)
    readiness.add_argument("--required-memory-gib", type=float, default=16.0)
    readiness.add_argument("--required-disk-gib", type=float, default=120.0)

    run = commands.add_parser("run-experiment", help="Run or resume an experiment shard")
    run.add_argument(
        "--experiment", choices=("pilot", "main", "holdout", "replication"), required=True
    )
    _run_arguments(run)
    pilot = commands.add_parser("pilot", help="Alias for run-experiment --experiment pilot")
    _run_arguments(pilot)

    merge = commands.add_parser("merge-run-artifacts", help="Merge isolated shard status records")
    merge.add_argument("--registry", default="results/run_registry.csv")
    merge.add_argument("--artifacts-dir", action="append", required=True)
    merge.add_argument("--out-registry")
    merge.add_argument("--audit", default="results/run_audit.json")
    merge.add_argument("--report", default="results/shard_merge_report.json")
    merge.add_argument("--experiment", choices=("pilot", "main", "holdout", "replication"))
    merge.add_argument("--skip-unreported", action="store_true")
    merge.add_argument("--skip-reason", default="")

    recover = commands.add_parser(
        "recover-submissions", help="Recover interrupted artifact exports"
    )
    recover.add_argument(
        "--experiment", choices=("pilot", "main", "holdout", "replication"), required=True
    )
    recover.add_argument("--registry", default="results/run_registry.csv")
    recover.add_argument("--audit", default="results/run_audit.json")

    evaluate = commands.add_parser(
        "evaluate-runs", help="Evaluate generated patches with SWE-bench"
    )
    evaluate.add_argument(
        "--experiment", choices=("pilot", "main", "holdout", "replication"), required=True
    )
    evaluate.add_argument("--registry", default="results/run_registry.csv")
    evaluate.add_argument("--audit", default="results/run_audit.json")
    evaluate.add_argument("--out", default="results/official_evaluation")
    evaluate.add_argument("--harness-root", default=".")
    evaluate.add_argument("--run-id", action="append", default=[])
    evaluate.add_argument("--run-ids-file")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--shard-index", type=int, default=0)
    evaluate.add_argument("--shard-count", type=int, default=1)
    evaluate.add_argument("--max-workers", type=int, default=1)
    evaluate.add_argument("--timeout-seconds", type=int, default=1800)
    evaluate.add_argument(
        "--cache-level", choices=("none", "base", "env", "instance"), default="env"
    )
    evaluate.add_argument("--required-memory-gib", type=float, default=16.0)
    evaluate.add_argument("--required-disk-gib", type=float, default=120.0)
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--prune-docker-after-batch", action="store_true")
    return parser


def _selected_ids(args: argparse.Namespace) -> set[str] | None:
    selected = set(args.run_id)
    if len(selected) != len(args.run_id):
        raise ValueError("duplicate --run-id values")
    if args.run_ids_file:
        file_ids = read_run_ids_file(args.run_ids_file)
        if selected.intersection(file_ids):
            raise ValueError("run IDs selected more than once")
        selected.update(file_ids)
    return selected or None


def _run_selected_experiment(args: argparse.Namespace, experiment: str) -> dict[str, object]:
    return run_experiment(
        experiment,
        registry_path=args.registry,
        audit_path=args.audit,
        cost_cap_usd=args.cost_cap_usd,
        per_run_cost_cap_usd=args.per_run_cost_cap_usd,
        limit=args.limit,
        run_ids=_selected_ids(args),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        required_memory_gib=args.required_memory_gib,
        required_disk_gib=args.required_disk_gib,
        mark_budget_exhausted_skipped=args.mark_budget_exhausted_skipped,
        prune_docker_after_task=args.prune_docker_after_task,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command
    if command == "synthetic":
        result = run_synthetic_benchmark(Path(args.out), seed=args.seed)
        print("Synthetic engineering validation only; this is not empirical model evidence.")
    elif command == "agent-smoke":
        result = run_deterministic_smoke(Path(args.out))
    elif command == "prepare-data":
        result = prepare_swebench_manifests(
            cache_dir=args.cache_dir,
            manifest_dir=args.manifest_dir,
            seed=args.seed,
            max_per_repo=args.max_per_repo,
            public_trajectory_ids_path=args.public_trajectory_ids,
            public_trajectory_repository=args.public_trajectory_repo,
        )
    elif command == "gold-smoke-plan":
        result = prepare_gold_smoke_plan(
            out_dir=args.out,
            instance_id=args.instance_id,
            dataset_name=args.dataset,
            split=args.split,
            harness_root=args.harness_root,
        )
    elif command == "model-preflight":
        result = write_model_preflight(load_model_config(args.config), args.out)
    elif command == "model-catalog":
        result = write_model_catalog(load_model_config(args.config), args.out)
    elif command == "model-smoke":
        result = write_model_smoke(load_model_config(args.config), args.out)
    elif command == "plan-experiments":
        result = build_default_experiment_plan()
    elif command == "append-replication-plan":
        result = append_replication_plan()
    elif command == "experiment-readiness":
        result = write_experiment_readiness(
            args.out,
            registry_path=args.registry,
            cost_cap_usd=args.cost_cap_usd,
            required_memory_gib=args.required_memory_gib,
            required_disk_gib=args.required_disk_gib,
        )
    elif command in {"pilot", "run-experiment"}:
        experiment = "pilot" if command == "pilot" else args.experiment
        result = _run_selected_experiment(args, experiment)
    elif command == "merge-run-artifacts":
        result = merge_shard_artifacts(
            registry_path=args.registry,
            artifact_roots=args.artifacts_dir,
            output_registry_path=args.out_registry,
            audit_path=args.audit,
            report_path=args.report,
            experiment=args.experiment,
            skip_unreported=args.skip_unreported,
            skip_reason=args.skip_reason,
        )
    elif command == "recover-submissions":
        result = recover_generated_submissions(
            args.experiment, registry_path=args.registry, audit_path=args.audit
        )
    elif command == "evaluate-runs":
        result = evaluate_experiment_runs(
            args.experiment,
            registry_path=args.registry,
            audit_path=args.audit,
            output_root=args.out,
            harness_root=args.harness_root,
            run_ids=_selected_ids(args),
            limit=args.limit,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            max_workers=args.max_workers,
            timeout_seconds=args.timeout_seconds,
            cache_level=args.cache_level,
            required_memory_gib=args.required_memory_gib,
            required_disk_gib=args.required_disk_gib,
            batch_size=args.batch_size,
            prune_docker_after_batch=args.prune_docker_after_batch,
        )
    else:
        parser.print_help()
        return 0
    print(json.dumps(result, indent=2, sort_keys=True))
    if command in {"pilot", "run-experiment"} and result.get("status") == "blocked":
        return 2
    if command == "evaluate-runs" and int(result.get("failed", 0)):
        return 3
    return 0
