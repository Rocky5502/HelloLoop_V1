from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATASET_SPECS = (
    {
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "split": "dev",
        "count": 12,
        "filename": "swebench_lite_dev_12.json",
    },
    {
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "split": "test",
        "count": 40,
        "filename": "swebench_lite_test_40.json",
    },
    {
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "count": 20,
        "filename": "swebench_verified_holdout_20.json",
    },
)

PROHIBITED_FIELDS = frozenset(
    {
        "patch",
        "gold_patch",
        "test_patch",
        "hints_text",
        "fail_to_pass",
        "pass_to_pass",
        "evaluator_logs",
        "hidden_tests",
        "solution_trace",
        "public_solution_trace",
    }
)
SAFE_TASK_FIELDS = frozenset(
    {"task_id", "repo", "base_commit", "problem_statement", "version", "stratification"}
)
INSTANCE_ID_RE = re.compile(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+")


class LeakageError(ValueError):
    """Raised when restricted benchmark information reaches an inference payload."""


def _normalized_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _walk_keys(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_key(key)
            child_path = f"{path}.{key}"
            yield normalized, child_path
            yield from _walk_keys(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_keys(child, f"{path}[{index}]")


def assert_inference_payload_safe(payload: Mapping[str, Any]) -> None:
    violations = [path for key, path in _walk_keys(payload) if key in PROHIBITED_FIELDS]
    if violations:
        raise LeakageError(
            "restricted benchmark fields in inference payload: " + ", ".join(violations)
        )


def _parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _issue_length_bucket(problem_statement: str) -> str:
    length = len(problem_statement)
    if length < 1_000:
        return "short"
    if length < 3_000:
        return "medium"
    return "long"


def _created_at_bucket(created_at: str | None) -> str:
    if not created_at:
        return "unknown"
    value = str(created_at)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value[:7] if len(value) >= 7 else "unknown"
    return f"{parsed.year}-Q{(parsed.month - 1) // 3 + 1}"


def _patch_statistics(patch: str) -> tuple[int, int]:
    files = set()
    changed_lines = 0
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                files.add(parts[2].removeprefix("a/"))
        elif (line.startswith("+") and not line.startswith("+++")) or (
            line.startswith("-") and not line.startswith("---")
        ):
            changed_lines += 1
    return len(files), changed_lines


def _stable_rank(seed: int, instance_id: str) -> str:
    return hashlib.sha256(f"{seed}:{instance_id}".encode()).hexdigest()


def _select_rows(
    rows: list[Mapping[str, Any]],
    count: int,
    seed: int,
    max_per_repo: int,
) -> list[Mapping[str, Any]]:
    if count > len(rows):
        raise ValueError(f"requested {count} rows from a split containing {len(rows)}")
    ranked = sorted(rows, key=lambda row: _stable_rank(seed, str(row["instance_id"])))
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ranked:
        buckets[_issue_length_bucket(str(row["problem_statement"]))].append(row)

    selected: list[Mapping[str, Any]] = []
    repo_counts: Counter[str] = Counter()
    indices = {bucket: 0 for bucket in buckets}
    bucket_order = [bucket for bucket in ("short", "medium", "long") if bucket in buckets]
    while len(selected) < count:
        made_progress = False
        for bucket in bucket_order:
            candidates = buckets[bucket]
            while indices[bucket] < len(candidates):
                candidate = candidates[indices[bucket]]
                indices[bucket] += 1
                repo = str(candidate["repo"])
                if repo_counts[repo] >= max_per_repo:
                    continue
                selected.append(candidate)
                repo_counts[repo] += 1
                made_progress = True
                break
            if len(selected) == count:
                break
        if not made_progress:
            raise ValueError(
                f"repository cap {max_per_repo} cannot produce {count} tasks; "
                "raise the cap explicitly"
            )
    return selected


def _safe_task(
    row: Mapping[str, Any],
    *,
    public_trajectory_ids: set[str] | None,
) -> dict[str, Any]:
    patch = str(row.get("patch", ""))
    files_touched, patch_size_proxy = _patch_statistics(patch)
    fail_to_pass = _parse_list(row.get("FAIL_TO_PASS", []))
    pass_to_pass = _parse_list(row.get("PASS_TO_PASS", []))
    instance_id = str(row["instance_id"])
    public_status = "unknown"
    if public_trajectory_ids is not None:
        public_status = "present" if instance_id in public_trajectory_ids else "not_found"
    task = {
        "task_id": instance_id,
        "repo": str(row["repo"]),
        "base_commit": str(row["base_commit"]),
        "problem_statement": str(row["problem_statement"]),
        "version": str(row.get("version", "")),
        "stratification": {
            "issue_chars": len(str(row["problem_statement"])),
            "issue_length_bucket": _issue_length_bucket(str(row["problem_statement"])),
            "created_at_bucket": _created_at_bucket(row.get("created_at")),
            "patch_size_proxy": patch_size_proxy,
            "files_touched": files_touched,
            "test_cardinality": len(fail_to_pass) + len(pass_to_pass),
            "difficulty": row.get("difficulty"),
            "public_trajectory_status": public_status,
        },
    }
    assert_inference_payload_safe(task)
    return task


def build_manifest(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    split: str,
    count: int,
    seed: int = 20260711,
    max_per_repo: int = 5,
    public_trajectory_ids: set[str] | None = None,
) -> dict[str, Any]:
    materialized = list(rows)
    selected = _select_rows(materialized, count, seed, max_per_repo)
    tasks = [_safe_task(row, public_trajectory_ids=public_trajectory_ids) for row in selected]
    manifest = {
        "schema_id": "helloloop.swebench_manifest.v1",
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "selection_algorithm": "sha256_rank_round_robin_issue_length_v1",
        "max_per_repo": max_per_repo,
        "task_count": len(tasks),
        "restricted_fields_removed": sorted(PROHIBITED_FIELDS),
        "tasks": tasks,
    }
    assert_inference_payload_safe({"tasks": tasks})
    return manifest


def task_to_inference_payload(task: Mapping[str, Any]) -> dict[str, str]:
    payload = {
        "task_id": str(task["task_id"]),
        "repo": str(task["repo"]),
        "base_commit": str(task["base_commit"]),
        "problem_statement": str(task["problem_statement"]),
        "version": str(task.get("version", "")),
    }
    assert_inference_payload_safe(payload)
    return payload


def audit_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    tasks = list(manifest.get("tasks", []))
    expected_count = int(manifest.get("task_count", -1))
    task_ids = [str(task.get("task_id", "")) for task in tasks]
    repo_counts = Counter(str(task.get("repo", "")) for task in tasks)
    violations: list[str] = []
    if expected_count != len(tasks):
        violations.append(f"task_count declares {expected_count}, found {len(tasks)}")
    if len(set(task_ids)) != len(task_ids):
        violations.append("duplicate task IDs")
    if any(not task_id for task_id in task_ids):
        violations.append("missing task ID")
    max_per_repo = int(manifest.get("max_per_repo", 0))
    if max_per_repo and any(count > max_per_repo for count in repo_counts.values()):
        violations.append("repository cap exceeded")
    for index, task in enumerate(tasks):
        extra_fields = set(task) - SAFE_TASK_FIELDS
        if extra_fields:
            violations.append(f"task {index} has unsafe keys: {sorted(extra_fields)}")
        try:
            assert_inference_payload_safe(task_to_inference_payload(task))
        except (KeyError, LeakageError) as error:
            violations.append(f"task {index} inference payload: {error}")
    public_status = Counter(
        str(task.get("stratification", {}).get("public_trajectory_status", "missing"))
        for task in tasks
    )
    return {
        "dataset": manifest.get("dataset"),
        "split": manifest.get("split"),
        "task_count": len(tasks),
        "unique_task_count": len(set(task_ids)),
        "repository_counts": dict(sorted(repo_counts.items())),
        "issue_length_counts": dict(
            sorted(
                Counter(
                    str(task.get("stratification", {}).get("issue_length_bucket", "missing"))
                    for task in tasks
                ).items()
            )
        ),
        "public_trajectory_status": dict(sorted(public_status.items())),
        "violations": violations,
        "status": "pass" if not violations else "fail",
    }


def _load_public_ids(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.exists():
        return None
    values = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(values, dict):
        values = values.get("instance_ids", [])
    return {str(value) for value in values}


def extract_public_trajectory_ids(
    repository: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    repo = Path(repository)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    paths = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    instance_ids = sorted(set(INSTANCE_ID_RE.findall(paths)))
    if not instance_ids:
        raise ValueError(f"no SWE-bench instance IDs found in Git tree at {repo}")
    payload = {
        "schema_id": "helloloop.public_trajectory_index.v1",
        "source": "https://github.com/SWE-bench/experiments.git",
        "source_commit": commit,
        "extraction": "instance ID regex over git ls-tree paths; no blobs checked out",
        "instance_count": len(instance_ids),
        "instance_ids": instance_ids,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def prepare_swebench_manifests(
    *,
    cache_dir: str | Path = "data/raw/huggingface",
    manifest_dir: str | Path = "manifests",
    seed: int = 20260711,
    max_per_repo: int = 5,
    public_trajectory_ids_path: str | Path | None = None,
    public_trajectory_repository: str | Path | None = None,
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise RuntimeError("install HelloLoop with the 'data' optional dependencies") from error

    cache_path = Path(cache_dir)
    output_path = Path(manifest_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    if public_trajectory_repository is not None:
        generated_index = output_path / "public_trajectory_instance_ids.json"
        extract_public_trajectory_ids(public_trajectory_repository, generated_index)
        public_trajectory_ids_path = generated_index
    public_ids = _load_public_ids(public_trajectory_ids_path)
    lock_entries = []
    audit_entries = []
    outputs = []
    for spec in DATASET_SPECS:
        dataset = load_dataset(
            spec["dataset"],
            split=spec["split"],
            cache_dir=str(cache_path),
        )
        manifest = build_manifest(
            dataset,
            dataset=spec["dataset"],
            split=spec["split"],
            count=spec["count"],
            seed=seed,
            max_per_repo=max_per_repo,
            public_trajectory_ids=public_ids,
        )
        target = output_path / spec["filename"]
        target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        lock_entries.append(
            {
                "dataset": spec["dataset"],
                "split": spec["split"],
                "source_rows": len(dataset),
                "dataset_fingerprint": getattr(dataset, "_fingerprint", "unknown"),
                "manifest": spec["filename"],
                "manifest_sha256": manifest_sha256,
                "selected_rows": spec["count"],
            }
        )
        audit_entries.append({"manifest": spec["filename"], **audit_manifest(manifest)})
        outputs.append(str(target))

    lock = {
        "schema_id": "helloloop.dataset_lock.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "max_per_repo": max_per_repo,
        "public_trajectory_audit": "available" if public_ids is not None else "unavailable",
        "datasets": lock_entries,
    }
    lock_path = output_path / "datasets.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "schema_id": "helloloop.leakage_audit.v1",
        "status": "pass" if all(item["status"] == "pass" for item in audit_entries) else "fail",
        "prohibited_fields": sorted(PROHIBITED_FIELDS),
        "public_trajectory_archive_available": public_ids is not None,
        "manifests": audit_entries,
    }
    audit_path = output_path / "leakage_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["status"] != "pass":
        raise LeakageError(f"manifest leakage audit failed; see {audit_path}")
    return {
        "manifests": outputs,
        "lock": str(lock_path),
        "audit": str(audit_path),
        "datasets": lock_entries,
    }
