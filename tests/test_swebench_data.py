from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from helloloop.benchmarks.swebench_data import (
    PROHIBITED_FIELDS,
    LeakageError,
    assert_inference_payload_safe,
    audit_manifest,
    build_manifest,
    extract_public_trajectory_ids,
    task_to_inference_payload,
)


def _rows(count: int = 18) -> list[dict]:
    rows = []
    for index in range(count):
        issue = "x" * (300 + (index % 3) * 1_400)
        rows.append(
            {
                "instance_id": f"repo{index % 6}__task-{index:03d}",
                "repo": f"org/repo{index % 6}",
                "base_commit": f"commit-{index:03d}",
                "problem_statement": issue,
                "version": "1.0",
                "created_at": "2024-02-01T00:00:00Z",
                "patch": "diff --git a/a.py b/a.py\n-old\n+new\n",
                "test_patch": "secret tests",
                "hints_text": "secret hint",
                "FAIL_TO_PASS": json.dumps(["test_failure"]),
                "PASS_TO_PASS": json.dumps(["test_regression"]),
            }
        )
    return rows


def test_manifest_is_deterministic_stratified_and_leakage_safe() -> None:
    first = build_manifest(
        _rows(),
        dataset="fixture/swebench",
        split="dev",
        count=12,
        max_per_repo=2,
    )
    second = build_manifest(
        _rows(),
        dataset="fixture/swebench",
        split="dev",
        count=12,
        max_per_repo=2,
    )
    assert first == second
    assert first["task_count"] == 12
    assert {task["stratification"]["issue_length_bucket"] for task in first["tasks"]} == {
        "short",
        "medium",
        "long",
    }
    encoded = json.dumps(first).lower()
    for prohibited in PROHIBITED_FIELDS:
        assert (
            f'"{prohibited.lower()}"' not in encoded
            or prohibited in first["restricted_fields_removed"]
        )
    for task in first["tasks"]:
        assert_inference_payload_safe(task_to_inference_payload(task))
    audit = audit_manifest(first)
    assert audit["status"] == "pass"
    assert audit["unique_task_count"] == 12


@pytest.mark.parametrize(
    "payload",
    [
        {"hints_text": "do this"},
        {"metadata": {"FAIL_TO_PASS": ["test_x"]}},
        {"messages": [{"test_patch": "hidden"}]},
        {"gold-patch": "hidden"},
    ],
)
def test_leakage_guard_rejects_nested_restricted_fields(payload: dict) -> None:
    with pytest.raises(LeakageError):
        assert_inference_payload_safe(payload)


def test_repository_cap_failure_is_explicit() -> None:
    with pytest.raises(ValueError, match="repository cap"):
        build_manifest(
            _rows(6),
            dataset="fixture/swebench",
            split="dev",
            count=6,
            max_per_repo=0,
        )


def test_manifest_audit_detects_duplicate_ids() -> None:
    manifest = build_manifest(
        _rows(),
        dataset="fixture/swebench",
        split="dev",
        count=3,
    )
    manifest["tasks"][1]["task_id"] = manifest["tasks"][0]["task_id"]
    audit = audit_manifest(manifest)
    assert audit["status"] == "fail"
    assert "duplicate task IDs" in audit["violations"]


def test_public_trajectory_ids_are_extracted_without_checkout(tmp_path: Path) -> None:
    repo = tmp_path / "archive"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    path = repo / "evaluation" / "org__repo-123" / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=HelloLoop CI",
            "-c",
            "user.email=ci@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    result = extract_public_trajectory_ids(repo, tmp_path / "ids.json")
    assert result["instance_ids"] == ["org__repo-123"]
