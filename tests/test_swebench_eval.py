from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from helloloop.benchmarks.swebench_eval import (
    HarnessConfig,
    SWEbenchPrediction,
    build_harness_command,
    classify_evaluator_log,
    copy_evaluator_artifacts,
    evaluation_complete,
    parse_instance_report,
    parse_run_report,
    prepare_gold_smoke_plan,
    read_predictions_jsonl,
    write_predictions_jsonl,
)


def test_prediction_round_trip_uses_official_keys(tmp_path: Path) -> None:
    path = write_predictions_jsonl(
        [
            SWEbenchPrediction("org__repo-2", "diff two", "model/x"),
            SWEbenchPrediction("org__repo-1", "diff one", "model/x"),
        ],
        tmp_path / "prediction.jsonl",
    )
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert list(lines[0]) == ["instance_id", "model_name_or_path", "model_patch"]
    assert [item.instance_id for item in read_predictions_jsonl(path)] == [
        "org__repo-1",
        "org__repo-2",
    ]

    pytest.importorskip("swebench")
    from swebench.harness.utils import get_predictions_from_file

    officially_loaded = get_predictions_from_file(str(path), "unused", "unused")
    assert officially_loaded == lines


def test_prediction_export_rejects_duplicates_and_secrets(tmp_path: Path) -> None:
    duplicate = SWEbenchPrediction("same", "diff", "model")
    with pytest.raises(ValueError, match="duplicate"):
        write_predictions_jsonl([duplicate, duplicate], tmp_path / "duplicate.jsonl")
    with pytest.raises(ValueError, match="secret-like"):
        fixture = "s" + "k-" + "testcredential123456789"
        SWEbenchPrediction("id", f"token {fixture}", "model").validate()
    SWEbenchPrediction(
        "id",
        "+    value = salted_hmac(key, payload, secret=self.secret)",
        "model",
    ).validate()


def test_harness_command_targets_official_module() -> None:
    command = build_harness_command(
        HarnessConfig(
            dataset_name="princeton-nlp/SWE-bench_Lite",
            split="dev",
            predictions_path="gold",
            run_id="gold-smoke",
            instance_ids=("org__repo-1",),
        )
    )
    assert command[:3] == [sys.executable, "-m", "swebench.harness.run_evaluation"]
    assert command[-2:] == ["--instance_ids", "org__repo-1"]


def test_official_report_parsers_and_resume(tmp_path: Path) -> None:
    report = {
        "total_instances": 1,
        "submitted_instances": 1,
        "completed_instances": 1,
        "resolved_instances": 1,
        "unresolved_instances": 0,
        "empty_patch_instances": 0,
        "error_instances": 0,
        "completed_ids": ["task-1"],
        "incomplete_ids": [],
        "empty_patch_ids": [],
        "submitted_ids": ["task-1"],
        "resolved_ids": ["task-1"],
        "unresolved_ids": [],
        "error_ids": [],
        "schema_version": 2,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    outcome = parse_run_report(path)
    assert outcome.success_rate == 1.0
    assert evaluation_complete(path, "task-1")
    instance_path = tmp_path / "instance.json"
    instance_path.write_text(json.dumps({"task-1": {"resolved": True}}), encoding="utf-8")
    assert parse_instance_report(instance_path, "task-1")
    assert evaluation_complete(instance_path, "task-1")


def test_log_classification_and_artifact_copy(tmp_path: Path) -> None:
    log = tmp_path / "test_output.txt"
    log.write_text(">>>>> Some Tests Failed\n", encoding="utf-8")
    classification = classify_evaluator_log(log)
    assert classification["agent_failure"]
    assert not classification["infrastructure_failure"]

    source = tmp_path / "harness" / "logs" / "run_evaluation" / "run" / "model__x" / "task"
    source.mkdir(parents=True)
    (source / "report.json").write_text("{}", encoding="utf-8")
    copied = copy_evaluator_artifacts(
        harness_root=tmp_path / "harness",
        run_id="run",
        model_name="model/x",
        instance_id="task",
        destination=tmp_path / "bundle",
    )
    assert copied == ["report.json"]


def test_gold_smoke_plan_retains_preflight_and_command(tmp_path: Path) -> None:
    result = prepare_gold_smoke_plan(
        out_dir=tmp_path / "gold",
        instance_id="org__repo-1",
        harness_root=tmp_path,
    )
    assert result["status"] in {"ready", "blocked"}
    assert (tmp_path / "gold" / "preflight.json").exists()
    command = json.loads((tmp_path / "gold" / "command.json").read_text(encoding="utf-8"))
    assert command["official_evaluator"] is True
    assert "SWE-bench/SWE-bench_Lite" in command["argv"]
    assert "test" in command["argv"]
