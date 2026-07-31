from __future__ import annotations

import csv

import pytest

from helloloop.enums import Decision, LoopVariant, Stage
from helloloop.schema import (
    JsonlTrajectoryWriter,
    StepEvidence,
    StepRecord,
    UncertaintyVector,
    read_trajectory,
    redact_credentials,
    summarize_trajectory,
    write_summary_csv,
    write_summary_json,
)


def make_step(step_index: int = 0) -> StepRecord:
    return StepRecord(
        run_id="run-1",
        task_id="task-1",
        benchmark="synthetic",
        loop_variant=LoopVariant.UNCERTAINTY_AWARE,
        model_name="test-model",
        seed=20260711,
        repetition=1,
        step_index=step_index,
        stage=Stage.VERIFICATION,
        action="run visible tests",
        decision=Decision.VERIFY_MORE,
        rationale="verification uncertainty remains moderate",
        uncertainty=UncertaintyVector(composite=0.31),
        evidence=[StepEvidence.from_text("test_output", "pytest failed")],
    )


def test_uncertainty_rejects_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="planning must be in"):
        UncertaintyVector(planning=1.2).validate()


def test_missing_ids_are_rejected() -> None:
    record = StepRecord(
        run_id="",
        task_id="task-1",
        benchmark="synthetic",
        loop_variant=LoopVariant.LINEAR,
        model_name="model",
        seed=1,
        repetition=1,
        step_index=0,
        stage=Stage.PLANNING,
        action="plan",
        decision=Decision.CONTINUE,
        rationale="continue",
        uncertainty=UncertaintyVector(),
    )
    with pytest.raises(ValueError, match="run_id is required"):
        record.validate()


def test_secret_redaction() -> None:
    fixture = "s" + "k-" + "testcredential123456789"
    evidence = StepEvidence.from_text("prompt", f"token {fixture} should vanish")
    assert "[REDACTED_SECRET]" in evidence.content
    assert fixture not in evidence.content
    assert evidence.redacted is True


def test_credential_redaction_does_not_rewrite_source_identifiers() -> None:
    source = "salted_hmac(key, payload, secret=self.secret)"
    assert redact_credentials(source) == (source, False)
    fixture = "s" + "k-" + "testcredential123456789"
    redacted, changed = redact_credentials(f"token {fixture}")
    assert changed is True
    assert fixture not in redacted


def test_jsonl_roundtrip(tmp_path) -> None:
    path = tmp_path / "runs" / "trajectory.jsonl"
    writer = JsonlTrajectoryWriter(path)
    writer.append(make_step())
    writer.append(make_step(step_index=1))
    records = read_trajectory(path)
    assert len(records) == 2
    assert records[0].schema_id == "helloloop.step.v1"
    assert records[1].step_index == 1


def test_csv_summary(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    writer = JsonlTrajectoryWriter(trajectory)
    writer.append(make_step())
    writer.append(make_step(step_index=1))
    summary = summarize_trajectory(trajectory)
    csv_path = tmp_path / "summary.csv"
    write_summary_csv([summary], csv_path)
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["steps"] == "2"
    assert rows[0]["schema_id"] == "helloloop.summary.v1"


def test_json_summary(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    writer = JsonlTrajectoryWriter(trajectory)
    writer.append(make_step())
    summary = summarize_trajectory(trajectory)
    out = tmp_path / "summary.json"
    write_summary_json(summary, out)
    assert "helloloop.summary.v1" in out.read_text(encoding="utf-8")
