from __future__ import annotations

import json
from pathlib import Path

from helloloop.experiments.recovery import recover_generated_submissions
from helloloop.experiments.registry import RunCell, load_registry, write_registry


def test_export_only_failure_recovers_original_submission_without_model_call(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_id = "swebench_lite_main__org__repo-1__linear__primary_model__r0"
    run_dir = Path("results/runs/swebench_lite_main") / run_id
    reason = "ValueError: prediction for org__repo-1 contains secret-like material"
    cell = RunCell(
        phase="phase13",
        experiment="swebench_lite_main",
        order=0,
        run_id=run_id,
        benchmark="swebench_lite_test",
        manifest_path="manifest.json",
        task_id="org__repo-1",
        loop_variant="linear",
        model_config="model.yaml",
        model_name="gpt-5.4-2026-03-05",
        repetition_id=0,
        seed=20260711,
        status="failed",
        reason=reason,
        failure_type="infrastructure",
        artifact_path=str(run_dir),
        official_evaluation_status="not_applicable",
        terminal_classification="infrastructure_failure",
    )
    registry = Path("results/run_registry.csv")
    write_registry([cell], registry)
    patch = "+    digest = salted_hmac(key, payload, secret=self.secret)\n"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"status": "complete", "estimated_cost_usd": 0.25}),
        encoding="utf-8",
    )
    (run_dir / "upstream_trajectory.json").write_text(
        json.dumps(
            {
                "info": {"exit_status": "Submitted", "submission": patch},
                "messages": [
                    {
                        "extra": {
                            "response": {"model": "gpt-5.4-2026-03-05"},
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = recover_generated_submissions(
        "main",
        registry_path=registry,
        audit_path="results/run_audit.json",
    )

    recovered = load_registry(registry)[0]
    event = json.loads((run_dir / "status_event.json").read_text(encoding="utf-8"))
    run_result = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))
    assert result["model_calls"] == 0
    assert result["recovered_run_ids"] == [run_id]
    assert recovered.status == "complete"
    assert recovered.official_evaluation_status == "pending"
    assert (run_dir / "patch.diff").read_text(encoding="utf-8") == patch
    assert "secret=self.secret" in (run_dir / "prediction.jsonl").read_text(encoding="utf-8")
    assert event["recovery"]["prior_reason"] == reason
    assert run_result["recovered_without_model_call"] is True
