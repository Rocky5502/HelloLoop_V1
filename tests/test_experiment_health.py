from __future__ import annotations

import json
from pathlib import Path

from helloloop.enums import LoopVariant
from helloloop.experiments.health import experiment_readiness
from helloloop.experiments.registry import ExperimentSpec, plan_experiment, write_registry


def test_readiness_reports_model_and_evaluator_blockers(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{"task_id": "task-1"}]}), encoding="utf-8")
    model = tmp_path / "model.yaml"
    model.write_text(
        "\n".join(
            (
                "name: fixture",
                "provider: openai_compatible",
                "model: UNRESOLVED",
                "endpoint: https://example.invalid/v1",
                "api_key_env: HELLOLOOP_TEST_KEY",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("HELLOLOOP_TEST_KEY", raising=False)
    spec = ExperimentSpec(
        phase="test",
        name="test",
        benchmark="test",
        manifest_path=str(manifest),
        loops=(LoopVariant.LINEAR,),
        model_configs=(str(model),),
    )
    registry = write_registry(plan_experiment(spec), tmp_path / "registry.csv")
    result = experiment_readiness(registry_path=registry, cost_cap_usd=1.0)
    assert not result["ready"]
    assert result["status_counts"] == {"planned": 1}
    assert any("model ID is unresolved" in blocker for blocker in result["blockers"])
    assert any("HELLOLOOP_TEST_KEY" in blocker for blocker in result["blockers"])
    assert result["observed_cost_usd"] == 0.0
