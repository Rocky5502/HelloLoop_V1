from __future__ import annotations

import json
from pathlib import Path

import pytest

from helloloop.experiments.model_freeze import configure_primary_model
from helloloop.experiments.registry import RunCell, load_registry, write_registry


def _cell(config: Path, *, status: str = "planned") -> RunCell:
    kwargs = {}
    if status == "complete":
        kwargs = {
            "terminal_classification": "submitted_patch",
        }
    return RunCell(
        phase="phase12",
        experiment="swebench_lite_dev_pilot",
        order=0,
        run_id="run-a",
        benchmark="swebench_lite_dev",
        manifest_path="manifest.json",
        task_id="task-a",
        loop_variant="linear",
        model_config=str(config),
        model_name="UNRESOLVED",
        repetition_id=0,
        seed=20260711,
        status=status,
        artifact_path="results/runs/run-a",
        **kwargs,
    )


def _config(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "name: primary_model",
                "provider: openai_compatible",
                "model: UNRESOLVED",
                "endpoint: https://example.invalid/v1",
                "api_key_env: HELLOLOOP_TEST_KEY",
                "input_price_per_million_usd: null",
                "output_price_per_million_usd: null",
                "context_limit: null",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_model_freeze_requires_catalog_and_updates_planned_registry(tmp_path: Path) -> None:
    config = tmp_path / "primary.yaml"
    _config(config)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps({"models": ["coding-model"], "retrieved_at": "2026-07-19T00:00:00Z"}),
        encoding="utf-8",
    )
    registry = write_registry([_cell(config)], tmp_path / "registry.csv")

    result = configure_primary_model(
        model_id="coding-model",
        input_price_per_million_usd=1.0,
        output_price_per_million_usd=2.0,
        context_limit=128_000,
        pricing_source="https://example.invalid/pricing",
        context_source="https://example.invalid/models",
        catalog_path=catalog,
        config_path=config,
        registry_path=registry,
        audit_path=tmp_path / "audit.json",
        freeze_path=tmp_path / "freeze.json",
        health_path=tmp_path / "health.json",
    )

    assert result["model_id"] == "coding-model"
    assert result["registry_cells_updated"] == 1
    assert result["temperature"] == 0.0
    assert result["top_p"] == 1.0
    assert result["max_tokens"] == 4096
    assert load_registry(registry)[0].model_name == "coding-model"
    assert "model: coding-model" in config.read_text(encoding="utf-8")


def test_model_freeze_refuses_unknown_model_and_started_registry(tmp_path: Path) -> None:
    config = tmp_path / "primary.yaml"
    _config(config)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"models": ["coding-model"]}), encoding="utf-8")
    registry = write_registry([_cell(config)], tmp_path / "registry.csv")
    common = {
        "input_price_per_million_usd": 1.0,
        "output_price_per_million_usd": 2.0,
        "context_limit": 128_000,
        "pricing_source": "https://example.invalid/pricing",
        "context_source": "https://example.invalid/models",
        "catalog_path": catalog,
        "config_path": config,
        "registry_path": registry,
        "audit_path": tmp_path / "audit.json",
        "freeze_path": tmp_path / "freeze.json",
        "health_path": tmp_path / "health.json",
    }
    with pytest.raises(ValueError, match="not present"):
        configure_primary_model(model_id="unknown", **common)

    started = _cell(config, status="complete")
    write_registry([started], registry)
    with pytest.raises(ValueError, match="frozen"):
        configure_primary_model(model_id="coding-model", **common)
