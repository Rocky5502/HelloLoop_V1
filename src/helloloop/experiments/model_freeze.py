from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from helloloop.models.openai_compatible import ModelConfig

from .health import write_experiment_readiness
from .registry import audit_registry, load_registry, write_registry


def configure_primary_model(
    *,
    model_id: str,
    input_price_per_million_usd: float,
    output_price_per_million_usd: float,
    context_limit: int,
    pricing_source: str,
    context_source: str,
    catalog_path: str | Path = "results/runs/model_smoke/catalog.json",
    config_path: str | Path = "configs/models/primary_model.yaml",
    registry_path: str | Path = "results/run_registry.csv",
    audit_path: str | Path = "results/run_audit.json",
    freeze_path: str | Path = "results/model_freeze.json",
    health_path: str | Path = "results/run_health.json",
    cost_cap_usd: float = 150.0,
) -> dict[str, Any]:
    if not model_id.strip():
        raise ValueError("exact model ID is required")
    if input_price_per_million_usd < 0 or output_price_per_million_usd < 0:
        raise ValueError("verified prices must be non-negative")
    if context_limit < 1:
        raise ValueError("context limit must be positive")
    for name, source in (("pricing", pricing_source), ("context", context_source)):
        if not source.startswith("https://"):
            raise ValueError(f"{name} source must be an HTTPS URL")

    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    if model_id not in catalog.get("models", []):
        raise ValueError("exact model ID is not present in the saved provider catalog")
    cells = load_registry(registry_path)
    started = [cell.run_id for cell in cells if cell.status != "planned"]
    if started:
        raise ValueError("model configuration is frozen after any registry cell starts")

    try:
        import yaml
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyYAML is required to configure the model") from error
    config_source = Path(config_path)
    config_data = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    config_data.update(
        {
            "model": model_id,
            "input_price_per_million_usd": input_price_per_million_usd,
            "output_price_per_million_usd": output_price_per_million_usd,
            "context_limit": context_limit,
        }
    )
    configured = ModelConfig(**config_data)
    configured.validate(require_runtime=True)

    configured_at = datetime.now(UTC).isoformat()
    config_resolved = config_source.resolve()

    def uses_primary_config(cell: Any) -> bool:
        return Path(cell.model_config).resolve() == config_resolved

    if not any(uses_primary_config(cell) for cell in cells):
        raise ValueError("no registry cells reference the selected model configuration")
    updated_cells = [
        replace(cell, model_name=model_id, updated_at=configured_at)
        if uses_primary_config(cell)
        else cell
        for cell in cells
    ]
    for cell in updated_cells:
        cell.validate()

    temporary = config_source.with_suffix(config_source.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    temporary.replace(config_source)
    write_registry(updated_cells, registry_path)
    audit = audit_registry(updated_cells)
    audit_target = Path(audit_path)
    audit_target.parent.mkdir(parents=True, exist_ok=True)
    audit_target.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    freeze = {
        "schema_id": "helloloop.model_freeze.v1",
        "configured_at": configured_at,
        "model_id": model_id,
        "catalog_path": str(catalog_path),
        "catalog_retrieved_at": catalog.get("retrieved_at"),
        "endpoint": configured.endpoint,
        "input_price_per_million_usd": input_price_per_million_usd,
        "output_price_per_million_usd": output_price_per_million_usd,
        "pricing_source": pricing_source,
        "context_limit": context_limit,
        "context_source": context_source,
        "prompt_template_version": configured.prompt_template_version,
        "seed": configured.seed,
        "temperature": configured.temperature,
        "top_p": configured.top_p,
        "max_tokens": configured.max_tokens,
        "registry_cells_updated": sum(uses_primary_config(cell) for cell in cells),
        "registry_audit": audit,
    }
    freeze_target = Path(freeze_path)
    freeze_target.parent.mkdir(parents=True, exist_ok=True)
    freeze_target.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze["readiness"] = write_experiment_readiness(
        health_path,
        registry_path=registry_path,
        cost_cap_usd=cost_cap_usd,
    )
    return freeze
