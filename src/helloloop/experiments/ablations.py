from __future__ import annotations

from copy import deepcopy
from typing import Any

BASE_ABLATION_CONFIG: dict[str, Any] = {
    "verbal_confidence": True,
    "self_consistency_disagreement": True,
    "semantic_disagreement": True,
    "execution_evidence": True,
    "verifier_disagreement": True,
    "progress_uncertainty": True,
    "memory_uncertainty": True,
    "escalation": True,
    "abstention": True,
    "cost_aware_control": True,
    "threshold_mode": "calibrated",
    "verification_sources": "visible_plus_independent",
    "independent_judge": True,
    "composite_mode": "weighted_average",
}

ABLATION_OVERRIDES: dict[str, dict[str, Any]] = {
    "no_verbal_confidence": {"verbal_confidence": False},
    "no_self_consistency_disagreement": {"self_consistency_disagreement": False},
    "no_semantic_disagreement": {"semantic_disagreement": False},
    "no_execution_evidence": {"execution_evidence": False},
    "no_verifier_disagreement": {"verifier_disagreement": False},
    "no_progress_uncertainty": {"progress_uncertainty": False},
    "no_memory_uncertainty": {"memory_uncertainty": False},
    "no_escalation": {"escalation": False},
    "no_abstention": {"abstention": False},
    "no_cost_aware_control": {"cost_aware_control": False},
    "fixed_thresholds": {"threshold_mode": "fixed"},
    "visible_tests_only": {"verification_sources": "visible_only"},
    "no_independent_judge": {"independent_judge": False},
    "maximum_composite": {"composite_mode": "maximum"},
}


def ablation_configs() -> dict[str, dict[str, Any]]:
    configs = {}
    for name, override in ABLATION_OVERRIDES.items():
        config = deepcopy(BASE_ABLATION_CONFIG)
        config.update(override)
        configs[name] = config
    return configs


def validate_ablation_design() -> dict[str, Any]:
    violations = []
    configs = ablation_configs()
    for name, config in configs.items():
        changed = [key for key, value in config.items() if BASE_ABLATION_CONFIG[key] != value]
        if len(changed) != 1:
            violations.append(f"{name} changes {len(changed)} factors: {changed}")
    return {
        "status": "pass" if not violations else "fail",
        "base": BASE_ABLATION_CONFIG,
        "ablations": configs,
        "violations": violations,
    }
