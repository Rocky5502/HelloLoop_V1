from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    severity: float = 0.5

    def validate(self) -> None:
        if self.name not in PERTURBATION_NAMES:
            raise ValueError(f"unknown perturbation: {self.name}")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")


PERTURBATION_NAMES = (
    "noisy_verifier_result",
    "hidden_test_failure",
    "irrelevant_retrieval",
    "contradictory_retrieval",
    "contaminated_memory",
    "tool_timeout",
    "malformed_tool_output",
    "reduced_token_budget",
    "reduced_step_budget",
    "ambiguous_task_description",
    "model_endpoint_failure",
    "repeated_noop_patches",
)


def _rng(seed: int, key: str, name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{key}:{name}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def apply_perturbation(
    payload: dict[str, Any],
    spec: PerturbationSpec,
    *,
    seed: int = 20260711,
    key: str = "default",
) -> dict[str, Any]:
    spec.validate()
    result = dict(payload)
    randomizer = _rng(seed, key, spec.name)
    active = randomizer.random() <= spec.severity
    result["perturbation"] = spec.name
    result["perturbation_severity"] = spec.severity
    result["perturbation_active"] = active
    if not active:
        return result
    if spec.name == "noisy_verifier_result":
        result["verifier_passed"] = not bool(result.get("verifier_passed", False))
    elif spec.name == "hidden_test_failure":
        result["hidden_tests_passed"] = False
    elif spec.name == "irrelevant_retrieval":
        result["retrieval_relevance"] = 0.0
    elif spec.name == "contradictory_retrieval":
        result["retrieval_contradiction"] = True
    elif spec.name == "contaminated_memory":
        result["memory_contaminated"] = True
    elif spec.name == "tool_timeout":
        result.update({"tool_timed_out": True, "returncode": -1})
    elif spec.name == "malformed_tool_output":
        result.update({"tool_output": "<malformed>", "malformed_output": True})
    elif spec.name == "reduced_token_budget":
        result["max_tokens"] = max(128, int(result.get("max_tokens", 4096) * 0.25))
    elif spec.name == "reduced_step_budget":
        result["max_steps"] = max(2, int(result.get("max_steps", 20) * 0.5))
    elif spec.name == "ambiguous_task_description":
        result["problem_statement"] = "Resolve the reported behavior with insufficient detail."
    elif spec.name == "model_endpoint_failure":
        result["endpoint_available"] = False
    elif spec.name == "repeated_noop_patches":
        result.update({"patch_changed": False, "repeated_noop_count": 3})
    return result
