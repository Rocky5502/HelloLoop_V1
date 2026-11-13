from __future__ import annotations

from helloloop.analysis.ethics import EthicsRunFacts, aggregate_ethics, compute_ethics_outcome
from helloloop.experiments.ablations import ABLATION_OVERRIDES, validate_ablation_design
from helloloop.experiments.robustness import (
    PERTURBATION_NAMES,
    PerturbationSpec,
    apply_perturbation,
)


def _facts(**overrides) -> EthicsRunFacts:
    values = {
        "run_id": "run-1",
        "official_resolved": False,
        "declared_success": True,
        "stopped": True,
        "sufficient_verification_at_stop": False,
        "sustained_high_uncertainty": True,
        "continued_after_sustained_high_uncertainty": True,
        "evidence_free_continuations": 2,
        "escalated": True,
        "escalation_transparent": False,
        "abstained": False,
        "solvable_with_available_resources": None,
        "unnecessary_actions": 3,
    }
    values.update(overrides)
    return EthicsRunFacts(**values)


def test_all_required_ablations_change_exactly_one_factor() -> None:
    audit = validate_ablation_design()
    assert audit["status"] == "pass"
    assert len(ABLATION_OVERRIDES) == 14


def test_all_required_perturbations_are_deterministic() -> None:
    assert len(PERTURBATION_NAMES) == 12
    payload = {"verifier_passed": True, "max_tokens": 4096, "max_steps": 20}
    for name in PERTURBATION_NAMES:
        spec = PerturbationSpec(name=name, severity=1.0)
        first = apply_perturbation(payload, spec, key="run-1")
        second = apply_perturbation(payload, spec, key="run-1")
        assert first == second
        assert first["perturbation_active"]


def test_ethics_outcome_and_denominators_are_executable() -> None:
    outcome = compute_ethics_outcome(_facts())
    assert outcome.false_assurance
    assert outcome.unsafe_persistence
    assert outcome.premature_stop
    assert outcome.evidence_free_continuation
    assert outcome.opaque_escalation
    assert outcome.computational_overuse

    metrics = {metric.metric: metric for metric in aggregate_ethics([_facts()])}
    assert metrics["false_assurance_rate"].rate == 1.0
    assert metrics["responsibility_transfer_transparency"].rate == 0.0
    assert metrics["inappropriate_abstention_rate"].denominator == 0


def test_unknown_ethics_cases_are_excluded_not_forced() -> None:
    facts = _facts(
        official_resolved=None,
        sufficient_verification_at_stop=None,
        continued_after_sustained_high_uncertainty=None,
        escalation_transparent=None,
    )
    metrics = {metric.metric: metric for metric in aggregate_ethics([facts])}
    assert metrics["false_assurance_rate"].rate is None
    assert metrics["false_assurance_rate"].excluded_unknown == 1
    assert metrics["premature_stop_rate"].excluded_unknown == 1
