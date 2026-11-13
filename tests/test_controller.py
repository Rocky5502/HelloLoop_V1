from __future__ import annotations

from helloloop.controller import ControllerState, Thresholds, decide
from helloloop.enums import Decision, LoopVariant, Stage
from helloloop.schema import UncertaintyVector


def state(
    loop: LoopVariant = LoopVariant.UNCERTAINTY_AWARE,
    uncertainty: UncertaintyVector | None = None,
    **kwargs,
) -> ControllerState:
    return ControllerState(
        loop_variant=loop,
        stage=kwargs.pop("stage", Stage.PLANNING),
        uncertainty=uncertainty or UncertaintyVector(composite=0.5),
        step_index=kwargs.pop("step_index", 0),
        max_steps=kwargs.pop("max_steps", 10),
        **kwargs,
    )


def test_hard_budget_wins() -> None:
    result = decide(state(cost_exhausted=True))
    assert result.decision == Decision.STOP_BUDGET


def test_linear_ignores_uncertainty() -> None:
    uncertainty = UncertaintyVector(retrieval=1.0, planning=1.0, composite=1.0)
    result = decide(state(loop=LoopVariant.LINEAR, uncertainty=uncertainty))
    assert result.decision == Decision.EXECUTE


def test_reflection_uses_fixed_replan() -> None:
    result = decide(state(loop=LoopVariant.REFLECTION, step_index=1, reflection_used=False))
    assert result.decision == Decision.REPLAN


def test_fixed_verification_uses_schedule() -> None:
    result = decide(state(loop=LoopVariant.VERIFICATION, verification_attempts=0))
    assert result.decision == Decision.VERIFY_MORE


def test_l3_retrieval_priority_over_planning() -> None:
    uncertainty = UncertaintyVector(task=0.1, retrieval=0.9, planning=0.9, composite=0.8)
    result = decide(state(uncertainty=uncertainty))
    assert result.decision == Decision.RETRIEVE
    assert result.request_more_context is True


def test_l3_task_uncertainty_wins_before_retrieval() -> None:
    uncertainty = UncertaintyVector(task=0.9, retrieval=0.9, planning=0.9, composite=0.8)
    result = decide(state(uncertainty=uncertainty))
    assert result.decision == Decision.REPLAN


def test_l3_cannot_stop_success_without_static_and_evidence() -> None:
    uncertainty = UncertaintyVector(termination=0.1, composite=0.1)
    result = decide(
        state(
            uncertainty=uncertainty,
            visible_tests_passed=True,
            static_checks_passed=False,
            evidence_count=1,
        )
    )
    assert result.decision == Decision.VERIFY_MORE


def test_l3_stop_success_requires_evidence_and_low_uncertainty() -> None:
    uncertainty = UncertaintyVector(termination=0.1, composite=0.1)
    result = decide(
        state(
            uncertainty=uncertainty,
            visible_tests_passed=True,
            static_checks_passed=True,
            evidence_count=2,
        )
    )
    assert result.decision == Decision.STOP_SUCCESS


def test_l3_high_verification_requests_independent_verifier() -> None:
    uncertainty = UncertaintyVector(verification=0.8, composite=0.6)
    result = decide(state(uncertainty=uncertainty))
    assert result.decision == Decision.VERIFY_MORE
    assert result.request_independent_verifier is True


def test_l3_escalates_on_stagnation() -> None:
    uncertainty = UncertaintyVector(progress=0.8, composite=0.8)
    result = decide(state(uncertainty=uncertainty, stagnation_steps=3))
    assert result.decision == Decision.ESCALATE


def test_l3_abstains_when_escalation_disabled() -> None:
    uncertainty = UncertaintyVector(progress=0.8, composite=0.8)
    result = decide(
        state(
            uncertainty=uncertainty,
            stagnation_steps=3,
            allow_escalation=False,
            allow_abstention=True,
        )
    )
    assert result.decision == Decision.ABSTAIN


def test_controller_deterministic() -> None:
    s = state(uncertainty=UncertaintyVector(retrieval=0.9, composite=0.8))
    assert decide(s, Thresholds()) == decide(s, Thresholds())
