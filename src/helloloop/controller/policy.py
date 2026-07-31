from __future__ import annotations

from dataclasses import dataclass

from helloloop.enums import Decision, LoopVariant, Stage
from helloloop.schema import UncertaintyVector

from .thresholds import Thresholds


@dataclass(frozen=True)
class ControllerState:
    loop_variant: LoopVariant
    stage: Stage
    uncertainty: UncertaintyVector
    step_index: int
    max_steps: int
    visible_tests_passed: bool = False
    static_checks_passed: bool = False
    evidence_count: int = 0
    reflection_used: bool = False
    verification_attempts: int = 0
    max_verification_attempts: int = 2
    stagnation_steps: int = 0
    cost_exhausted: bool = False
    allow_escalation: bool = True
    allow_abstention: bool = True


@dataclass(frozen=True)
class ControllerDecision:
    decision: Decision
    rationale: str
    request_more_context: bool = False
    request_independent_verifier: bool = False


def decide(state: ControllerState, thresholds: Thresholds | None = None) -> ControllerDecision:
    thresholds = thresholds or Thresholds()
    if state.cost_exhausted or state.step_index >= state.max_steps:
        return ControllerDecision(Decision.STOP_BUDGET, "hard budget exhausted")

    if state.loop_variant == LoopVariant.LINEAR:
        return _decide_linear(state)
    if state.loop_variant == LoopVariant.REFLECTION:
        return _decide_reflection(state)
    if state.loop_variant == LoopVariant.VERIFICATION:
        return _decide_verification(state, thresholds)
    if state.loop_variant == LoopVariant.UNCERTAINTY_AWARE:
        return _decide_uncertainty_aware(state, thresholds)
    raise ValueError(f"unsupported loop variant: {state.loop_variant}")


def _success_evidence_sufficient(state: ControllerState, thresholds: Thresholds) -> bool:
    return (
        state.visible_tests_passed
        and state.static_checks_passed
        and state.evidence_count > 0
        and state.uncertainty.termination <= thresholds.termination_low
        and state.uncertainty.composite <= thresholds.termination_low
    )


def _decide_linear(state: ControllerState) -> ControllerDecision:
    if state.visible_tests_passed:
        return ControllerDecision(Decision.STOP_SUCCESS, "linear stop after visible success")
    return ControllerDecision(Decision.EXECUTE, "linear loop continues until success or budget")


def _decide_reflection(state: ControllerState) -> ControllerDecision:
    if state.visible_tests_passed:
        return ControllerDecision(
            Decision.STOP_SUCCESS, "reflection loop stops after visible success"
        )
    if not state.reflection_used and state.step_index > 0:
        return ControllerDecision(Decision.REPLAN, "fixed reflection retry after failure")
    return ControllerDecision(Decision.EXECUTE, "reflection loop executes fixed retry policy")


def _decide_verification(state: ControllerState, thresholds: Thresholds) -> ControllerDecision:
    if state.visible_tests_passed and state.static_checks_passed:
        return ControllerDecision(Decision.STOP_SUCCESS, "fixed verification passed")
    if state.verification_attempts < state.max_verification_attempts:
        return ControllerDecision(Decision.VERIFY_MORE, "fixed verification schedule")
    if state.uncertainty.execution >= thresholds.execution_high:
        return ControllerDecision(Decision.REVISE, "fixed verification found execution risk")
    return ControllerDecision(Decision.EXECUTE, "fixed verification continues after failed checks")


def _decide_uncertainty_aware(
    state: ControllerState,
    thresholds: Thresholds,
) -> ControllerDecision:
    if _success_evidence_sufficient(state, thresholds):
        return ControllerDecision(
            Decision.STOP_SUCCESS, "evidence sufficient and residual uncertainty low"
        )
    if state.visible_tests_passed and not _success_evidence_sufficient(state, thresholds):
        return ControllerDecision(
            Decision.VERIFY_MORE, "visible success lacks sufficient evidence for stop"
        )

    if state.uncertainty.task >= thresholds.task_high:
        return ControllerDecision(
            Decision.REPLAN, "high task uncertainty; extract constraints before acting"
        )
    if state.uncertainty.retrieval >= thresholds.retrieval_high:
        return ControllerDecision(
            Decision.RETRIEVE,
            "high retrieval uncertainty; gather more context",
            request_more_context=True,
        )
    if state.uncertainty.planning >= thresholds.planning_high:
        return ControllerDecision(Decision.REPLAN, "high planning uncertainty")
    if (
        state.uncertainty.progress >= thresholds.progress_high
        or state.stagnation_steps >= thresholds.stagnation_steps
    ):
        if state.allow_escalation:
            return ControllerDecision(Decision.ESCALATE, "persistent uncertainty plus stagnation")
        if state.allow_abstention:
            return ControllerDecision(
                Decision.ABSTAIN, "persistent unresolved uncertainty without escalation path"
            )
    if state.stage == Stage.EXECUTION:
        return ControllerDecision(
            Decision.VERIFY_MORE, "execution completed; verify before continuing"
        )
    if state.uncertainty.execution > thresholds.execution_high:
        return ControllerDecision(Decision.REVISE, "high execution uncertainty")
    if state.uncertainty.verification >= thresholds.verification_high:
        return ControllerDecision(
            Decision.VERIFY_MORE,
            "high verification uncertainty; request independent verification",
            request_independent_verifier=True,
        )
    if (
        state.uncertainty.verification >= thresholds.verification_moderate
        and state.verification_attempts < state.max_verification_attempts
    ):
        return ControllerDecision(Decision.VERIFY_MORE, "moderate verification uncertainty")
    if state.uncertainty.composite > thresholds.termination_low:
        return ControllerDecision(
            Decision.EXECUTE, "continued action justified by residual uncertainty"
        )
    return ControllerDecision(
        Decision.STOP_UNRESOLVED, "continued action unjustified without evidence"
    )
