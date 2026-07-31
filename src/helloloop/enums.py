from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    TASK_INTERPRETATION = "task_interpretation"
    PLANNING = "planning"
    RETRIEVAL = "retrieval"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    MEMORY = "memory"
    CONTROL = "control"
    TERMINATION = "termination"


class Decision(StrEnum):
    CONTINUE = "continue"
    RETRIEVE = "retrieve"
    REPLAN = "replan"
    EXECUTE = "execute"
    VERIFY_MORE = "verify_more"
    REVISE = "revise"
    DELEGATE = "delegate"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"
    STOP_SUCCESS = "stop_success"
    STOP_UNRESOLVED = "stop_unresolved"
    STOP_BUDGET = "stop_budget"


class LoopVariant(StrEnum):
    LINEAR = "linear"
    REFLECTION = "reflection"
    VERIFICATION = "verification"
    UNCERTAINTY_AWARE = "uncertainty_aware"
