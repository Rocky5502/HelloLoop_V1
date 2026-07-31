from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    task_high: float = 0.70
    planning_high: float = 0.60
    retrieval_high: float = 0.60
    execution_high: float = 0.50
    verification_moderate: float = 0.40
    verification_high: float = 0.55
    progress_high: float = 0.60
    termination_low: float = 0.20
    stagnation_steps: int = 3
