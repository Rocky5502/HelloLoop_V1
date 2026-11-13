from __future__ import annotations

from .base import clamp01


def progress_uncertainty(
    repeated_actions: int = 0,
    repeated_file_reads: int = 0,
    repeated_patches: int = 0,
    no_test_improvement_steps: int = 0,
    no_new_evidence_steps: int = 0,
    cycle_detected: bool = False,
) -> float:
    signals = [
        min(1.0, repeated_actions / 4.0),
        min(1.0, repeated_file_reads / 6.0),
        min(1.0, repeated_patches / 3.0),
        min(1.0, no_test_improvement_steps / 4.0),
        min(1.0, no_new_evidence_steps / 4.0),
        1.0 if cycle_detected else 0.0,
    ]
    return clamp01(sum(signals) / len(signals))
