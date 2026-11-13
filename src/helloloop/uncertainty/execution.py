from __future__ import annotations

from .base import clamp01, safe_ratio


def execution_uncertainty(
    exit_code: int | None = None,
    timeout: bool = False,
    tests_passed: int | None = None,
    tests_failed: int | None = None,
    no_op_edit: bool = False,
    repeated_command_failures: int = 0,
    malformed_output: bool = False,
    partial_completion: bool = False,
) -> float:
    signals: list[float] = []
    if exit_code is not None:
        signals.append(0.0 if exit_code == 0 else 0.65)
    if timeout:
        signals.append(1.0)
    if tests_passed is not None or tests_failed is not None:
        passed = tests_passed or 0
        failed = tests_failed or 0
        signals.append(safe_ratio(failed, passed + failed, fallback=0.5))
    if no_op_edit:
        signals.append(0.75)
    if repeated_command_failures:
        signals.append(min(1.0, repeated_command_failures / 3.0))
    if malformed_output:
        signals.append(0.70)
    if partial_completion:
        signals.append(0.55)
    if not signals:
        return 0.5
    return clamp01(sum(signals) / len(signals))
