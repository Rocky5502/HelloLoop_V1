from __future__ import annotations

from collections.abc import Mapping

from .base import DEFAULT_FALLBACK_UNCERTAINTY, clamp01, normalized_entropy


def verifier_uncertainty(
    verifier_results: Mapping[str, bool | None],
    fallback: float = DEFAULT_FALLBACK_UNCERTAINTY,
) -> float:
    if not verifier_results:
        return clamp01(fallback)
    known = [value for value in verifier_results.values() if value is not None]
    unknown_count = sum(value is None for value in verifier_results.values())
    if not known:
        return clamp01(fallback)
    pass_count = sum(bool(value) for value in known)
    fail_count = len(known) - pass_count
    disagreement = normalized_entropy([pass_count, fail_count])
    unknown_penalty = unknown_count / len(verifier_results)
    return clamp01(0.75 * disagreement + 0.25 * unknown_penalty)
