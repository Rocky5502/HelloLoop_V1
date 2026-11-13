from __future__ import annotations

import math
from collections.abc import Mapping

from .base import DEFAULT_FALLBACK_UNCERTAINTY, clamp01


def composite_uncertainty(
    components: Mapping[str, float | None],
    weights: Mapping[str, float] | None = None,
    mode: str = "weighted_average",
    fallback: float = DEFAULT_FALLBACK_UNCERTAINTY,
) -> float:
    active = {
        key: clamp01(value if value is not None else fallback) for key, value in components.items()
    }
    if not active:
        return clamp01(fallback)
    if mode == "maximum":
        return max(active.values())
    if mode == "logistic":
        weighted = _weighted_average(active, weights)
        return clamp01(1.0 / (1.0 + math.exp(-8.0 * (weighted - 0.5))))
    if mode != "weighted_average":
        raise ValueError(f"unknown composite mode: {mode}")
    return _weighted_average(active, weights)


def _weighted_average(
    components: Mapping[str, float], weights: Mapping[str, float] | None
) -> float:
    if not weights:
        return clamp01(sum(components.values()) / len(components))
    numerator = 0.0
    denominator = 0.0
    for key, value in components.items():
        weight = max(0.0, float(weights.get(key, 0.0)))
        numerator += value * weight
        denominator += weight
    if denominator == 0:
        return clamp01(sum(components.values()) / len(components))
    return clamp01(numerator / denominator)
