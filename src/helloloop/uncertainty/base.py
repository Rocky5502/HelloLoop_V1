from __future__ import annotations

import math
from collections.abc import Iterable

DEFAULT_FALLBACK_UNCERTAINTY = 0.5


def clamp01(value: float) -> float:
    if math.isnan(value):
        return DEFAULT_FALLBACK_UNCERTAINTY
    return max(0.0, min(1.0, float(value)))


def normalized_entropy(counts: Iterable[int]) -> float:
    values = [count for count in counts if count > 0]
    if len(values) <= 1:
        return 0.0
    total = sum(values)
    entropy = -sum((count / total) * math.log(count / total) for count in values)
    return clamp01(entropy / math.log(len(values)))


def safe_ratio(
    numerator: float, denominator: float, fallback: float = DEFAULT_FALLBACK_UNCERTAINTY
) -> float:
    if denominator == 0:
        return fallback
    return clamp01(numerator / denominator)
