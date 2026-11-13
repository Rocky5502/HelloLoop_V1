from __future__ import annotations

from .base import DEFAULT_FALLBACK_UNCERTAINTY, clamp01


def verbal_uncertainty(
    confidence: float | None, fallback: float = DEFAULT_FALLBACK_UNCERTAINTY
) -> float:
    """Convert strict machine-readable confidence to uncertainty."""
    if confidence is None:
        return clamp01(fallback)
    return clamp01(1.0 - float(confidence))
