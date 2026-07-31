from __future__ import annotations

from .base import DEFAULT_FALLBACK_UNCERTAINTY, clamp01


def memory_uncertainty(
    retrieval_relevance: float | None = None,
    provenance_known: bool = True,
    contradiction: bool = False,
    staleness: float = 0.0,
    unsupported_prior_claims: int = 0,
    fallback: float = DEFAULT_FALLBACK_UNCERTAINTY,
) -> float:
    signals: list[float] = []
    if retrieval_relevance is None:
        signals.append(clamp01(fallback))
    else:
        signals.append(1.0 - clamp01(retrieval_relevance))
    signals.append(0.0 if provenance_known else 0.65)
    signals.append(1.0 if contradiction else 0.0)
    signals.append(clamp01(staleness))
    signals.append(min(1.0, unsupported_prior_claims / 3.0))
    return clamp01(sum(signals) / len(signals))
