from __future__ import annotations

from collections import Counter

from .base import DEFAULT_FALLBACK_UNCERTAINTY, clamp01, normalized_entropy


def disagreement_uncertainty(
    candidates: list[str],
    fallback: float = DEFAULT_FALLBACK_UNCERTAINTY,
) -> dict[str, float | str | None]:
    if not candidates:
        return {
            "dominant": None,
            "dominant_vote_disagreement": clamp01(fallback),
            "normalized_entropy": clamp01(fallback),
            "pairwise_disagreement": clamp01(fallback),
            "uncertainty": clamp01(fallback),
        }

    counts = Counter(candidates)
    dominant, dominant_count = counts.most_common(1)[0]
    n = len(candidates)
    dominant_vote = 1.0 - dominant_count / n
    entropy = normalized_entropy(counts.values())
    if n <= 1:
        pairwise = 0.0
    else:
        agreeing_pairs = sum(count * (count - 1) / 2 for count in counts.values())
        total_pairs = n * (n - 1) / 2
        pairwise = 1.0 - agreeing_pairs / total_pairs
    uncertainty = (dominant_vote + entropy + pairwise) / 3.0
    return {
        "dominant": dominant,
        "dominant_vote_disagreement": clamp01(dominant_vote),
        "normalized_entropy": clamp01(entropy),
        "pairwise_disagreement": clamp01(pairwise),
        "uncertainty": clamp01(uncertainty),
    }
