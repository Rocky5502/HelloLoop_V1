from __future__ import annotations

import math
import re

from .base import DEFAULT_FALLBACK_UNCERTAINTY, clamp01, normalized_entropy

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def semantic_uncertainty(
    candidates: list[str],
    threshold: float = 0.55,
    fallback: float = DEFAULT_FALLBACK_UNCERTAINTY,
) -> dict[str, float | int]:
    """Cluster texts using deterministic token-overlap similarity.

    This is an offline degradation path. Later phases can swap in embeddings
    while preserving the same output contract.
    """
    if not candidates:
        value = clamp01(fallback)
        return {"clusters": 0, "normalized_entropy": value, "uncertainty": value}
    clusters: list[list[set[str]]] = []
    for candidate in candidates:
        tokens = _token_set(candidate)
        placed = False
        for cluster in clusters:
            centroid_tokens = set().union(*cluster)
            if _jaccard(tokens, centroid_tokens) >= threshold:
                cluster.append(tokens)
                placed = True
                break
        if not placed:
            clusters.append([tokens])
    cluster_sizes = [len(cluster) for cluster in clusters]
    entropy = normalized_entropy(cluster_sizes)
    cluster_penalty = (
        0.0 if len(candidates) <= 1 else math.log(len(clusters)) / math.log(len(candidates))
    )
    uncertainty = (entropy + cluster_penalty) / 2.0
    return {
        "clusters": len(clusters),
        "normalized_entropy": clamp01(entropy),
        "uncertainty": clamp01(uncertainty),
    }
