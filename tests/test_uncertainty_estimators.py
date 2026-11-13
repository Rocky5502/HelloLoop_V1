from __future__ import annotations

import pytest

from helloloop.uncertainty import (
    composite_uncertainty,
    disagreement_uncertainty,
    execution_uncertainty,
    memory_uncertainty,
    progress_uncertainty,
    semantic_uncertainty,
    verbal_uncertainty,
    verifier_uncertainty,
)


def assert_unit_interval(value: float) -> None:
    assert 0.0 <= value <= 1.0


def test_verbal_uncertainty() -> None:
    assert verbal_uncertainty(0.8) == pytest.approx(0.2)
    assert verbal_uncertainty(None) == pytest.approx(0.5)


def test_disagreement_uncertainty_toy_example() -> None:
    result = disagreement_uncertainty(["plan_a", "plan_b", "plan_a"])
    assert result["dominant"] == "plan_a"
    assert result["dominant_vote_disagreement"] == pytest.approx(1 / 3)
    assert_unit_interval(float(result["uncertainty"]))


def test_disagreement_empty_fallback() -> None:
    result = disagreement_uncertainty([])
    assert result["uncertainty"] == pytest.approx(0.5)


def test_semantic_uncertainty_degrades_without_embeddings() -> None:
    similar = semantic_uncertainty(["fix parser bug", "fix parser bug"])
    diverse = semantic_uncertainty(["fix parser bug", "update billing docs", "change auth policy"])
    assert_unit_interval(float(similar["uncertainty"]))
    assert float(diverse["uncertainty"]) >= float(similar["uncertainty"])


def test_execution_uncertainty_increases_with_failures() -> None:
    clean = execution_uncertainty(exit_code=0, tests_passed=10, tests_failed=0)
    failing = execution_uncertainty(exit_code=1, tests_passed=8, tests_failed=4)
    assert failing > clean
    assert_unit_interval(failing)


def test_verifier_uncertainty_handles_disagreement_and_unknowns() -> None:
    agree = verifier_uncertainty({"static": True, "targeted": True})
    disagree = verifier_uncertainty({"static": True, "targeted": False, "judge": None})
    assert disagree > agree
    assert_unit_interval(disagree)


def test_progress_uncertainty_increases_with_cycles() -> None:
    low = progress_uncertainty()
    high = progress_uncertainty(
        repeated_actions=4,
        repeated_file_reads=6,
        repeated_patches=3,
        no_test_improvement_steps=4,
        no_new_evidence_steps=4,
        cycle_detected=True,
    )
    assert high > low
    assert high == pytest.approx(1.0)


def test_memory_uncertainty_known_good_vs_contradictory() -> None:
    good = memory_uncertainty(retrieval_relevance=0.9, provenance_known=True)
    bad = memory_uncertainty(
        retrieval_relevance=0.1,
        provenance_known=False,
        contradiction=True,
        staleness=0.9,
        unsupported_prior_claims=3,
    )
    assert bad > good


def test_composite_modes_and_weights() -> None:
    components = {"planning": 0.2, "verification": 0.8}
    weighted = composite_uncertainty(components, weights={"planning": 1, "verification": 3})
    maximum = composite_uncertainty(components, mode="maximum")
    logistic = composite_uncertainty(components, mode="logistic")
    assert weighted == pytest.approx(0.65)
    assert maximum == pytest.approx(0.8)
    assert_unit_interval(logistic)


def test_composite_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError, match="unknown composite"):
        composite_uncertainty({"x": 0.5}, mode="mystery")
