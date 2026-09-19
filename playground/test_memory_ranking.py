import pytest

from flycoder.tools.memory import (
    Experience,
    MemoryStore,
    RankedMemory,
    calculate_outcome_score,
    calculate_ranking_score,
    rank_experiences,
)


def test_successful_verified_experience_has_full_outcome_score():
    experience = Experience(
        task="Fix authentication",
        action="Update middleware",
        outcome="Passed",
        success=True,
        verified=True,
    )

    assert calculate_outcome_score(experience) == 1.0


def test_successful_unverified_experience_has_partial_outcome_score():
    experience = Experience(
        task="Fix authentication",
        action="Update middleware",
        outcome="Passed",
        success=True,
        verified=False,
    )

    assert calculate_outcome_score(experience) == 0.5


def test_failed_experience_has_zero_outcome_score():
    experience = Experience(
        task="Fix authentication",
        action="Update middleware",
        outcome="Failed",
        success=False,
        verified=False,
    )

    assert calculate_outcome_score(experience) == 0.0


def test_ranking_score_uses_similarity_and_outcome():
    score = calculate_ranking_score(
        similarity=0.8,
        outcome=1.0,
    )

    assert score == pytest.approx(0.86)


def test_ranking_score_with_failed_outcome():
    score = calculate_ranking_score(
        similarity=0.8,
        outcome=0.0,
    )

    assert score == pytest.approx(0.56)


def test_rank_experiences_returns_ranked_memory():
    store = MemoryStore()

    experience = Experience(
        task="Fix authentication middleware",
        action="Update middleware",
        outcome="Passed",
        success=True,
        verified=True,
    )

    store.record_experience(experience)

    results = rank_experiences(
        store,
        "Fix authentication middleware",
    )

    assert len(results) == 1
    assert isinstance(results[0], RankedMemory)
    assert results[0].experience is experience
    assert results[0].similarity_score == 1.0
    assert results[0].outcome_score == 1.0
    assert results[0].ranking_score == 1.0


def test_verified_success_can_rank_above_unverified_success():
    store = MemoryStore()

    unverified = Experience(
        task="Fix authentication middleware",
        action="First approach",
        outcome="Passed",
        success=True,
        verified=False,
    )

    verified = Experience(
        task="Fix authentication middleware",
        action="Second approach",
        outcome="Passed",
        success=True,
        verified=True,
    )

    store.record_experience(unverified)
    store.record_experience(verified)

    results = rank_experiences(
        store,
        "Fix authentication middleware",
    )

    assert len(results) == 2
    assert results[0].experience is verified
    assert results[1].experience is unverified
    assert results[0].ranking_score > results[1].ranking_score


def test_failed_memory_is_retained_but_ranked_lower():
    store = MemoryStore()

    failed = Experience(
        task="Fix authentication middleware",
        action="Failed approach",
        outcome="Tests failed",
        success=False,
        verified=False,
    )

    successful = Experience(
        task="Fix authentication middleware",
        action="Successful approach",
        outcome="Tests passed",
        success=True,
        verified=True,
    )

    store.record_experience(failed)
    store.record_experience(successful)

    results = rank_experiences(
        store,
        "Fix authentication middleware",
    )

    assert len(results) == 2
    assert results[0].experience is successful
    assert results[1].experience is failed
    assert results[0].ranking_score > results[1].ranking_score


def test_similarity_remains_dominant_in_ranking():
    store = MemoryStore()

    highly_similar_unverified = Experience(
        task="Fix authentication middleware",
        action="Unverified approach",
        outcome="Passed",
        success=True,
        verified=False,
    )

    less_similar_verified = Experience(
        task="Fix authentication",
        action="Verified approach",
        outcome="Passed",
        success=True,
        verified=True,
    )

    store.record_experience(highly_similar_unverified)
    store.record_experience(less_similar_verified)

    results = rank_experiences(
        store,
        "Fix authentication middleware",
    )

    assert len(results) == 2
    assert results[0].experience is highly_similar_unverified


def test_minimum_similarity_score_is_respected():
    store = MemoryStore()

    weak = Experience(
        task="Fix authentication",
        action="Weak match",
        outcome="Passed",
        success=True,
        verified=True,
    )

    strong = Experience(
        task="Fix authentication middleware",
        action="Strong match",
        outcome="Passed",
        success=True,
        verified=True,
    )

    store.record_experience(weak)
    store.record_experience(strong)

    results = rank_experiences(
        store,
        "Fix authentication middleware",
        minimum_score=0.75,
    )

    assert len(results) == 1
    assert results[0].experience is strong


def test_ranking_preserves_insertion_order_for_equal_scores():
    store = MemoryStore()

    first = Experience(
        task="Fix authentication",
        action="First",
        outcome="Passed",
        success=True,
        verified=True,
    )

    second = Experience(
        task="Update authentication",
        action="Second",
        outcome="Passed",
        success=True,
        verified=True,
    )

    store.record_experience(first)
    store.record_experience(second)

    results = rank_experiences(
        store,
        "authentication",
    )

    assert len(results) == 2
    assert results[0].experience is first
    assert results[1].experience is second


def test_rank_experiences_does_not_modify_memory():
    store = MemoryStore()

    first = Experience(
        task="Fix authentication",
        action="First",
        outcome="Passed",
        success=True,
        verified=True,
    )

    second = Experience(
        task="Fix database",
        action="Second",
        outcome="Passed",
        success=True,
        verified=True,
    )

    store.record_experience(first)
    store.record_experience(second)

    before = list(store.experiences)

    rank_experiences(
        store,
        "Fix authentication",
    )

    assert store.experiences == before
