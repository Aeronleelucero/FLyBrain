from flycoder.tools.memory import (
    Experience,
    MemoryStore,
    SimilarityResult,
    find_similar_experiences,
    similarity_score,
)


def test_similarity_score_returns_zero_for_empty_text():
    score, matched_terms = similarity_score("", "authentication")

    assert score == 0.0
    assert matched_terms == []


def test_similarity_score_returns_zero_for_unrelated_text():
    score, matched_terms = similarity_score(
        "authentication middleware",
        "database connection",
    )

    assert score == 0.0
    assert matched_terms == []


def test_similarity_score_detects_shared_terms():
    score, matched_terms = similarity_score(
        "fix authentication middleware",
        "update authentication middleware",
    )

    assert score == 1.0
    assert matched_terms == [
        "authentication",
        "middleware",
    ]


def test_similarity_score_is_case_insensitive():
    first_score, first_terms = similarity_score(
        "Fix Authentication",
        "fix authentication",
    )

    second_score, second_terms = similarity_score(
        "fix authentication",
        "Fix Authentication",
    )

    assert first_score == 1.0
    assert second_score == 1.0
    assert first_terms == ["authentication"]
    assert second_terms == ["authentication"]


def test_similarity_score_ignores_common_stop_words():
    score, matched_terms = similarity_score(
        "fix the authentication issue",
        "fix an authentication issue",
    )

    assert score == 1.0
    assert matched_terms == [
        "authentication",
        "issue",
    ]


def test_similarity_score_ignores_generic_task_verbs():
    score, matched_terms = similarity_score(
        "Fix authentication",
        "Update authentication",
    )

    assert score == 1.0
    assert matched_terms == ["authentication"]


def test_similarity_result_contains_experience_and_matches():
    experience = Experience(
        task="Fix authentication middleware",
        action="Updated middleware",
        outcome="Passed",
        success=True,
    )

    result = SimilarityResult(
        experience=experience,
        score=0.5,
        matched_terms=["authentication", "middleware"],
    )

    assert result.experience is experience
    assert result.score == 0.5
    assert result.matched_terms == [
        "authentication",
        "middleware",
    ]


def test_find_similar_experiences_returns_matching_memory():
    store = MemoryStore()

    authentication = Experience(
        task="Fix authentication middleware",
        action="Updated middleware",
        outcome="Passed",
        success=True,
    )

    database = Experience(
        task="Fix database connection",
        action="Updated connection",
        outcome="Passed",
        success=True,
    )

    store.record_experience(authentication)
    store.record_experience(database)

    results = find_similar_experiences(
        store,
        "Fix authentication",
    )

    assert len(results) == 1
    assert results[0].experience is authentication
    assert results[0].matched_terms == [
        "authentication",
    ]


def test_find_similar_experiences_are_sorted_by_score():
    store = MemoryStore()

    weak = Experience(
        task="Fix authentication",
        action="Updated auth",
        outcome="Passed",
        success=True,
    )

    strong = Experience(
        task="Fix authentication middleware",
        action="Updated middleware",
        outcome="Passed",
        success=True,
    )

    store.record_experience(weak)
    store.record_experience(strong)

    results = find_similar_experiences(
        store,
        "Fix authentication middleware",
    )

    assert len(results) == 2
    assert results[0].experience is strong
    assert results[1].experience is weak
    assert results[0].score > results[1].score


def test_find_similar_experiences_preserves_insertion_order_for_ties():
    store = MemoryStore()

    first = Experience(
        task="Fix authentication",
        action="First approach",
        outcome="Passed",
        success=True,
    )

    second = Experience(
        task="Update authentication",
        action="Second approach",
        outcome="Passed",
        success=True,
    )

    store.record_experience(first)
    store.record_experience(second)

    results = find_similar_experiences(
        store,
        "authentication",
    )

    assert len(results) == 2
    assert results[0].experience is first
    assert results[1].experience is second
    assert results[0].score == results[1].score


def test_find_similar_experiences_supports_minimum_score():
    store = MemoryStore()

    weak = Experience(
        task="Fix authentication",
        action="Updated auth",
        outcome="Passed",
        success=True,
    )

    strong = Experience(
        task="Fix authentication middleware",
        action="Updated middleware",
        outcome="Passed",
        success=True,
    )

    store.record_experience(weak)
    store.record_experience(strong)

    results = find_similar_experiences(
        store,
        "Fix authentication middleware",
        minimum_score=0.75,
    )

    assert len(results) == 1
    assert results[0].experience is strong


def test_find_similar_experiences_returns_empty_for_no_matches():
    store = MemoryStore()

    store.record_experience(
        Experience(
            task="Fix authentication",
            action="Updated middleware",
            outcome="Passed",
            success=True,
        )
    )

    results = find_similar_experiences(
        store,
        "Database connection",
    )

    assert results == []


def test_find_similar_experiences_returns_empty_for_empty_store():
    store = MemoryStore()

    results = find_similar_experiences(
        store,
        "authentication",
    )

    assert results == []


def test_similarity_does_not_modify_memory_store():
    store = MemoryStore()

    experience = Experience(
        task="Fix authentication",
        action="Updated middleware",
        outcome="Passed",
        success=True,
    )

    store.record_experience(experience)
    before = list(store.experiences)

    find_similar_experiences(
        store,
        "authentication",
    )

    assert store.experiences == before