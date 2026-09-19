from datetime import datetime, timedelta, timezone

from flycoder.tools.learning import (
    LearningContext,
    build_learning_guidance,
    retrieve_learning_context,
)
from flycoder.tools.memory import (
    Experience,
    MemoryStore,
)


BASE_TIME = datetime(
    2026,
    1,
    1,
    tzinfo=timezone.utc,
)


def test_empty_task_returns_empty_learning_context():
    store = MemoryStore()

    context = retrieve_learning_context(
        store,
        "   ",
        now=BASE_TIME,
    )

    assert isinstance(context, LearningContext)
    assert context.task == ""
    assert context.memories == []


def test_relevant_memory_is_retrieved():
    store = MemoryStore()

    experience = Experience(
        task="Fix authentication middleware",
        action="Update middleware",
        outcome="Tests passed",
        success=True,
        verified=True,
        recorded_at=BASE_TIME,
    )

    store.record_experience(experience)

    context = retrieve_learning_context(
        store,
        "Fix authentication middleware",
        now=BASE_TIME,
    )

    assert len(context.memories) == 1
    assert context.memories[0].experience is experience


def test_invalid_memory_is_excluded():
    store = MemoryStore()

    invalid = Experience(
        task="",
        action="Update middleware",
        outcome="Tests passed",
        success=True,
        verified=True,
        recorded_at=BASE_TIME,
    )

    store.record_experience(invalid)

    context = retrieve_learning_context(
        store,
        "authentication middleware",
        now=BASE_TIME,
    )

    assert context.memories == []


def test_successful_and_failed_memories_are_separated():
    store = MemoryStore()

    successful = Experience(
        task="Fix authentication middleware",
        action="Use middleware update",
        outcome="Passed",
        success=True,
        verified=True,
        recorded_at=BASE_TIME,
    )

    failed = Experience(
        task="Fix authentication middleware",
        action="Change unrelated configuration",
        outcome="Failed",
        success=False,
        errors=["Connection refused"],
        recorded_at=BASE_TIME,
    )

    store.record_experience(successful)
    store.record_experience(failed)

    context = retrieve_learning_context(
        store,
        "Fix authentication middleware",
        now=BASE_TIME,
    )

    assert len(context.successful_memories) == 1
    assert context.successful_memories[0].experience is successful

    assert len(context.failed_memories) == 1
    assert context.failed_memories[0].experience is failed


def test_verified_memories_are_identified():
    store = MemoryStore()

    verified = Experience(
        task="Fix authentication",
        action="Verified middleware update",
        outcome="Passed",
        success=True,
        verified=True,
        recorded_at=BASE_TIME,
    )

    unverified = Experience(
        task="Fix authentication",
        action="Unverified middleware update",
        outcome="Passed",
        success=True,
        verified=False,
        recorded_at=BASE_TIME,
    )

    store.record_experience(verified)
    store.record_experience(unverified)

    context = retrieve_learning_context(
        store,
        "Fix authentication",
        now=BASE_TIME,
    )

    assert len(context.verified_memories) == 1
    assert context.verified_memories[0].experience is verified


def test_learning_context_preserves_decay_score():
    store = MemoryStore()

    experience = Experience(
        task="Fix authentication middleware",
        action="Update middleware",
        outcome="Passed",
        success=True,
        verified=True,
        recorded_at=BASE_TIME,
    )

    store.record_experience(experience)

    context = retrieve_learning_context(
        store,
        "Fix authentication middleware",
        now=BASE_TIME + timedelta(days=30),
        half_life_days=30,
    )

    assert len(context.memories) == 1
    assert context.memories[0].decay_score == 0.5


def test_learning_context_does_not_modify_store():
    store = MemoryStore()

    experience = Experience(
        task="Fix authentication",
        action="Update middleware",
        outcome="Passed",
        success=True,
        verified=True,
        recorded_at=BASE_TIME,
    )

    store.record_experience(experience)
    before = list(store.experiences)

    retrieve_learning_context(
        store,
        "Fix authentication",
        now=BASE_TIME,
    )

    assert store.experiences == before


def test_learning_guidance_mentions_verified_success():
    store = MemoryStore()

    experience = Experience(
        task="Fix authentication",
        action="Update authentication middleware",
        outcome="Passed",
        success=True,
        verified=True,
        recorded_at=BASE_TIME,
    )

    store.record_experience(experience)

    context = retrieve_learning_context(
        store,
        "Fix authentication",
        now=BASE_TIME,
    )

    guidance = build_learning_guidance(context)

    assert guidance == [
        "A similar verified solution previously succeeded: "
        "Update authentication middleware."
    ]


def test_learning_guidance_mentions_unverified_success():
    store = MemoryStore()

    experience = Experience(
        task="Fix authentication",
        action="Try middleware update",
        outcome="Passed",
        success=True,
        verified=False,
        recorded_at=BASE_TIME,
    )

    store.record_experience(experience)

    context = retrieve_learning_context(
        store,
        "Fix authentication",
        now=BASE_TIME,
    )

    guidance = build_learning_guidance(context)

    assert guidance == [
        "A similar previous solution succeeded: "
        "Try middleware update."
    ]


def test_learning_guidance_mentions_failure_errors():
    store = MemoryStore()

    experience = Experience(
        task="Fix authentication",
        action="Change configuration",
        outcome="Failed",
        success=False,
        errors=["Connection refused"],
        recorded_at=BASE_TIME,
    )

    store.record_experience(experience)

    context = retrieve_learning_context(
        store,
        "Fix authentication",
        now=BASE_TIME,
    )

    guidance = build_learning_guidance(context)

    assert guidance == [
        "A similar previous attempt failed with: "
        "Connection refused."
    ]


def test_learning_guidance_mentions_failure_without_errors():
    store = MemoryStore()

    experience = Experience(
        task="Fix authentication",
        action="Change configuration",
        outcome="Failed",
        success=False,
        recorded_at=BASE_TIME,
    )

    store.record_experience(experience)

    context = retrieve_learning_context(
        store,
        "Fix authentication",
        now=BASE_TIME,
    )

    guidance = build_learning_guidance(context)

    assert guidance == [
        "A similar previous attempt failed: "
        "Change configuration."
    ]
