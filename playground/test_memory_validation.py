from datetime import datetime, timezone

from flycoder.tools.memory import (
    Experience,
    MemoryStore,
    validate_experience,
    validate_memory_store,
)


def test_valid_experience_passes_validation():
    experience = Experience(
        task="Fix authentication",
        action="Update middleware",
        outcome="Tests passed",
        success=True,
    )

    result = validate_experience(experience)

    assert result.valid is True
    assert result.errors == []


def test_blank_task_is_invalid():
    experience = Experience(
        task="   ",
        action="Update middleware",
        outcome="Tests passed",
        success=True,
    )

    result = validate_experience(experience)

    assert result.valid is False
    assert "task must be a non-empty string" in result.errors


def test_blank_action_is_invalid():
    experience = Experience(
        task="Fix authentication",
        action="",
        outcome="Tests passed",
        success=True,
    )

    result = validate_experience(experience)

    assert result.valid is False
    assert "action must be a non-empty string" in result.errors


def test_blank_outcome_is_invalid():
    experience = Experience(
        task="Fix authentication",
        action="Update middleware",
        outcome="",
        success=True,
    )

    result = validate_experience(experience)

    assert result.valid is False
    assert "outcome must be a non-empty string" in result.errors


def test_validation_checks_boolean_fields():
    experience = Experience(
        task="Fix authentication",
        action="Update middleware",
        outcome="Tests passed",
        success=True,
        verified=False,
    )

    result = validate_experience(experience)

    assert result.valid is True


def test_validation_accepts_explicit_recorded_timestamp():
    experience = Experience(
        task="Fix authentication",
        action="Update middleware",
        outcome="Tests passed",
        success=True,
        recorded_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    result = validate_experience(experience)

    assert result.valid is True


def test_memory_store_validation_preserves_experience_order():
    store = MemoryStore()

    first = Experience(
        task="Fix authentication",
        action="Update middleware",
        outcome="Passed",
        success=True,
    )
    second = Experience(
        task="Fix database",
        action="Update connection",
        outcome="Failed",
        success=False,
    )

    store.record_experience(first)
    store.record_experience(second)

    results = validate_memory_store(store)

    assert len(results) == 2
    assert results[0].valid is True
    assert results[1].valid is True
