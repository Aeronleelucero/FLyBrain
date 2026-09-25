from datetime import datetime, timezone

from flycoder.tools.learning import (
    LearningContext,
    LearningMemory,
)
from flycoder.tools.learning_validation import (
    build_validation_guidance,
    validate_learning_context,
)
from flycoder.tools.memory import Experience


BASE_TIME = datetime(
    2026,
    1,
    1,
    tzinfo=timezone.utc,
)


def make_memory(
    *,
    success=True,
    verified=False,
    decay_score=1.0,
    action="inspect_files",
):
    experience = Experience(
        task="Fix authentication",
        action=action,
        outcome="Completed",
        success=success,
        verified=verified,
        recorded_at=BASE_TIME,
    )

    return LearningMemory(
        experience=experience,
        similarity_score=0.9,
        outcome_score=(
            1.0
            if verified
            else 0.5
        )
        if success
        else 0.0,
        ranking_score=0.8,
        decay_score=decay_score,
    )


def test_verified_fresh_success_is_high_trust():
    context = LearningContext(
        task="Fix authentication",
        memories=[
            make_memory(
                success=True,
                verified=True,
                decay_score=1.0,
            )
        ],
    )

    result = validate_learning_context(
        context
    )

    assert len(result.valid_memories) == 1
    assert (
        result.valid_memories[0].trust_level
        == "HIGH"
    )
    assert result.has_reliable_learning is True


def test_verified_decayed_success_becomes_medium_trust():
    context = LearningContext(
        task="Fix authentication",
        memories=[
            make_memory(
                success=True,
                verified=True,
                decay_score=0.5,
            )
        ],
    )

    result = validate_learning_context(
        context
    )

    assert (
        result.valid_memories[0].trust_level
        == "MEDIUM"
    )


def test_verified_heavily_decayed_success_becomes_low_trust():
    context = LearningContext(
        task="Fix authentication",
        memories=[
            make_memory(
                success=True,
                verified=True,
                decay_score=0.2,
            )
        ],
    )

    result = validate_learning_context(
        context
    )

    assert (
        result.valid_memories[0].trust_level
        == "LOW"
    )


def test_unverified_success_is_not_high_trust():
    context = LearningContext(
        task="Fix authentication",
        memories=[
            make_memory(
                success=True,
                verified=False,
                decay_score=1.0,
            )
        ],
    )

    result = validate_learning_context(
        context
    )

    assert (
        result.valid_memories[0].trust_level
        == "MEDIUM"
    )

    assert result.high_trust_memories == []


def test_failed_experience_is_cautionary():
    context = LearningContext(
        task="Fix authentication",
        memories=[
            make_memory(
                success=False,
                verified=False,
                decay_score=1.0,
            )
        ],
    )

    result = validate_learning_context(
        context
    )

    assert (
        result.valid_memories[0].trust_level
        == "LOW"
    )

    assert result.has_reliable_learning is False
    assert result.warnings


def test_invalid_experience_is_rejected():
    invalid = Experience(
        task="",
        action="inspect_files",
        outcome="Completed",
        success=True,
        verified=True,
        recorded_at=BASE_TIME,
    )

    memory = LearningMemory(
        experience=invalid,
        similarity_score=0.9,
        outcome_score=1.0,
        ranking_score=0.9,
        decay_score=1.0,
    )

    context = LearningContext(
        task="Fix authentication",
        memories=[memory],
    )

    result = validate_learning_context(
        context
    )

    assert result.valid_memories == []
    assert len(result.rejected_memories) == 1
    assert (
        result.rejected_memories[0].trust_level
        == "REJECTED"
    )


def test_validation_does_not_modify_context():
    memory = make_memory(
        success=True,
        verified=True,
    )

    context = LearningContext(
        task="Fix authentication",
        memories=[memory],
    )

    before = list(
        context.memories
    )

    validate_learning_context(
        context
    )

    assert context.memories == before


def test_validation_guidance_mentions_trust():
    context = LearningContext(
        task="Fix authentication",
        memories=[
            make_memory(
                success=True,
                verified=True,
            ),
            make_memory(
                success=False,
                action="run_tests",
            ),
        ],
    )

    result = validate_learning_context(
        context
    )

    guidance = build_validation_guidance(
        result
    )

    assert any(
        "High-trust" in item
        for item in guidance
    )

    assert any(
        "Low-trust" in item
        for item in guidance
    )


def test_rejected_learning_memory_is_excluded_from_strategy_support():
    from flycoder.tools.action_selection import select_action
    from flycoder.tools.learning import (
        LearningContext,
        LearningMemory,
    )
    from flycoder.tools.memory import Experience
    from flycoder.tools.strategy import select_strategy
    from flycoder.tools.learning_validation import validate_learning_context

    experience = Experience(
        task="fix authentication bug",
        action="write code",
        outcome="invalid recorded experience",
        success="not-a-bool",
    )

    memory = LearningMemory(
        experience=experience,
        similarity_score=1.0,
        outcome_score=1.0,
        ranking_score=1.0,
        decay_score=1.0,
    )

    context = LearningContext(
        task="fix authentication bug",
        memories=[memory],
    )

    validation = validate_learning_context(context)

    assert len(validation.rejected_memories) == 1

    strategy = select_strategy(
        "fix authentication bug",
        context,
    )

    action = select_action(
        "fix authentication bug",
        strategy,
        context,
    )

    assert strategy.confidence == 0.55
    assert action.confidence == 0.55
