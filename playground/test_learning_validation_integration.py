from datetime import datetime, timezone

from flycoder.tools.action_selection import select_action
from flycoder.tools.learning import (
    LearningContext,
    LearningMemory,
)
from flycoder.tools.learning_validation import (
    validate_learning_context,
)
from flycoder.tools.memory import Experience
from flycoder.tools.strategy import select_strategy


BASE_TIME = datetime(
    2026,
    1,
    1,
    tzinfo=timezone.utc,
)


def make_memory(
    *,
    action,
    success=True,
    verified=True,
    decay_score=1.0,
):
    experience = Experience(
        task="Fix authentication middleware",
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


def test_invalid_learning_is_rejected_before_strategy_support():
    invalid = Experience(
        task="",
        action="repair authentication",
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
        task="Fix authentication middleware",
        memories=[memory],
    )

    decision = select_strategy(
        "Fix authentication middleware",
        context,
    )

    assert decision.supporting_memories == []
    assert decision.rejected_memory_count == 1
    assert decision.validation_guidance


def test_high_trust_learning_supports_strategy():
    memory = make_memory(
        action="repair authentication",
        verified=True,
        decay_score=1.0,
    )

    context = LearningContext(
        task="Fix authentication middleware",
        memories=[memory],
    )

    decision = select_strategy(
        "Fix authentication middleware",
        context,
    )

    assert decision.strategy == "repair"
    assert len(decision.supporting_memories) == 1
    assert decision.trusted_memory_count == 1
    assert decision.confidence >= 0.80


def test_low_trust_learning_does_not_increase_strategy_confidence():
    memory = make_memory(
        action="repair authentication",
        verified=True,
        decay_score=0.10,
    )

    context = LearningContext(
        task="Fix authentication middleware",
        memories=[memory],
    )

    decision = select_strategy(
        "Fix authentication middleware",
        context,
    )

    assert decision.strategy == "repair"
    assert len(decision.supporting_memories) == 1
    assert decision.trusted_memory_count == 0
    assert decision.confidence == 0.55
    assert any(
        "low trust" in risk.lower()
        for risk in decision.risks
    )


def test_high_trust_learning_supports_selected_action():
    memory = make_memory(
        action="explain error",
        verified=True,
        decay_score=1.0,
    )

    context = LearningContext(
        task="Fix authentication middleware",
        memories=[memory],
    )

    strategy = select_strategy(
        "Fix authentication middleware",
        context,
    )

    action = select_action(
        "Fix authentication middleware",
        strategy,
        context,
    )

    # The task/strategy remains the source of the deterministic
    # action candidate. Learning validates and supports the
    # recommendation but does not override it.
    assert action.action == "inspect_files"
    assert action.trusted_memory_count == 1
    assert action.confidence == 0.55
    assert action.validation_guidance


def test_invalid_learning_does_not_support_selected_action():
    invalid = Experience(
        task="",
        action="explain error",
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
        task="Fix authentication middleware",
        memories=[memory],
    )

    strategy = select_strategy(
        "Fix authentication middleware",
        context,
    )

    action = select_action(
        "Fix authentication middleware",
        strategy,
        context,
    )

    assert action.action == "inspect_files"
    assert action.trusted_memory_count == 0
    assert action.confidence == 0.55
    assert action.rejected_memory_count == 1


def test_failed_learning_remains_cautionary():
    memory = make_memory(
        action="repair authentication",
        success=False,
        verified=False,
        decay_score=1.0,
    )

    context = LearningContext(
        task="Fix authentication middleware",
        memories=[memory],
    )

    validation = validate_learning_context(
        context
    )

    assert len(validation.valid_memories) == 1
    assert validation.valid_memories[0].trust_level == "LOW"
    assert validation.has_reliable_learning is False

    decision = select_strategy(
        "Fix authentication middleware",
        context,
    )

    assert decision.strategy == "repair"
    assert decision.confidence == 0.55
    assert "cautionary context" in decision.reason.lower()


def test_validation_does_not_grant_execution_permission():
    memory = make_memory(
        action="write code",
        verified=True,
        decay_score=1.0,
    )

    context = LearningContext(
        task="Fix authentication middleware",
        memories=[memory],
    )

    validation = validate_learning_context(
        context
    )

    assert validation.has_reliable_learning is True

    # Validation produces evidence only.
    assert not hasattr(
        validation,
        "execute",
    )
    assert not hasattr(
        validation,
        "approve",
    )
