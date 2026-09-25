"""Tests for FLY-CODER Phase 9.1 strategy selection."""

from flycoder.tools.learning import (
    LearningContext,
    LearningMemory,
)
from flycoder.tools.memory import Experience
from flycoder.tools.strategy import (
    STRATEGIES,
    StrategyDecision,
    select_strategy,
)


def make_memory(
    *,
    task: str = "Fix authentication validation",
    action: str = "Fix authentication middleware",
    success: bool = True,
    verified: bool = True,
) -> LearningMemory:
    """Build a learning memory for strategy tests."""

    experience = Experience(
        task=task,
        action=action,
        outcome="Tests passed",
        success=success,
        verified=verified,
    )

    return LearningMemory(
        experience=experience,
        similarity_score=0.8,
        outcome_score=1.0 if verified else 0.5,
        ranking_score=0.86,
        decay_score=1.0,
    )


def test_strategy_vocabulary_is_stable():
    assert STRATEGIES == (
        "inspect_only",
        "modify_existing",
        "create_new",
        "refactor",
        "repair",
        "test_first",
        "research_first",
    )


def test_select_strategy_returns_strategy_decision():
    result = select_strategy(
        "Fix authentication validation"
    )

    assert isinstance(result, StrategyDecision)


def test_fix_task_selects_repair():
    result = select_strategy(
        "Fix authentication validation"
    )

    assert result.strategy == "repair"


def test_refactor_task_selects_refactor():
    result = select_strategy(
        "Refactor authentication middleware"
    )

    assert result.strategy == "refactor"


def test_testing_task_selects_test_first():
    result = select_strategy(
        "Add tests for authentication"
    )

    assert result.strategy == "test_first"


def test_create_task_selects_create_new():
    result = select_strategy(
        "Create a new authentication service"
    )

    assert result.strategy == "create_new"


def test_update_task_selects_modify_existing():
    result = select_strategy(
        "Update authentication middleware"
    )

    assert result.strategy == "modify_existing"


def test_unknown_task_defaults_to_inspect_only():
    result = select_strategy(
        "Investigate the authentication behavior"
    )

    assert result.strategy == "inspect_only"


def test_empty_task_defaults_to_inspect_only():
    result = select_strategy("")

    assert result.strategy == "inspect_only"
    assert result.confidence == 1.0
    assert result.risks


def test_repair_strategy_has_risk():
    result = select_strategy(
        "Fix the broken authentication flow"
    )

    assert result.strategy == "repair"
    assert result.risks


def test_verified_memory_supports_strategy():
    context = LearningContext(
        task="Fix authentication validation",
        memories=[
            make_memory(
                action="Fix authentication middleware",
                verified=True,
            )
        ],
    )

    result = select_strategy(
        "Fix authentication validation",
        context,
    )

    assert result.strategy == "repair"
    assert len(result.supporting_memories) == 1
    assert result.confidence == 0.80


def test_unverified_memory_supports_strategy():
    context = LearningContext(
        task="Fix authentication validation",
        memories=[
            make_memory(
                action="Fix authentication middleware",
                verified=False,
            )
        ],
    )

    result = select_strategy(
        "Fix authentication validation",
        context,
    )

    assert result.strategy == "repair"
    assert len(result.supporting_memories) == 1
    assert result.confidence == 0.65


def test_unrelated_memory_does_not_support_strategy():
    context = LearningContext(
        task="Fix authentication validation",
        memories=[
            make_memory(
                action="Create dashboard component",
                verified=True,
            )
        ],
    )

    result = select_strategy(
        "Fix authentication validation",
        context,
    )

    assert result.strategy == "repair"
    assert result.supporting_memories == []
    assert result.confidence == 0.55


def test_strategy_selection_does_not_modify_context():
    memory = make_memory()

    context = LearningContext(
        task="Fix authentication validation",
        memories=[memory],
    )

    before = list(context.memories)

    select_strategy(
        "Fix authentication validation",
        context,
    )

    assert context.memories == before


def test_strategy_selection_does_not_modify_memory():
    memory = make_memory()

    context = LearningContext(
        task="Fix authentication validation",
        memories=[memory],
    )

    experience = memory.experience

    select_strategy(
        "Fix authentication validation",
        context,
    )

    assert experience.task == "Fix authentication validation"
    assert experience.action == "Fix authentication middleware"
    assert experience.success is True
    assert experience.verified is True
