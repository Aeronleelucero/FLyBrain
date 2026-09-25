"""Tests for FLY-CODER Phase 9.2 action selection."""

from flycoder.tools.action_selection import (
    ActionDecision,
    select_action,
)
from flycoder.tools.learning import (
    LearningContext,
    LearningMemory,
)
from flycoder.tools.memory import Experience
from flycoder.tools.strategy import StrategyDecision


def make_strategy(
    strategy: str,
    confidence: float = 0.80,
) -> StrategyDecision:
    """Build a strategy decision for testing."""

    return StrategyDecision(
        strategy=strategy,
        reason="Test strategy.",
        confidence=confidence,
    )


def make_memory(
    action: str,
    verified: bool = True,
) -> LearningMemory:
    """Build a learning memory for testing."""

    experience = Experience(
        task="Add authentication validation",
        action=action,
        outcome="Tests passed",
        success=True,
        verified=verified,
    )

    return LearningMemory(
        experience=experience,
        similarity_score=0.80,
        outcome_score=1.0,
        ranking_score=0.85,
        decay_score=1.0,
    )


def test_select_action_returns_action_decision():
    decision = select_action(
        "Add authentication validation",
        make_strategy("create_new"),
    )

    assert isinstance(decision, ActionDecision)


def test_create_new_selects_inspect_files():
    decision = select_action(
        "Add authentication validation",
        make_strategy("create_new"),
    )

    assert decision.action == "inspect_files"
    assert decision.strategy == "create_new"


def test_modify_existing_selects_inspect_files():
    decision = select_action(
        "Update authentication validation",
        make_strategy("modify_existing"),
    )

    assert decision.action == "inspect_files"


def test_refactor_selects_inspect_files():
    decision = select_action(
        "Refactor authentication validation",
        make_strategy("refactor"),
    )

    assert decision.action == "inspect_files"


def test_repair_without_explicit_error_selects_inspection():
    decision = select_action(
        "Repair authentication handling",
        make_strategy("repair"),
    )

    assert decision.action == "inspect_files"


def test_repair_with_error_selects_explain_error():
    decision = select_action(
        "Fix authentication error",
        make_strategy("repair"),
    )

    assert decision.action == "explain_error"


def test_test_first_selects_run_tests():
    decision = select_action(
        "Test authentication validation",
        make_strategy("test_first"),
    )

    assert decision.action == "run_tests"


def test_explicit_verification_overrides_generic_action():
    decision = select_action(
        "Verify authentication validation",
        make_strategy("create_new"),
    )

    assert decision.action == "run_tests"


def test_unknown_strategy_falls_back_to_inspection():
    decision = select_action(
        "Understand authentication behavior",
        make_strategy("unknown"),
    )

    assert decision.action == "inspect_files"


def test_empty_task_selects_inspection():
    decision = select_action(
        "",
        make_strategy("inspect_only"),
    )

    assert decision.action == "inspect_files"
    assert decision.confidence == 1.0


def test_action_confidence_is_bounded():
    decision = select_action(
        "Add authentication validation",
        make_strategy("create_new", confidence=1.0),
    )

    assert 0.0 <= decision.confidence <= 1.0


def test_learning_context_can_support_action():
    context = LearningContext(
        task="Add authentication validation",
        memories=[
            make_memory("Inspect authentication files"),
        ],
    )

    decision = select_action(
        "Add authentication validation",
        make_strategy("create_new"),
        context,
    )

    assert decision.action == "inspect_files"
    assert "Previous experience supports this action." in (
        decision.reason
    )


def test_unrelated_memory_does_not_change_action():
    context = LearningContext(
        task="Add authentication validation",
        memories=[
            make_memory("Run database migrations"),
        ],
    )

    decision = select_action(
        "Add authentication validation",
        make_strategy("create_new"),
        context,
    )

    assert decision.action == "inspect_files"
    assert "Previous experience supports this action." not in (
        decision.reason
    )


def test_action_selection_does_not_modify_context():
    context = LearningContext(
        task="Add authentication validation",
        memories=[
            make_memory("Inspect authentication files"),
        ],
    )

    before = list(context.memories)

    select_action(
        "Add authentication validation",
        make_strategy("create_new"),
        context,
    )

    assert context.memories == before


def test_action_selection_does_not_modify_strategy():
    strategy = make_strategy("create_new")

    before = (
        strategy.strategy,
        strategy.reason,
        strategy.confidence,
        list(strategy.supporting_memories),
        list(strategy.risks),
    )

    select_action(
        "Add authentication validation",
        strategy,
    )

    after = (
        strategy.strategy,
        strategy.reason,
        strategy.confidence,
        list(strategy.supporting_memories),
        list(strategy.risks),
    )

    assert after == before


def test_action_selection_does_not_execute_action():
    decision = select_action(
        "Add authentication validation",
        make_strategy("create_new"),
    )

    assert decision.action == "inspect_files"

    # Selection only returns metadata. It never returns an
    # ActionResult and never executes the registered action.
    assert not hasattr(decision, "success")
    assert not hasattr(decision, "data")


def test_selected_action_is_registered():
    from flycoder.actions import create_action_registry

    registry = create_action_registry()

    decision = select_action(
        "Add authentication validation",
        make_strategy("create_new"),
    )

    assert decision.action in registry.list_actions()


def test_research_first_selects_inspection():
    decision = select_action(
        "Research the authentication implementation",
        make_strategy("research_first"),
    )

    assert decision.action == "inspect_files"


def test_action_reason_is_present():
    decision = select_action(
        "Add authentication validation",
        make_strategy("create_new"),
    )

    assert decision.reason


def test_action_risks_are_present():
    decision = select_action(
        "Add authentication validation",
        make_strategy("create_new"),
    )

    assert decision.risks
