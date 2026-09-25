from datetime import datetime, timezone

from flycoder.state import CodingState
from flycoder.tools.action_selection import select_action
from flycoder.tools.learning import (
    LearningContext,
    LearningMemory,
    build_learning_guidance,
)
from flycoder.tools.memory import Experience, MemoryStore
from flycoder.tools.planning_pipeline import (
    create_integrated_plan,
)
from flycoder.tools.strategy import (
    select_strategy,
)
from flycoder.tools.filesystem import Workspace


def make_memory(
    action: str,
    *,
    success: bool = True,
    verified: bool = False,
    error: str | None = None,
) -> LearningMemory:
    experience = Experience(
        task="Add authentication validation",
        action=action,
        outcome=(
            "Previous action succeeded."
            if success
            else "Previous action failed."
        ),
        success=success,
        files=["auth.py"],
        errors=[error] if error else [],
        notes=[],
        verified=verified,
        recorded_at=datetime.now(
            timezone.utc
        ),
    )

    return LearningMemory(
        experience=experience,
        similarity_score=0.9,
        outcome_score=0.9 if success else 0.2,
        ranking_score=0.9,
        decay_score=1.0,
    )


def test_learning_guidance_contains_successful_experience():
    context = LearningContext(
        task="Add authentication validation",
        memories=[
            make_memory(
                "inspect_files",
                success=True,
                verified=True,
            )
        ],
    )

    guidance = build_learning_guidance(
        context
    )

    assert guidance
    assert (
        "similar verified solution"
        in guidance[0].lower()
    )
    assert "inspect_files" in guidance[0]


def test_learning_guidance_contains_failed_experience():
    context = LearningContext(
        task="Add authentication validation",
        memories=[
            make_memory(
                "run_tests",
                success=False,
                error="Authentication test failed.",
            )
        ],
    )

    guidance = build_learning_guidance(
        context
    )

    assert guidance
    assert (
        "previous attempt failed"
        in guidance[0].lower()
    )
    assert (
        "Authentication test failed."
        in guidance[0]
    )


def test_learning_support_reaches_strategy():
    context = LearningContext(
        task="Fix authentication validation",
        memories=[
            make_memory(
                "fix_error",
                success=True,
                verified=True,
            )
        ],
    )

    decision = select_strategy(
        "Fix authentication validation",
        context,
    )

    assert decision.strategy == "repair"
    assert decision.supporting_memories
    assert decision.learning_guidance


def test_verified_learning_support_increases_strategy_confidence():
    without_memory = select_strategy(
        "Fix authentication validation"
    )

    context = LearningContext(
        task="Fix authentication validation",
        memories=[
            make_memory(
                "fix_error",
                success=True,
                verified=True,
            )
        ],
    )

    with_memory = select_strategy(
        "Fix authentication validation",
        context,
    )

    assert (
        with_memory.confidence
        > without_memory.confidence
    )


def test_learning_support_reaches_action():
    context = LearningContext(
        task="Add authentication validation",
        memories=[
            make_memory(
                "inspect_files",
                success=True,
                verified=True,
            )
        ],
    )

    strategy = select_strategy(
        "Add authentication validation",
        context,
    )

    decision = select_action(
        "Add authentication validation",
        strategy,
        context,
    )

    assert decision.action == "inspect_files"
    assert decision.learning_guidance
    assert (
        "Previous experience supports this action."
        in decision.reason
    )


def test_learning_support_increases_action_confidence():
    strategy = select_strategy(
        "Add authentication validation"
    )

    without_memory = select_action(
        "Add authentication validation",
        strategy,
    )

    context = LearningContext(
        task="Add authentication validation",
        memories=[
            make_memory(
                "inspect_files",
                success=True,
                verified=True,
            )
        ],
    )

    with_memory = select_action(
        "Add authentication validation",
        strategy,
        context,
    )

    assert (
        with_memory.confidence
        > without_memory.confidence
    )


def test_empty_learning_context_is_safe():
    context = LearningContext(
        task="Inspect the project"
    )

    strategy = select_strategy(
        "Inspect the project",
        context,
    )

    action = select_action(
        "Inspect the project",
        strategy,
        context,
    )

    assert strategy.strategy == "inspect_only"
    assert action.action == "inspect_files"
    assert strategy.learning_guidance == []
    assert action.learning_guidance == []


def test_planning_preserves_supplied_memory_store(tmp_path):
    workspace = Workspace(tmp_path)

    store = MemoryStore()

    store.record_experience(
        Experience(
            task="Fix authentication validation",
            action="fix_error",
            outcome="Fixed successfully.",
            success=True,
            files=["auth.py"],
            verified=True,
        )
    )

    state = CodingState(
        task="Fix authentication validation"
    )

    result = create_integrated_plan(
        workspace,
        state,
        memory_store=store,
    )

    assert result.learning.memories
    assert result.strategy.strategy == "repair"
    assert result.strategy.supporting_memories


def test_learning_does_not_execute_actions(tmp_path):
    workspace = Workspace(tmp_path)

    store = MemoryStore()

    store.record_experience(
        Experience(
            task="Add authentication validation",
            action="write_code",
            outcome="Previous implementation succeeded.",
            success=True,
            files=["auth.py"],
            verified=True,
        )
    )

    state = CodingState(
        task="Add authentication validation"
    )

    result = create_integrated_plan(
        workspace,
        state,
        memory_store=store,
    )

    assert result.learning.memories

    # Planning only.
    # No files should be created or modified.
    assert list(tmp_path.iterdir()) == []

    # Learning must not grant repair approval.
    assert state.repair_approved is False

    # Planning must not mark a repair as applied.
    assert state.repair_applied is False
