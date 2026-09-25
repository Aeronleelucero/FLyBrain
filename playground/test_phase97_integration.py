from pathlib import Path

from flycoder.agent import FlyCoderAgent
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace
from flycoder.tools.learning_feedback import (
    LearningOutcome,
    record_learning_outcome,
)
from flycoder.tools.memory import MemoryStore
from flycoder.tools.planning_pipeline import create_integrated_plan


def test_phase97_learning_loop_from_execution_to_next_plan(
    tmp_path: Path,
):
    """
    Verify the complete Phase 9.7 learning loop:

    execution
        -> learning feedback
        -> MemoryStore
        -> learning retrieval
        -> learning validation
        -> strategy selection
        -> action selection

    Learning remains advisory and does not grant execution approval.
    """

    (tmp_path / "auth.py").write_text(
        "def authenticate():\n"
        "    return True\n",
        encoding="utf-8",
    )

    store = MemoryStore()

    # --------------------------------------------------------------
    # Phase 1: Execute an action through the real agent.
    # --------------------------------------------------------------

    agent = FlyCoderAgent(
        tmp_path,
        memory_store=store,
    )

    state = CodingState(
        task="Inspect the authentication project",
    )

    execution = agent.run_once(state)

    assert execution.action == "inspect_files"
    assert execution.success is True

    assert len(store.experiences) == 1
    assert store.experiences[0].action == "inspect_files"

    # --------------------------------------------------------------
    # Phase 2: Record a verified successful repair experience.
    #
    # This represents the knowledge the agent should be able to
    # retrieve during the next planning cycle.
    # --------------------------------------------------------------

    learning_outcome = LearningOutcome(
        task="fix authentication error",
        task_intent="repair",
        action="repair authentication error",
        success=True,
        strategy="repair",
        tests_run=True,
        tests_passed=True,
        repair_applied=True,
        message="Authentication repair succeeded and tests passed.",
        evidence=[
            "Authentication repair was applied.",
            "Relevant tests passed.",
        ],
        lessons=[
            "Confirm the authentication failure before modifying code.",
        ],
    )

    recorded = record_learning_outcome(
        store,
        learning_outcome,
        files=["auth.py"],
    )

    assert recorded.success is True
    assert recorded.verified is True
    assert recorded.action == "repair authentication error"
    assert "auth.py" in recorded.files

    assert len(store.experiences) == 2

    # --------------------------------------------------------------
    # Phase 3: Start a new planning cycle using the SAME store.
    # --------------------------------------------------------------

    planning_state = CodingState(
        task="fix authentication error",
    )

    workspace = Workspace(tmp_path)

    integrated = create_integrated_plan(
        workspace,
        planning_state,
        memory_store=store,
    )

    # --------------------------------------------------------------
    # Phase 4: Learning was retrieved.
    # --------------------------------------------------------------

    assert integrated.learning.task == "fix authentication error"
    assert integrated.learning.memories

    matching_memories = [
        memory
        for memory in integrated.learning.memories
        if memory.experience.action == "repair authentication error"
    ]

    assert matching_memories

    # --------------------------------------------------------------
    # Phase 5: Retrieved learning was validated.
    # --------------------------------------------------------------

    assert integrated.learning_validation.valid_memories

    validated_memories = [
        memory
        for memory in integrated.learning_validation.valid_memories
        if (
            memory.memory.experience.action
            == "repair authentication error"
        )
    ]

    assert validated_memories

    validated = validated_memories[0]

    assert validated.valid is True
    assert validated.trust_level == "HIGH"

    # --------------------------------------------------------------
    # Phase 6: Strategy consumed the validated learning.
    # --------------------------------------------------------------

    assert integrated.strategy.strategy == "repair"

    strategy_support = [
        memory
        for memory in integrated.strategy.supporting_memories
        if memory.experience.action == "repair authentication error"
    ]

    assert strategy_support

    assert (
        "verified experience supports this strategy"
        in integrated.strategy.reason
    )

    assert integrated.strategy.trusted_memory_count >= 1
    assert integrated.strategy.rejected_memory_count == 0

    # --------------------------------------------------------------
    # Phase 7: Action consumed the same validated learning.
    # --------------------------------------------------------------

    assert integrated.action.strategy == "repair"

    assert (
        "Previous experience supports this action."
        in integrated.action.reason
    )

    assert integrated.action.trusted_memory_count >= 1
    assert integrated.action.rejected_memory_count == 0

    # --------------------------------------------------------------
    # Phase 8: Learning remains advisory.
    #
    # The planning cycle must not grant repair approval or mutate
    # the CodingState's repair authorization flags.
    # --------------------------------------------------------------

    assert planning_state.repair_approved is False
    assert planning_state.repair_applied is False
    assert planning_state.repair_proposed is False

    # Planning itself must not execute an action.
    assert len(store.experiences) == 2
