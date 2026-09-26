from datetime import timezone

import pytest

from flycoder.tools.task_lifecycle import (
    ALLOWED_TRANSITIONS,
    TASK_BLOCKED,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_INITIALIZING,
    TASK_NEEDS_HUMAN,
    TASK_PENDING,
    TASK_RUNNING,
    TASK_VERIFYING,
    TaskLifecycle,
)


def test_new_task_starts_pending():
    lifecycle = TaskLifecycle(task="inspect project")

    assert lifecycle.task == "inspect project"
    assert lifecycle.status == TASK_PENDING
    assert lifecycle.is_terminal is False
    assert lifecycle.started_at is None
    assert lifecycle.completed_at is None


def test_task_can_initialize_and_start():
    lifecycle = TaskLifecycle(task="inspect project")

    lifecycle.transition(TASK_INITIALIZING, "Task initialization started.")
    lifecycle.transition(TASK_RUNNING, "Task is ready to execute.")

    assert lifecycle.status == TASK_RUNNING
    assert lifecycle.started_at is not None
    assert lifecycle.started_at.tzinfo == timezone.utc
    assert len(lifecycle.transitions) == 2


def test_task_can_enter_verification():
    lifecycle = TaskLifecycle(task="run tests")

    lifecycle.transition(TASK_INITIALIZING)
    lifecycle.transition(TASK_RUNNING)
    lifecycle.transition(TASK_VERIFYING, "Execution finished.")

    assert lifecycle.status == TASK_VERIFYING
    assert lifecycle.is_terminal is False


def test_task_can_complete():
    lifecycle = TaskLifecycle(task="run tests")

    lifecycle.transition(TASK_INITIALIZING)
    lifecycle.transition(TASK_RUNNING)
    lifecycle.transition(TASK_VERIFYING)
    lifecycle.transition(TASK_COMPLETED, "Verification passed.")

    assert lifecycle.completed is True
    assert lifecycle.failed is False
    assert lifecycle.is_terminal is True
    assert lifecycle.completed_at is not None


@pytest.mark.parametrize(
    "terminal_state",
    [
        TASK_FAILED,
        TASK_BLOCKED,
        TASK_NEEDS_HUMAN,
    ],
)
def test_task_can_enter_terminal_interruption_states(terminal_state):
    lifecycle = TaskLifecycle(task="repair project")

    lifecycle.transition(TASK_INITIALIZING)
    lifecycle.transition(TASK_RUNNING)
    lifecycle.transition(
        terminal_state,
        "Execution cannot continue.",
    )

    assert lifecycle.status == terminal_state
    assert lifecycle.is_terminal is True
    assert lifecycle.completed_at is not None


def test_invalid_transition_is_rejected():
    lifecycle = TaskLifecycle(task="inspect project")

    with pytest.raises(ValueError, match="Invalid task lifecycle transition"):
        lifecycle.transition(TASK_COMPLETED)


def test_terminal_task_cannot_transition_again():
    lifecycle = TaskLifecycle(task="inspect project")

    lifecycle.transition(TASK_INITIALIZING)
    lifecycle.transition(TASK_RUNNING)
    lifecycle.transition(TASK_VERIFYING)
    lifecycle.transition(TASK_COMPLETED)

    with pytest.raises(ValueError, match="Invalid task lifecycle transition"):
        lifecycle.transition(TASK_RUNNING)


def test_actions_are_recorded():
    lifecycle = TaskLifecycle(task="review project")

    lifecycle.record_action("inspect_files")
    lifecycle.record_action("read_file")
    lifecycle.record_action("review_code")

    assert lifecycle.actions == [
        "inspect_files",
        "read_file",
        "review_code",
    ]


def test_errors_are_recorded():
    lifecycle = TaskLifecycle(task="repair project")

    lifecycle.record_error("Test failed.")
    lifecycle.record_error("Syntax error.")

    assert lifecycle.errors == [
        "Test failed.",
        "Syntax error.",
    ]


def test_iteration_is_recorded():
    lifecycle = TaskLifecycle(task="autonomous task")

    lifecycle.record_iteration(1)
    assert lifecycle.current_iteration == 1

    lifecycle.record_iteration(5)
    assert lifecycle.current_iteration == 5


def test_negative_iteration_is_rejected():
    lifecycle = TaskLifecycle(task="autonomous task")

    with pytest.raises(ValueError, match="iteration must be non-negative"):
        lifecycle.record_iteration(-1)


def test_transition_history_preserves_reason():
    lifecycle = TaskLifecycle(task="test task")

    transition = lifecycle.transition(
        TASK_INITIALIZING,
        "Preparing autonomous execution.",
    )

    assert transition.from_state == TASK_PENDING
    assert transition.to_state == TASK_INITIALIZING
    assert transition.reason == "Preparing autonomous execution."
    assert transition.timestamp.tzinfo == timezone.utc


def test_allowed_transitions_define_terminal_states():
    assert ALLOWED_TRANSITIONS[TASK_COMPLETED] == set()
    assert ALLOWED_TRANSITIONS[TASK_FAILED] == set()
    assert ALLOWED_TRANSITIONS[TASK_BLOCKED] == set()
    assert ALLOWED_TRANSITIONS[TASK_NEEDS_HUMAN] == set()
