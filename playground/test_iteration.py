from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.execution import ExecutionObservation
from flycoder.tools.iteration import (
    IterationObservation,
    build_state_signature,
    observe_iteration,
)


def test_iteration_observation_records_action():
    state = CodingState(task="Inspect project")

    result = ActionResult(
        action="inspect_files",
        success=True,
        message="Files inspected.",
    )

    observation = observe_iteration(
        1,
        result,
        state,
    )

    assert isinstance(observation, IterationObservation)
    assert observation.iteration == 1
    assert observation.action == "inspect_files"
    assert observation.success is True
    assert observation.message == "Files inspected."


def test_state_signature_is_deterministic():
    state = CodingState(task="Inspect project")

    first = build_state_signature(state)
    second = build_state_signature(state)

    assert first == second


def test_state_change_is_detected():
    state = CodingState(task="Inspect project")

    result = ActionResult(
        action="read_file",
        success=True,
        message="File read.",
    )

    previous_signature = build_state_signature(state)

    state.current_file = "example.py"
    state.current_file_content = "print('hello')"

    observation = observe_iteration(
        1,
        result,
        state,
        previous_signature=previous_signature,
    )

    assert observation.progress is True
    assert "Task state changed" in observation.notes[1]


def test_no_state_change_is_not_progress():
    state = CodingState(task="Inspect project")

    result = ActionResult(
        action="inspect_files",
        success=True,
        message="Files inspected.",
    )

    previous_signature = build_state_signature(state)

    observation = observe_iteration(
        1,
        result,
        state,
        previous_signature=previous_signature,
    )

    assert observation.progress is False
    assert "No relevant task-state change" in observation.notes[1]


def test_execution_observation_next_action_is_preserved():
    state = CodingState(task="Repair project")

    result = ActionResult(
        action="run_tests",
        success=False,
        message="Tests failed.",
    )

    execution = ExecutionObservation(
        action="run_tests",
        success=False,
        message="Tests failed.",
        next_action="fix_error",
        reason="Inspect the test failure.",
    )

    observation = observe_iteration(
        1,
        result,
        state,
        execution,
    )

    assert observation.next_action == "fix_error"


def test_successful_iteration_records_success_note():
    state = CodingState(task="Inspect project")

    result = ActionResult(
        action="read_file",
        success=True,
        message="File read.",
    )

    observation = observe_iteration(
        1,
        result,
        state,
    )

    assert "Action completed successfully." in observation.notes


def test_failed_iteration_records_failure_note():
    state = CodingState(task="Run tests")

    result = ActionResult(
        action="run_tests",
        success=False,
        message="Tests failed.",
    )

    observation = observe_iteration(
        1,
        result,
        state,
    )

    assert "Action failed." in observation.notes


def test_finished_state_is_recorded():
    state = CodingState(task="Finish task")
    state.finished = True

    result = ActionResult(
        action="finish",
        success=True,
        message="Task completed.",
    )

    observation = observe_iteration(
        1,
        result,
        state,
    )

    assert observation.progress is False
    assert "Task is marked finished." in observation.notes


def test_human_input_state_is_recorded():
    state = CodingState(task="Repair project")
    state.user_input_needed = True

    result = ActionResult(
        action="propose_repair",
        success=True,
        message="Repair proposed.",
    )

    observation = observe_iteration(
        1,
        result,
        state,
    )

    assert "Human input is required." in observation.notes
