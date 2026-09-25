from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.execution import (
    ExecutionObservation,
    observe_action,
)


def test_observe_action_returns_observation():
    state = CodingState(task="Inspect project")

    result = ActionResult(
        action="inspect_files",
        success=True,
        message="Files inspected.",
    )

    observation = observe_action(result, state)

    assert isinstance(observation, ExecutionObservation)
    assert observation.action == "inspect_files"
    assert observation.success is True


def test_test_failure_recommends_error_inspection():
    state = CodingState(task="Fix failing tests")

    result = ActionResult(
        action="run_tests",
        success=False,
        message="Tests failed.",
    )

    observation = observe_action(result, state)

    assert observation.next_action == "fix_error"
    assert observation.success is False


def test_test_success_does_not_force_next_action():
    state = CodingState(task="Run tests")

    result = ActionResult(
        action="run_tests",
        success=True,
        message="All tests passed.",
    )

    observation = observe_action(result, state)

    assert observation.next_action is None
    assert observation.success is True


def test_error_inspection_recommends_reading_file():
    state = CodingState(task="Fix failing test")
    state.error_inspected = True
    state.current_file = "example.py"

    result = ActionResult(
        action="fix_error",
        success=False,
        message="Failure inspected.",
    )

    observation = observe_action(result, state)

    assert observation.next_action == "read_file"


def test_read_file_recommends_repair_proposal():
    state = CodingState(task="Fix failing test")
    state.task_intent = "repair"
    state.error_inspected = True
    state.current_file = "example.py"
    state.current_file_content = "assert False"

    result = ActionResult(
        action="read_file",
        success=True,
        message="File read.",
    )

    observation = observe_action(result, state)

    assert observation.next_action == "propose_repair"


def test_repair_proposal_requires_human_input():
    state = CodingState(task="Fix failing test")

    result = ActionResult(
        action="propose_repair",
        success=True,
        message="Repair proposal created.",
    )

    observation = observe_action(result, state)

    assert observation.next_action is None
    assert observation.requires_human_input is True


def test_user_input_boundary_always_stops_adaptation():
    state = CodingState(task="Fix failing test")
    state.user_input_needed = True

    result = ActionResult(
        action="fix_error",
        success=False,
        message="Failure inspected.",
    )

    observation = observe_action(result, state)

    assert observation.next_action is None
    assert observation.requires_human_input is True


def test_write_code_recommends_tests_after_success():
    state = CodingState(task="Implement feature")

    result = ActionResult(
        action="write_code",
        success=True,
        message="File written.",
    )

    observation = observe_action(result, state)

    assert observation.next_action == "run_tests"


def test_failed_write_does_not_continue_automatically():
    state = CodingState(task="Implement feature")

    result = ActionResult(
        action="write_code",
        success=False,
        message="Write failed.",
    )

    observation = observe_action(result, state)

    assert observation.next_action is None


def test_observation_does_not_modify_state():
    state = CodingState(task="Fix failing test")
    state.error_inspected = True
    state.current_file = "example.py"

    before = (
        state.error_inspected,
        state.current_file,
        state.repair_proposed,
        state.user_input_needed,
    )

    result = ActionResult(
        action="fix_error",
        success=False,
        message="Failure inspected.",
    )

    observe_action(result, state)

    after = (
        state.error_inspected,
        state.current_file,
        state.repair_proposed,
        state.user_input_needed,
    )

    assert after == before
