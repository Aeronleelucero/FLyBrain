from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.recovery import (
    RecoveryDecision,
    classify_failure,
    recover_from_failure,
)


def test_success_is_not_a_failure():
    result = ActionResult(
        action="run_tests",
        success=True,
        message="All tests passed.",
    )

    assert classify_failure(result) == "success"

    decision = recover_from_failure(
        result,
        CodingState(task="Run tests"),
    )

    assert isinstance(decision, RecoveryDecision)
    assert decision.failed is False
    assert decision.recovery_action is None


def test_failed_tests_recommend_error_inspection():
    result = ActionResult(
        action="run_tests",
        success=False,
        message="Tests failed.",
    )

    state = CodingState(task="Run tests")

    assert classify_failure(result) == "test_failure"

    decision = recover_from_failure(result, state)

    assert decision.failed is True
    assert decision.recovery_action == "fix_error"
    assert decision.retry_allowed is False


def test_failed_read_recommends_workspace_inspection():
    result = ActionResult(
        action="read_file",
        success=False,
        message="File not found.",
    )

    decision = recover_from_failure(
        result,
        CodingState(task="Read file"),
    )

    assert classify_failure(result) == "read_failure"
    assert decision.recovery_action == "inspect_files"
    assert decision.retry_allowed is False


def test_failed_write_does_not_blindly_retry():
    result = ActionResult(
        action="write_code",
        success=False,
        message="Permission denied.",
    )

    decision = recover_from_failure(
        result,
        CodingState(task="Repair code"),
    )

    assert classify_failure(result) == "write_failure"
    assert decision.recovery_action is None
    assert decision.retry_allowed is False


def test_failed_diagnosis_recommends_explanation():
    result = ActionResult(
        action="fix_error",
        success=False,
        message="Could not inspect failure.",
    )

    decision = recover_from_failure(
        result,
        CodingState(task="Repair code"),
    )

    assert classify_failure(result) == "diagnostic_failure"
    assert decision.recovery_action == "explain_error"


def test_failed_repair_proposal_requires_human_input():
    result = ActionResult(
        action="propose_repair",
        success=False,
        message="Could not generate proposal.",
    )

    decision = recover_from_failure(
        result,
        CodingState(task="Repair code"),
    )

    assert classify_failure(result) == "repair_proposal_failure"
    assert decision.recovery_action is None
    assert decision.requires_human_input is True
    assert decision.retry_allowed is False


def test_user_input_boundary_blocks_recovery():
    result = ActionResult(
        action="run_tests",
        success=False,
        message="Tests failed.",
    )

    state = CodingState(task="Repair code")
    state.user_input_needed = True

    decision = recover_from_failure(result, state)

    assert decision.recovery_action is None
    assert decision.requires_human_input is True
    assert decision.retry_allowed is False


def test_unknown_failure_stops_automatic_recovery():
    result = ActionResult(
        action="some_unknown_action",
        success=False,
        message="Unexpected failure.",
    )

    decision = recover_from_failure(
        result,
        CodingState(task="Unknown"),
    )

    assert classify_failure(result) == "unknown_failure"
    assert decision.recovery_action is None
    assert decision.retry_allowed is False


def test_recovery_does_not_mutate_state():
    result = ActionResult(
        action="run_tests",
        success=False,
        message="Tests failed.",
    )

    state = CodingState(task="Run tests")
    before = state.__dict__.copy()

    recover_from_failure(result, state)

    assert state.__dict__ == before


def test_recovery_is_advisory_only():
    result = ActionResult(
        action="run_tests",
        success=False,
        message="Tests failed.",
    )

    decision = recover_from_failure(
        result,
        CodingState(task="Run tests"),
    )

    assert decision.recovery_action == "fix_error"

    # A recovery decision only recommends an action.
    # It does not execute anything.
    assert isinstance(decision.recovery_action, str)