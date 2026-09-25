from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.learning_feedback import (
    LearningOutcome,
    build_learning_outcome,
)


def test_learning_outcome_contains_basic_execution_result():
    state = CodingState(
        task="inspect the project"
    )

    state.task_intent = "inspect"

    result = ActionResult(
        action="inspect_files",
        success=True,
        message="Files inspected.",
    )

    outcome = build_learning_outcome(
        result,
        state,
    )

    assert isinstance(
        outcome,
        LearningOutcome,
    )

    assert outcome.task == state.task
    assert outcome.task_intent == "inspect"
    assert outcome.action == "inspect_files"
    assert outcome.success is True


def test_failed_action_is_recorded():
    state = CodingState(
        task="repair the code"
    )

    state.task_intent = "repair"

    result = ActionResult(
        action="fix_error",
        success=False,
        message="Unable to diagnose error.",
    )

    outcome = build_learning_outcome(
        result,
        state,
    )

    assert outcome.success is False
    assert outcome.error_present is False
    assert outcome.evidence
    assert outcome.lessons


def test_test_result_is_recorded():
    state = CodingState(
        task="run tests"
    )

    state.task_intent = "test"
    state.tests_run = True
    state.tests_passed = True

    result = ActionResult(
        action="run_tests",
        success=True,
        message="Tests passed.",
    )

    outcome = build_learning_outcome(
        result,
        state,
    )

    assert outcome.tests_run is True
    assert outcome.tests_passed is True

    assert any(
        "passed" in evidence.lower()
        for evidence in outcome.evidence
    )


def test_failed_tests_are_recorded():
    state = CodingState(
        task="run tests"
    )

    state.task_intent = "test"
    state.tests_run = True
    state.tests_passed = False

    result = ActionResult(
        action="run_tests",
        success=True,
        message="Tests completed with failures.",
    )

    outcome = build_learning_outcome(
        result,
        state,
    )

    assert outcome.tests_run is True
    assert outcome.tests_passed is False

    assert any(
        "did not pass" in evidence.lower()
        for evidence in outcome.evidence
    )


def test_repair_state_is_recorded():
    state = CodingState(
        task="repair the code"
    )

    state.task_intent = "repair"
    state.current_file = "example.py"

    state.repair_proposed = True
    state.repair_approved = True
    state.repair_applied = True

    result = ActionResult(
        action="write_code",
        success=True,
        message="Repair applied.",
    )

    outcome = build_learning_outcome(
        result,
        state,
    )

    assert outcome.repair_proposed is True
    assert outcome.repair_approved is True
    assert outcome.repair_applied is True


def test_guardrail_block_is_recorded():
    state = CodingState(
        task="repair the code"
    )

    state.task_intent = "repair"

    result = ActionResult(
        action="write_code",
        success=False,
        message="Execution blocked.",
        data={
            "guardrail": {
                "allowed": False,
                "requires_human_approval": True,
            }
        },
    )

    outcome = build_learning_outcome(
        result,
        state,
    )

    assert outcome.guardrail_blocked is True
    assert outcome.human_input_required is True

    assert any(
        "guardrail" in evidence.lower()
        for evidence in outcome.evidence
    )


def test_strategy_is_recorded():
    state = CodingState(
        task="repair the code"
    )

    result = ActionResult(
        action="fix_error",
        success=True,
        message="Error investigated.",
    )

    outcome = build_learning_outcome(
        result,
        state,
        strategy="repair",
    )

    assert outcome.strategy == "repair"


def test_confidence_is_recorded():
    state = CodingState(
        task="repair the code"
    )

    result = ActionResult(
        action="fix_error",
        success=True,
        message="Error investigated.",
    )

    outcome = build_learning_outcome(
        result,
        state,
        confidence_score=0.85,
    )

    assert outcome.confidence_score == 0.85


def test_recovery_action_is_recorded():
    state = CodingState(
        task="repair the code"
    )

    result = ActionResult(
        action="fix_error",
        success=False,
        message="Repair failed.",
    )

    outcome = build_learning_outcome(
        result,
        state,
        recovery_action="inspect_files",
    )

    assert outcome.recovery_action == "inspect_files"

    assert any(
        "inspect_files" in evidence
        for evidence in outcome.evidence
    )


def test_learning_feedback_does_not_mutate_state():
    state = CodingState(
        task="repair the code"
    )

    state.task_intent = "repair"
    state.current_file = "example.py"
    state.repair_proposed = True

    result = ActionResult(
        action="propose_repair",
        success=True,
        message="Proposal created.",
    )

    before = state.__dict__.copy()

    build_learning_outcome(
        result,
        state,
    )

    assert state.__dict__ == before
