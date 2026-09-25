from pathlib import Path

from flycoder.actions.registry import ActionResult
from flycoder.agent import FlyCoderAgent
from flycoder.state import CodingState
from flycoder.tools.guardrails import check_guardrails


def prepare_unapproved_repair_state() -> CodingState:
    """Create state that is ready for a proposed repair but not approved."""

    state = CodingState(
        task="repair the code",
    )

    state.files = ["example.py"]
    state.current_file = "example.py"
    state.current_file_content = "print('hello')"

    state.last_error = "Test failed."
    state.error_inspected = True

    state.repair_proposed = True
    state.repair_approved = False
    state.repair_applied = False

    # Prevent the normal decision engine from choosing run_tests.
    state.tests_run = True
    state.tests_passed = False

    return state


def test_unapproved_write_is_blocked_by_guardrail():
    state = prepare_unapproved_repair_state()

    result = ActionResult(
        action="write_code",
        success=True,
        message="Would write code.",
    )

    decision = check_guardrails(
        result,
        state,
    )

    assert decision.allowed is False
    assert decision.requires_human_approval is True

    assert any(
        "explicitly approved repair" in violation
        for violation in decision.violations
    )


def test_agent_does_not_execute_blocked_write():
    state = prepare_unapproved_repair_state()

    agent = FlyCoderAgent(Path("."))

    executed = False

    def forbidden_execute(*args, **kwargs):
        nonlocal executed

        executed = True

        return ActionResult(
            action="write_code",
            success=True,
            message="This action must never execute.",
        )

    agent.actions.execute = forbidden_execute

    # Force the agent to attempt write_code so that this test
    # exercises the actual pre-execution guardrail.
    agent.decide = lambda current_state: "write_code"

    result = agent.run_once(state)

    assert executed is False
    assert result.action == "write_code"
    assert result.success is False

    assert result.data is not None
    assert "guardrail" in result.data

    guardrail = result.data["guardrail"]

    assert guardrail["allowed"] is False
    assert guardrail["requires_human_approval"] is True


def test_blocked_write_contains_guardrail_reason():
    state = prepare_unapproved_repair_state()

    agent = FlyCoderAgent(Path("."))

    agent.decide = lambda current_state: "write_code"

    result = agent.run_once(state)

    assert result.action == "write_code"
    assert result.success is False

    assert "guardrail" in result.data

    guardrail = result.data["guardrail"]

    assert guardrail["allowed"] is False
    assert guardrail["reason"]

    assert any(
        "explicitly approved repair" in violation
        for violation in guardrail["violations"]
    )


def test_approved_write_can_cross_guardrail():
    state = prepare_unapproved_repair_state()

    state.repair_approved = True

    result = ActionResult(
        action="write_code",
        success=True,
        message="Would write code.",
    )

    decision = check_guardrails(
        result,
        state,
    )

    assert decision.allowed is True
    assert decision.requires_human_approval is False


def test_agent_executes_safe_action_after_guardrail():
    state = CodingState(
        task="inspect the project",
    )

    agent = FlyCoderAgent(Path("."))

    executed = False

    def safe_execute(*args, **kwargs):
        nonlocal executed

        executed = True

        return ActionResult(
            action="inspect_files",
            success=True,
            message="Inspection executed.",
            data={},
        )

    agent.actions.execute = safe_execute

    agent.decide = lambda current_state: "inspect_files"

    result = agent.run_once(state)

    assert executed is True
    assert result.action == "inspect_files"
    assert result.success is True

    assert "guardrail" in result.data
    assert result.data["guardrail"]["allowed"] is True


def test_user_input_boundary_blocks_agent_execution():
    state = CodingState(
        task="continue repair",
    )

    state.user_input_needed = True

    agent = FlyCoderAgent(Path("."))

    executed = False

    def forbidden_execute(*args, **kwargs):
        nonlocal executed

        executed = True

        return ActionResult(
            action="read_file",
            success=True,
            message="Should not execute.",
        )

    agent.actions.execute = forbidden_execute

    agent.decide = lambda current_state: "read_file"

    result = agent.run_once(state)

    assert executed is False
    assert result.action == "read_file"
    assert result.success is False

    assert "guardrail" in result.data
    assert result.data["guardrail"]["allowed"] is False
    assert (
        result.data["guardrail"]["requires_human_approval"]
        is True
    )


def test_guardrail_check_does_not_approve_repair():
    state = prepare_unapproved_repair_state()

    agent = FlyCoderAgent(Path("."))

    agent.decide = lambda current_state: "write_code"

    agent.run_once(state)

    assert state.repair_approved is False
    assert state.repair_applied is False


def test_guardrail_does_not_modify_repair_state():
    state = prepare_unapproved_repair_state()

    before = state.__dict__.copy()

    result = ActionResult(
        action="write_code",
        success=True,
        message="Guardrail check.",
    )

    check_guardrails(
        result,
        state,
    )

    assert state.__dict__ == before