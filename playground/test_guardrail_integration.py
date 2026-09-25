from flycoder.actions.registry import ActionRegistry, ActionResult
from flycoder.state import CodingState
from flycoder.tools.guardrails import check_guardrails


def test_unapproved_write_code_is_blocked():
    state = CodingState(task="Repair code")

    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.repair_proposed = True
    state.repair_approved = False

    result = ActionResult(
        action="write_code",
        success=True,
        message="Would write code.",
    )

    decision = check_guardrails(result, state)

    assert decision.allowed is False
    assert decision.requires_human_approval is True


def test_approved_write_code_passes_guardrails():
    state = CodingState(task="Repair code")

    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.repair_proposed = True
    state.repair_approved = True

    result = ActionResult(
        action="write_code",
        success=True,
        message="Would write code.",
    )

    decision = check_guardrails(result, state)

    assert decision.allowed is True


def test_user_input_boundary_blocks_action():
    state = CodingState(task="Continue")

    state.user_input_needed = True

    result = ActionResult(
        action="read_file",
        success=True,
        message="Would read file.",
    )

    decision = check_guardrails(result, state)

    assert decision.allowed is False
    assert decision.requires_human_approval is True


def test_guardrail_decision_can_be_recorded_on_successful_action():
    state = CodingState(task="Inspect")

    result = ActionResult(
        action="inspect_files",
        success=True,
        message="Files inspected.",
        data={},
    )

    decision = check_guardrails(result, state)

    assert decision.allowed is True

    result.data["guardrail"] = decision

    assert "guardrail" in result.data
    assert result.data["guardrail"].allowed is True


def test_blocked_guardrail_does_not_execute_action():
    executed = False

    def dangerous_action(**kwargs):
        nonlocal executed
        executed = True

        return ActionResult(
            action="write_code",
            success=True,
            message="Dangerous action executed.",
        )

    registry = ActionRegistry()
    registry.register(
        "write_code",
        dangerous_action,
    )

    state = CodingState(task="Repair")

    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.repair_proposed = True
    state.repair_approved = False

    preflight = ActionResult(
        action="write_code",
        success=True,
        message="Pre-execution guardrail check.",
    )

    decision = check_guardrails(
        preflight,
        state,
    )

    assert decision.allowed is False

    if decision.allowed:
        registry.execute(
            "write_code",
            state=state,
        )

    assert executed is False


def test_read_only_action_can_pass_execution_guardrail():
    state = CodingState(task="Inspect")

    result = ActionResult(
        action="inspect_files",
        success=True,
        message="Inspect files.",
    )

    decision = check_guardrails(result, state)

    assert decision.allowed is True


def test_confidence_does_not_override_guardrails():
    state = CodingState(task="Repair")

    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.repair_proposed = True
    state.repair_approved = False

    result = ActionResult(
        action="write_code",
        success=True,
        message="High-confidence write.",
    )

    decision = check_guardrails(result, state)

    assert decision.allowed is False
    assert state.repair_approved is False


def test_strategy_does_not_override_guardrails():
    state = CodingState(task="Repair")

    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.repair_proposed = True
    state.repair_approved = False

    selected_strategy = "repair"

    result = ActionResult(
        action="write_code",
        success=True,
        message=f"Strategy selected: {selected_strategy}",
    )

    decision = check_guardrails(result, state)

    assert decision.allowed is False


def test_recovery_does_not_override_guardrails():
    state = CodingState(task="Repair")

    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.repair_proposed = True
    state.repair_approved = False

    result = ActionResult(
        action="write_code",
        success=False,
        message="Previous recovery attempted.",
    )

    decision = check_guardrails(result, state)

    assert decision.allowed is False


def test_guardrail_check_does_not_modify_state():
    state = CodingState(task="Repair")

    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.repair_proposed = True

    before = state.__dict__.copy()

    result = ActionResult(
        action="write_code",
        success=True,
        message="Check only.",
    )

    check_guardrails(result, state)

    assert state.__dict__ == before
