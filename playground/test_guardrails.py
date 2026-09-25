from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.guardrails import (
    GuardrailDecision,
    check_action_safety,
    check_approval_boundary,
    check_guardrails,
    check_state_consistency,
)


def test_guardrails_return_decision():
    result = ActionResult(
        action="inspect_files",
        success=True,
        message="Files inspected.",
    )

    decision = check_guardrails(
        result,
        CodingState(task="Inspect workspace"),
    )

    assert isinstance(decision, GuardrailDecision)
    assert decision.action == "inspect_files"


def test_normal_read_action_is_allowed():
    result = ActionResult(
        action="read_file",
        success=True,
        message="File read.",
    )

    state = CodingState(task="Read file")
    state.current_file = "example.py"

    decision = check_guardrails(result, state)

    assert decision.allowed is True
    assert decision.violations == []


def test_human_input_blocks_execution():
    result = ActionResult(
        action="read_file",
        success=True,
        message="File read.",
    )

    state = CodingState(task="Read file")
    state.user_input_needed = True

    decision = check_guardrails(result, state)

    assert decision.allowed is False
    assert decision.requires_human_approval is True


def test_approve_repair_is_always_blocked_by_guardrail():
    result = ActionResult(
        action="approve_repair",
        success=True,
        message="Repair approved.",
    )

    state = CodingState(task="Repair code")
    state.current_file = "example.py"
    state.repair_proposed = True

    decision = check_guardrails(result, state)

    assert decision.allowed is False
    assert decision.requires_human_approval is True


def test_approve_repair_requires_existing_proposal():
    result = ActionResult(
        action="approve_repair",
        success=False,
        message="Approval requested.",
    )

    state = CodingState(task="Repair code")

    decision = check_guardrails(result, state)

    assert decision.allowed is False
    assert any(
        "proposal exists" in violation
        for violation in decision.violations
    )


def test_repair_action_requires_current_file():
    result = ActionResult(
        action="propose_repair",
        success=True,
        message="Proposal generated.",
    )

    state = CodingState(task="Repair code")

    decision = check_guardrails(result, state)

    assert decision.allowed is False
    assert any(
        "current file" in violation
        for violation in decision.violations
    )


def test_write_code_requires_file_content():
    result = ActionResult(
        action="write_code",
        success=True,
        message="Code written.",
    )

    state = CodingState(task="Modify code")
    state.current_file = "example.py"

    decision = check_guardrails(result, state)

    assert decision.allowed is False
    assert any(
        "file content" in violation
        for violation in decision.violations
    )


def test_write_code_with_context_is_allowed():
    result = ActionResult(
        action="write_code",
        success=True,
        message="Code written.",
    )

    state = CodingState(task="Modify code")
    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.repair_proposed = True
    state.repair_approved = True

    decision = check_guardrails(result, state)

    assert decision.allowed is True
    assert decision.violations == []


def test_already_applied_repair_cannot_be_approved():
    result = ActionResult(
        action="approve_repair",
        success=True,
        message="Repair approved.",
    )

    state = CodingState(task="Repair code")
    state.current_file = "example.py"
    state.repair_proposed = True
    state.repair_approved = True
    state.repair_applied = True

    decision = check_guardrails(result, state)

    assert decision.allowed is False
    assert any(
        "already been applied" in violation
        for violation in decision.violations
    )


def test_success_does_not_bypass_repair_safety():
    result = ActionResult(
        action="propose_repair",
        success=True,
        message="Proposal generated.",
    )

    state = CodingState(task="Repair code")
    state.current_file = "example.py"

    decision = check_guardrails(result, state)

    assert decision.allowed is False


def test_unknown_action_is_blocked():
    violations, warnings = check_action_safety(
        "destroy_everything",
        CodingState(task="Unknown"),
    )

    assert violations
    assert any(
        "Unknown action" in violation
        for violation in violations
    )


def test_fix_error_requires_error():
    violations, warnings = check_action_safety(
        "fix_error",
        CodingState(task="Repair code"),
    )

    assert any(
        "recorded error" in violation
        for violation in violations
    )


def test_fix_error_is_allowed_with_error():
    state = CodingState(task="Repair code")
    state.last_error = "Test failed."

    violations, warnings = check_action_safety(
        "fix_error",
        state,
    )

    assert violations == []


def test_write_code_warns_when_error_not_inspected():
    state = CodingState(task="Modify code")
    state.current_file = "example.py"
    state.current_file_content = "print('hello')"

    violations, warnings = check_action_safety(
        "write_code",
        state,
    )

    assert violations == []
    assert warnings


def test_write_code_is_clean_after_error_inspection():
    state = CodingState(task="Modify code")
    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.error_inspected = True

    violations, warnings = check_action_safety(
        "write_code",
        state,
    )

    assert violations == []


def test_propose_repair_requires_error_inspection():
    state = CodingState(task="Repair code")
    state.current_file = "example.py"
    state.current_file_content = "print('hello')"

    violations, warnings = check_action_safety(
        "propose_repair",
        state,
    )

    assert any(
        "error to be inspected" in violation
        for violation in violations
    )


def test_propose_repair_is_allowed_after_analysis():
    state = CodingState(task="Repair code")
    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.error_inspected = True

    violations, warnings = check_action_safety(
        "propose_repair",
        state,
    )

    assert violations == []
    assert warnings


def test_duplicate_repair_proposal_is_blocked():
    state = CodingState(task="Repair code")
    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.error_inspected = True
    state.repair_proposed = True

    violations, warnings = check_action_safety(
        "propose_repair",
        state,
    )

    assert any(
        "already exists" in violation
        for violation in violations
    )


def test_rollback_requires_applied_repair():
    state = CodingState(task="Rollback")

    violations, warnings = check_action_safety(
        "rollback_repair",
        state,
    )

    assert any(
        "applied repair" in violation
        for violation in violations
    )


def test_rollback_allowed_after_applied_repair():
    state = CodingState(task="Rollback")
    state.repair_proposed = True
    state.repair_approved = True
    state.repair_applied = True

    violations, warnings = check_action_safety(
        "rollback_repair",
        state,
    )

    assert violations == []
    assert warnings


def test_read_only_actions_are_safe_without_repair_context():
    state = CodingState(task="Inspect")

    for action in (
        "inspect_files",
        "run_tests",
        "explain_error",
        "review_code",
        "improve_code",
    ):
        violations, warnings = check_action_safety(
            action,
            state,
        )

        assert violations == []


def test_guardrail_does_not_mutate_state():
    state = CodingState(task="Repair code")
    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.repair_proposed = True
    state.repair_approved = True

    result = ActionResult(
        action="write_code",
        success=True,
        message="Code written.",
    )

    before = state.__dict__.copy()

    check_guardrails(result, state)

    assert state.__dict__ == before


# ==========================================================
# Phase 9.6.3 — State Consistency
# ==========================================================


def test_consistent_default_state_has_no_violations():
    state = CodingState(task="Inspect workspace")

    violations, warnings = check_state_consistency(state)

    assert violations == []


def test_approved_repair_requires_proposal():
    state = CodingState(task="Repair")
    state.repair_approved = True

    violations, warnings = check_state_consistency(state)

    assert any(
        "repair_approved" in violation
        for violation in violations
    )


def test_applied_repair_requires_proposal():
    state = CodingState(task="Repair")
    state.repair_applied = True

    violations, warnings = check_state_consistency(state)

    assert any(
        "repair_applied" in violation
        for violation in violations
    )


def test_applied_repair_requires_approval():
    state = CodingState(task="Repair")
    state.repair_proposed = True
    state.repair_applied = True

    violations, warnings = check_state_consistency(state)

    assert any(
        "repair_approved" in violation
        for violation in violations
    )


def test_repair_proposal_can_warn_about_missing_description():
    state = CodingState(task="Repair")
    state.current_file = "example.py"
    state.repair_proposed = True
    state.repair_description = None
    state.proposed_content = None

    violations, warnings = check_state_consistency(state)

    assert violations == []
    assert len(warnings) >= 2


def test_repair_applied_can_warn_about_missing_original_content():
    state = CodingState(task="Repair")
    state.current_file = "example.py"
    state.repair_proposed = True
    state.repair_approved = True
    state.repair_applied = True

    violations, warnings = check_state_consistency(state)

    assert violations == []
    assert any(
        "original content" in warning
        for warning in warnings
    )


def test_file_content_requires_current_file():
    state = CodingState(task="Read")

    state.current_file_content = "print('hello')"

    violations, warnings = check_state_consistency(state)

    assert any(
        "current_file_content" in violation
        for violation in violations
    )


def test_repair_proposal_requires_current_file():
    state = CodingState(task="Repair")
    state.repair_proposed = True

    violations, warnings = check_state_consistency(state)

    assert any(
        "current file" in violation
        for violation in violations
    )


def test_error_inspected_without_error_is_warning():
    state = CodingState(task="Repair")
    state.error_inspected = True
    state.last_error = None

    violations, warnings = check_state_consistency(state)

    assert violations == []
    assert any(
        "error_inspected" in warning
        for warning in warnings
    )


def test_tests_passed_requires_tests_run():
    state = CodingState(task="Testing")
    state.tests_passed = True
    state.tests_run = False

    violations, warnings = check_state_consistency(state)

    assert any(
        "tests_passed" in violation
        for violation in violations
    )


def test_valid_repair_state_is_consistent():
    state = CodingState(task="Repair")

    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.last_error = "Test failed."
    state.error_inspected = True

    state.repair_proposed = True
    state.repair_description = "Fix failing assertion."
    state.proposed_content = "print('fixed')"

    state.repair_approved = True
    state.repair_applied = True
    state.repair_original_content = "print('hello')"

    violations, warnings = check_state_consistency(state)

    assert violations == []


def test_inconsistent_state_blocks_guardrails():
    state = CodingState(task="Repair")
    state.repair_applied = True

    result = ActionResult(
        action="run_tests",
        success=True,
        message="Tests completed.",
    )

    decision = check_guardrails(result, state)

    assert decision.allowed is False
    assert decision.violations


def test_consistency_check_does_not_mutate_state():
    state = CodingState(task="Repair")
    state.repair_proposed = True
    state.current_file = "example.py"

    before = state.__dict__.copy()

    check_state_consistency(state)

    assert state.__dict__ == before


# ==========================================================
# Phase 9.6.4 — Approval Boundary
# ==========================================================


def test_read_only_action_does_not_require_approval():
    state = CodingState(task="Inspect")

    violations, warnings, required = check_approval_boundary(
        "inspect_files",
        state,
    )

    assert violations == []
    assert required is False


def test_propose_repair_does_not_require_approval_to_create_proposal():
    state = CodingState(task="Repair")

    violations, warnings, required = check_approval_boundary(
        "propose_repair",
        state,
    )

    assert violations == []
    assert required is False
    assert warnings


def test_propose_repair_warning_does_not_grant_permission():
    state = CodingState(task="Repair")

    violations, warnings, required = check_approval_boundary(
        "propose_repair",
        state,
    )

    assert required is False
    assert any(
        "does not authorize" in warning
        for warning in warnings
    )


def test_approve_repair_requires_human_approval():
    state = CodingState(task="Repair")
    state.repair_proposed = True

    violations, warnings, required = check_approval_boundary(
        "approve_repair",
        state,
    )

    assert violations
    assert required is True
    assert any(
        "explicit human authorization" in violation
        for violation in violations
    )


def test_user_input_boundary_requires_human_approval():
    state = CodingState(task="Repair")
    state.user_input_needed = True

    violations, warnings, required = check_approval_boundary(
        "read_file",
        state,
    )

    assert violations
    assert required is True


def test_write_code_requires_approved_repair():
    state = CodingState(task="Repair")
    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.repair_proposed = True

    violations, warnings, required = check_approval_boundary(
        "write_code",
        state,
    )

    assert violations
    assert required is True
    assert any(
        "explicitly approved repair" in violation
        for violation in violations
    )


def test_write_code_can_pass_approval_boundary_after_approval():
    state = CodingState(task="Repair")
    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.repair_proposed = True
    state.repair_approved = True

    violations, warnings, required = check_approval_boundary(
        "write_code",
        state,
    )

    assert violations == []
    assert required is False


def test_rollback_requires_approved_repair():
    state = CodingState(task="Rollback")
    state.repair_proposed = True
    state.repair_applied = True

    violations, warnings, required = check_approval_boundary(
        "rollback_repair",
        state,
    )

    assert violations
    assert required is True


def test_rollback_can_pass_after_approval():
    state = CodingState(task="Rollback")
    state.repair_proposed = True
    state.repair_approved = True
    state.repair_applied = True

    violations, warnings, required = check_approval_boundary(
        "rollback_repair",
        state,
    )

    assert violations == []
    assert required is False


def test_high_confidence_cannot_be_used_as_approval():
    state = CodingState(task="Repair")
    state.repair_proposed = True

    # Confidence fields do not exist as approval state.
    # Approval remains explicitly represented by repair_approved.
    violations, warnings, required = check_approval_boundary(
        "approve_repair",
        state,
    )

    assert required is True
    assert state.repair_approved is False


def test_approval_boundary_does_not_mutate_state():
    state = CodingState(task="Repair")
    state.repair_proposed = True
    state.current_file = "example.py"

    before = state.__dict__.copy()

    check_approval_boundary(
        "approve_repair",
        state,
    )

    assert state.__dict__ == before


def test_guardrails_expose_approval_requirement():
    state = CodingState(task="Repair")
    state.current_file = "example.py"
    state.repair_proposed = True

    result = ActionResult(
        action="approve_repair",
        success=True,
        message="Approval requested.",
    )

    decision = check_guardrails(result, state)

    assert decision.allowed is False
    assert decision.requires_human_approval is True


def test_guardrails_allow_normal_action_without_approval():
    state = CodingState(task="Inspect")

    result = ActionResult(
        action="inspect_files",
        success=True,
        message="Files inspected.",
    )

    decision = check_guardrails(result, state)

    assert decision.allowed is True
    assert decision.requires_human_approval is False