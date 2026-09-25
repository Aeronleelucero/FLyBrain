"""Execution guardrails for FLY-CODER Phase 9.6."""

from __future__ import annotations

from dataclasses import dataclass, field

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState


@dataclass
class GuardrailDecision:
    """Describe whether an action is permitted by execution guardrails."""

    action: str
    allowed: bool
    reason: str = ""
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requires_human_approval: bool = False


REPAIR_SENSITIVE_ACTIONS = {
    "write_code",
    "propose_repair",
    "approve_repair",
    "rollback_repair",
}


READ_ONLY_ACTIONS = {
    "inspect_files",
    "read_file",
    "run_tests",
    "explain_error",
    "review_code",
    "improve_code",
}


KNOWN_ACTIONS = READ_ONLY_ACTIONS | REPAIR_SENSITIVE_ACTIONS | {
    "fix_error",
}


def check_state_consistency(
    state: CodingState,
) -> tuple[list[str], list[str]]:
    """Check CodingState for contradictory or incomplete state.

    Returns:
        A tuple containing state violations and non-blocking warnings.

    This function only validates state. It never mutates state.
    """

    violations: list[str] = []
    warnings: list[str] = []

    # ----------------------------------------------------------
    # Repair state relationships
    # ----------------------------------------------------------

    if state.repair_approved and not state.repair_proposed:
        violations.append(
            "repair_approved cannot be true without repair_proposed."
        )

    if state.repair_applied and not state.repair_proposed:
        violations.append(
            "repair_applied cannot be true without repair_proposed."
        )

    if state.repair_applied and not state.repair_approved:
        violations.append(
            "repair_applied cannot be true without repair_approved."
        )

    # ----------------------------------------------------------
    # Repair content relationships
    # ----------------------------------------------------------

    if state.repair_proposed and not state.repair_description:
        warnings.append(
            "A repair proposal exists without a repair description."
        )

    if state.repair_proposed and state.proposed_content is None:
        warnings.append(
            "A repair proposal exists without proposed content."
        )

    if state.repair_applied and state.repair_original_content is None:
        warnings.append(
            "An applied repair does not have original content recorded."
        )

    # ----------------------------------------------------------
    # File context relationships
    # ----------------------------------------------------------

    if state.current_file_content is not None and not state.current_file:
        violations.append(
            "current_file_content cannot exist without current_file."
        )

    if state.repair_proposed and not state.current_file:
        violations.append(
            "A repair proposal requires a current file."
        )

    # ----------------------------------------------------------
    # Error relationships
    # ----------------------------------------------------------

    if state.error_inspected and not state.last_error:
        warnings.append(
            "error_inspected is true but no last_error is recorded."
        )

    # ----------------------------------------------------------
    # Test relationships
    # ----------------------------------------------------------

    if state.tests_passed and not state.tests_run:
        violations.append(
            "tests_passed cannot be true when tests_run is false."
        )

    return violations, warnings


def check_action_safety(
    action: str,
    state: CodingState,
) -> tuple[list[str], list[str]]:
    """Return action-specific violations and warnings.

    This function only evaluates safety conditions. It never executes
    an action and never grants repair approval.
    """

    violations: list[str] = []
    warnings: list[str] = []

    if action not in KNOWN_ACTIONS:
        violations.append(
            f"Unknown action '{action}' cannot pass the execution guardrail."
        )
        return violations, warnings

    if action in READ_ONLY_ACTIONS:
        return violations, warnings

    if action == "fix_error":
        if not state.last_error:
            violations.append(
                "fix_error requires a recorded error to investigate."
            )

        return violations, warnings

    if action == "write_code":
        if not state.current_file:
            violations.append(
                "write_code requires a current file."
            )

        if state.current_file_content is None:
            violations.append(
                "write_code requires current file content."
            )

        if not state.error_inspected:
            warnings.append(
                "The current error has not been explicitly inspected."
            )

        return violations, warnings

    if action == "propose_repair":
        if not state.current_file:
            violations.append(
                "propose_repair requires a current file."
            )

        if state.current_file_content is None:
            violations.append(
                "propose_repair requires current file content."
            )

        if not state.error_inspected:
            violations.append(
                "propose_repair requires the error to be inspected first."
            )

        if state.repair_proposed:
            violations.append(
                "A repair proposal already exists."
            )

        warnings.append(
            "A repair proposal must remain subject to explicit approval."
        )

        return violations, warnings

    if action == "approve_repair":
        violations.append(
            "approve_repair is always controlled by explicit human approval."
        )

        if not state.repair_proposed:
            violations.append(
                "A repair cannot be approved before a proposal exists."
            )

        if state.repair_applied:
            violations.append(
                "The repair has already been applied."
            )

        return violations, warnings

    if action == "rollback_repair":
        if not state.repair_applied:
            violations.append(
                "rollback_repair requires an applied repair."
            )

        warnings.append(
            "Rollback changes workspace state and must remain safety-controlled."
        )

        return violations, warnings

    return violations, warnings


def check_approval_boundary(
    action: str,
    state: CodingState,
) -> tuple[list[str], list[str], bool]:
    """Check whether the action crosses the human approval boundary.

    Returns:
        A tuple containing:
            - approval violations,
            - approval warnings,
            - whether human approval is required.

    This function never approves a repair and never executes an action.
    """

    violations: list[str] = []
    warnings: list[str] = []
    requires_human_approval = False

    # ----------------------------------------------------------
    # Explicit user-input boundary
    # ----------------------------------------------------------

    if state.user_input_needed:
        violations.append(
            "The workflow is waiting for explicit human input or approval."
        )
        requires_human_approval = True

    # ----------------------------------------------------------
    # Repair proposal boundary
    # ----------------------------------------------------------

    if action == "propose_repair":
        warnings.append(
            "Generating a repair proposal does not authorize applying it."
        )

    # ----------------------------------------------------------
    # Repair approval boundary
    # ----------------------------------------------------------

    if action == "approve_repair":
        violations.append(
            "approve_repair requires explicit human authorization."
        )
        requires_human_approval = True

    # ----------------------------------------------------------
    # Repair-sensitive actions
    # ----------------------------------------------------------

    if action in {
        "write_code",
        "rollback_repair",
    }:
        if not state.repair_approved:
            violations.append(
                f"{action} requires an explicitly approved repair."
            )
            requires_human_approval = True

    return violations, warnings, requires_human_approval


def check_guardrails(
    result: ActionResult,
    state: CodingState,
) -> GuardrailDecision:
    """Check whether the current action satisfies execution guardrails.

    Guardrails are deterministic safety checks. They do not execute actions,
    choose actions, or grant repair approval.
    """

    action = result.action

    violations: list[str] = []
    warnings: list[str] = []

    # ----------------------------------------------------------
    # Guardrail 1: action must be known
    # ----------------------------------------------------------

    if not action:
        violations.append("The action name is empty.")
    else:
        action_violations, action_warnings = check_action_safety(
            action,
            state,
        )

        violations.extend(action_violations)
        warnings.extend(action_warnings)

    # ----------------------------------------------------------
    # Guardrail 2: state must be internally consistent
    # ----------------------------------------------------------

    state_violations, state_warnings = check_state_consistency(
        state,
    )

    violations.extend(state_violations)
    warnings.extend(state_warnings)

    # ----------------------------------------------------------
    # Guardrail 3: approval boundary
    # ----------------------------------------------------------

    approval_violations, approval_warnings, approval_required = (
        check_approval_boundary(
            action,
            state,
        )
    )

    violations.extend(approval_violations)
    warnings.extend(approval_warnings)

    # ----------------------------------------------------------
    # Guardrail 4: successful execution does not bypass safety
    # ----------------------------------------------------------

    if result.success and action in REPAIR_SENSITIVE_ACTIONS:
        warnings.append(
            "The action reported success, but normal repair safety "
            "controls still apply."
        )

    allowed = not violations

    if allowed:
        reason = (
            "The action satisfies the current execution guardrails. "
            "Execution permission remains separate from confidence and "
            "strategy selection."
        )
    else:
        reason = (
            "The action is blocked because one or more execution "
            "guardrails were violated."
        )

    return GuardrailDecision(
        action=action,
        allowed=allowed,
        reason=reason,
        violations=violations,
        warnings=warnings,
        requires_human_approval=approval_required,
    )