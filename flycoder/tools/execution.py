"""Adaptive execution observation tools for FLY-CODER Phase 9.3."""

from __future__ import annotations

from dataclasses import dataclass, field

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState


@dataclass
class ExecutionObservation:
    """Describe the result of one executed action."""

    action: str
    success: bool
    message: str
    next_action: str | None = None
    reason: str = ""
    risks: list[str] = field(default_factory=list)
    requires_human_input: bool = False


def observe_action(
    result: ActionResult,
    state: CodingState,
) -> ExecutionObservation:
    """Observe an action result and recommend a safe next action.

    This function is advisory only. It never executes an action and
    never grants approval for a repair.
    """

    action = result.action

    # ----------------------------------------------------------
    # Human approval/input boundary
    # ----------------------------------------------------------

    if state.user_input_needed:
        return ExecutionObservation(
            action=action,
            success=result.success,
            message=result.message,
            next_action=None,
            reason=(
                "The current workflow requires user input or approval "
                "before execution can continue."
            ),
            risks=[
                "Adaptive execution must not bypass the human approval boundary."
            ],
            requires_human_input=True,
        )

    # ----------------------------------------------------------
    # Test failure
    # ----------------------------------------------------------

    if action == "run_tests" and not result.success:
        return ExecutionObservation(
            action=action,
            success=False,
            message=result.message,
            next_action="fix_error",
            reason=(
                "The test run failed, so the next safe step is to "
                "inspect the failure before modifying code."
            ),
            risks=[
                "The root cause should be confirmed before proposing a repair."
            ],
        )

    # ----------------------------------------------------------
    # Test success
    # ----------------------------------------------------------

    if action == "run_tests" and result.success:
        return ExecutionObservation(
            action=action,
            success=True,
            message=result.message,
            next_action=None,
            reason=(
                "The test run completed successfully. The existing "
                "decision engine can determine whether the task is complete."
            ),
        )

    # ----------------------------------------------------------
    # Error inspection
    # ----------------------------------------------------------

    if action == "fix_error":
        if state.error_inspected and state.current_file:
            return ExecutionObservation(
                action=action,
                success=result.success,
                message=result.message,
                next_action="read_file",
                reason=(
                    "The failure has been inspected and a relevant file "
                    "was identified. Read it before proposing a repair."
                ),
                risks=[
                    "The source should be understood before modification."
                ],
            )

        return ExecutionObservation(
            action=action,
            success=result.success,
            message=result.message,
            next_action=None,
            reason=(
                "The error inspection did not identify enough information "
                "for a safe automatic next step."
            ),
            risks=[
                "Manual investigation may be required."
            ],
        )

    # ----------------------------------------------------------
    # File read after a repair failure
    # ----------------------------------------------------------

    if action == "read_file":
        if (
            state.task_intent == "repair"
            and state.error_inspected
            and state.current_file_content is not None
            and not state.repair_proposed
        ):
            return ExecutionObservation(
                action=action,
                success=result.success,
                message=result.message,
                next_action="propose_repair",
                reason=(
                    "The failing file is now available, so a repair "
                    "proposal can be generated without modifying the file."
                ),
                risks=[
                    "The repair proposal must remain subject to explicit approval."
                ],
            )

        return ExecutionObservation(
            action=action,
            success=result.success,
            message=result.message,
            next_action=None,
            reason=(
                "The file was read. The existing decision engine should "
                "determine the next task-specific action."
            ),
        )

    # ----------------------------------------------------------
    # Repair proposal
    # ----------------------------------------------------------

    if action == "propose_repair":
        return ExecutionObservation(
            action=action,
            success=result.success,
            message=result.message,
            next_action=None,
            reason=(
                "A repair proposal has reached the approval boundary. "
                "Execution must stop until the user explicitly approves it."
            ),
            risks=[
                "Never automatically execute approve_repair.",
                "Never treat confidence or learned experience as permission."
            ],
            requires_human_input=True,
        )

    # ----------------------------------------------------------
    # Code modification
    # ----------------------------------------------------------

    if action == "write_code":
        if result.success:
            return ExecutionObservation(
                action=action,
                success=True,
                message=result.message,
                next_action="run_tests",
                reason=(
                    "Code was modified successfully, so the next step "
                    "is verification through tests."
                ),
                risks=[
                    "The modification must be verified before completion."
                ],
            )

        return ExecutionObservation(
            action=action,
            success=False,
            message=result.message,
            next_action=None,
            reason=(
                "The code modification failed, so automatic continuation "
                "should stop until the failure is understood."
            ),
            risks=[
                "Do not assume the workspace was modified successfully."
            ],
        )

    # ----------------------------------------------------------
    # Default
    # ----------------------------------------------------------

    return ExecutionObservation(
        action=action,
        success=result.success,
        message=result.message,
        next_action=None,
        reason=(
            "No specialized adaptive rule applies. The existing "
            "decision engine remains responsible for the next action."
        ),
    )
