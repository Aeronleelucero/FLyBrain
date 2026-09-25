"""Failure recovery tools for FLY-CODER Phase 9.4."""

from __future__ import annotations

from dataclasses import dataclass, field

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState


@dataclass
class RecoveryDecision:
    """Describe how the agent should recover from an action result."""

    action: str
    failed: bool
    recovery_action: str | None = None
    reason: str = ""
    risks: list[str] = field(default_factory=list)
    requires_human_input: bool = False
    retry_allowed: bool = False


def classify_failure(result: ActionResult) -> str:
    """Classify a failed action into a small set of recovery categories."""

    if result.success:
        return "success"

    action = result.action

    if action == "run_tests":
        return "test_failure"

    if action == "read_file":
        return "read_failure"

    if action == "write_code":
        return "write_failure"

    if action == "fix_error":
        return "diagnostic_failure"

    if action == "propose_repair":
        return "repair_proposal_failure"

    return "unknown_failure"


def recover_from_failure(
    result: ActionResult,
    state: CodingState,
) -> RecoveryDecision:
    """Determine a safe recovery recommendation.

    This function is advisory only. It never executes an action and
    never grants permission to modify code or approve a repair.
    """

    if result.success:
        return RecoveryDecision(
            action=result.action,
            failed=False,
            reason="The action succeeded, so failure recovery is not required.",
        )

    if state.user_input_needed:
        return RecoveryDecision(
            action=result.action,
            failed=True,
            reason=(
                "The workflow requires user input or approval. "
                "Failure recovery must stop at the human boundary."
            ),
            risks=[
                "Do not bypass explicit user approval.",
                "Do not automatically execute approve_repair.",
            ],
            requires_human_input=True,
        )

    failure_type = classify_failure(result)

    if failure_type == "test_failure":
        return RecoveryDecision(
            action=result.action,
            failed=True,
            recovery_action="fix_error",
            reason=(
                "The test suite failed. Inspect the failure before "
                "attempting a repair."
            ),
            risks=[
                "The failure may be caused by the test environment.",
                "The root cause should be identified before modifying code.",
            ],
            retry_allowed=False,
        )

    if failure_type == "read_failure":
        return RecoveryDecision(
            action=result.action,
            failed=True,
            recovery_action="inspect_files",
            reason=(
                "Reading the selected file failed. Reinspect the workspace "
                "before attempting another file read."
            ),
            risks=[
                "The selected path may be invalid or unavailable.",
                "Do not repeatedly retry an unresolved path.",
            ],
            retry_allowed=False,
        )

    if failure_type == "write_failure":
        return RecoveryDecision(
            action=result.action,
            failed=True,
            recovery_action=None,
            reason=(
                "Writing code failed. The cause must be understood before "
                "attempting another modification."
            ),
            risks=[
                "The workspace may be partially affected.",
                "Blindly retrying a write could create additional changes.",
            ],
            retry_allowed=False,
        )

    if failure_type == "diagnostic_failure":
        return RecoveryDecision(
            action=result.action,
            failed=True,
            recovery_action="explain_error",
            reason=(
                "Error diagnosis failed. Gather a clearer explanation "
                "before continuing with repair."
            ),
            risks=[
                "The underlying failure may not be understood yet.",
            ],
            retry_allowed=False,
        )

    if failure_type == "repair_proposal_failure":
        return RecoveryDecision(
            action=result.action,
            failed=True,
            recovery_action=None,
            reason=(
                "The repair proposal could not be generated. "
                "Stop rather than attempting an unreviewed modification."
            ),
            risks=[
                "No safe repair proposal is currently available.",
                "Human investigation may be required.",
            ],
            requires_human_input=True,
            retry_allowed=False,
        )

    return RecoveryDecision(
        action=result.action,
        failed=True,
        recovery_action=None,
        reason=(
            "The failure type is unknown. Stop automatic recovery until "
            "the failure can be understood."
        ),
        risks=[
            "Unknown failures must not trigger blind retries.",
        ],
        retry_allowed=False,
    )