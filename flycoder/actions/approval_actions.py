"""Repair approval actions for FLY-CODER."""

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace
from flycoder.tools.freshness import record_code_change
from flycoder.tools.human_input import (
    HUMAN_INPUT_STALE_PROPOSAL,
    clear_human_input,
    request_human_input,
)
from flycoder.tools.proposals import approval_problems


def _reject_as_stale(
    state: CodingState,
    message: str,
    problems: list[str],
) -> ActionResult:
    """Invalidate the approval and stop for a new human decision.

    Only the approval is invalidated. Whether a repair has already been
    applied is a fact about the workspace and is never reset here.
    """

    state.repair_approval = None
    state.repair_approved = False
    request_human_input(state, HUMAN_INPUT_STALE_PROPOSAL)

    return ActionResult(
        action="approve_repair",
        success=False,
        message=message,
        data={
            "file": state.current_file,
            "stale_proposal": True,
            "requires_new_proposal": True,
            "problems": problems,
        },
    )


def approve_repair_action(
    workspace: Workspace,
    state: CodingState,
) -> ActionResult:
    """Apply a repair proposal that a human explicitly approved.

    Phase 10.7: the approval must be an ``ApprovalRecord`` bound to the
    exact current proposal (fingerprint of file, original, and proposed
    content) and the current code version, and the file on disk must
    still contain the exact original content. Any mismatch invalidates
    the approval and requires a new human decision. Applying a repair is
    not verification; the change must be verified afterwards.
    """

    if state.repair_applied:
        # Defensive: an applied repair is never re-applied, and its
        # proposal, approval, and applied state are left untouched.
        return ActionResult(
            action="approve_repair",
            success=False,
            message=(
                "The repair has already been applied; it cannot be approved "
                "or applied again."
            ),
            data={"file": state.current_file, "already_applied": True},
        )

    if not state.repair_proposed:
        return ActionResult(
            action="approve_repair",
            success=False,
            message="No repair proposal is available.",
        )

    if not state.current_file:
        return ActionResult(
            action="approve_repair",
            success=False,
            message="No file is selected for repair.",
        )

    if state.proposed_content is None:
        return ActionResult(
            action="approve_repair",
            success=False,
            message="No proposed content is available.",
        )

    problems = approval_problems(state)

    if problems:
        if state.repair_approval is None:
            # No human approval at all: nothing to invalidate.
            return ActionResult(
                action="approve_repair",
                success=False,
                message=(
                    "No explicit human approval of this exact proposal is "
                    "recorded."
                ),
                data={"file": state.current_file, "problems": problems},
            )

        return _reject_as_stale(
            state,
            "Repair approval rejected because the approved proposal no "
            "longer matches the current proposal or code version. A new "
            "review is required.",
            problems,
        )

    # Re-read the file before applying the proposal. This prevents an old
    # proposal from overwriting changes made after the proposal was created.
    current_content = workspace.read_file(state.current_file)

    if current_content != state.proposal_original_content:
        return _reject_as_stale(
            state,
            "Repair approval rejected because the file changed "
            "after the proposal was created. A new proposal is required.",
            ["The file on disk no longer matches the proposal's original."],
        )

    # Preserve the exact pre-change content for rollback.
    state.repair_original_content = current_content

    workspace.atomic_write_file(
        relative_path=state.current_file,
        content=state.proposed_content,
    )

    approval = state.repair_approval

    state.current_file_content = state.proposed_content
    state.repair_approved = True
    state.repair_applied = True
    # The approval was used for exactly one application.
    state.repair_approval = None
    clear_human_input(state)
    state.last_error = None
    state.error_inspected = False

    # The file changed: earlier test results no longer apply.
    # Approval itself does not verify anything.
    record_code_change(state)

    return ActionResult(
        action="approve_repair",
        success=True,
        message="Repair approved and applied.",
        data={
            "file": state.current_file,
            "repair_description": state.repair_description,
            "fingerprint": approval.fingerprint,
        },
    )
