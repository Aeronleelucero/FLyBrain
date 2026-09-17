"""Repair approval actions for FLY-CODER."""

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace


def approve_repair_action(
    workspace: Workspace,
    state: CodingState,
) -> ActionResult:
    """Apply an approved repair proposal."""

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

    workspace.write_file(
        relative_path=state.current_file,
        content=state.proposed_content,
    )

    state.current_file_content = state.proposed_content
    state.repair_approved = True
    state.repair_applied = True
    state.user_input_needed = False
    state.tests_run = False
    state.tests_passed = False
    state.last_error = None
    state.error_inspected = False

    return ActionResult(
        action="approve_repair",
        success=True,
        message="Repair approved and applied.",
        data={
            "file": state.current_file,
            "repair_description": state.repair_description,
        },
    )
