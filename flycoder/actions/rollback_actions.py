"""Repair rollback actions for FLY-CODER."""

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace


def rollback_repair_action(
    workspace: Workspace,
    state: CodingState,
) -> ActionResult:
    """Restore the original content of the most recently applied repair."""

    if not state.repair_applied:
        return ActionResult(
            action="rollback_repair",
            success=False,
            message="No applied repair is available for rollback.",
        )

    if not state.current_file:
        return ActionResult(
            action="rollback_repair",
            success=False,
            message="No file is selected for rollback.",
        )

    if state.repair_original_content is None:
        return ActionResult(
            action="rollback_repair",
            success=False,
            message="No original content is available for rollback.",
        )

    current_content = workspace.read_file(state.current_file)

    if (
        state.proposed_content is None
        or current_content != state.proposed_content
    ):
        return ActionResult(
            action="rollback_repair",
            success=False,
            message=(
                "Rollback rejected because the file changed after "
                "the repair was applied."
            ),
            data={
                "file": state.current_file,
                "changed_after_apply": True,
                "requires_manual_review": True,
            },
        )

    workspace.atomic_write_file(
        relative_path=state.current_file,
        content=state.repair_original_content,
    )

    state.current_file_content = state.repair_original_content
    state.repair_applied = False
    state.repair_approved = False
    state.repair_proposed = False
    state.proposed_content = None
    state.repair_original_content = None
    state.user_input_needed = False

    return ActionResult(
        action="rollback_repair",
        success=True,
        message="Repair rolled back successfully.",
        data={
            "file": state.current_file,
        },
    )
