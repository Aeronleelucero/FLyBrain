"""Repair proposal actions for FLY-CODER."""

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.changes import (
    build_change_diff,
    create_change_proposal,
)
from flycoder.tools.filesystem import Workspace


def propose_repair_action(
    workspace: Workspace,
    state: CodingState,
) -> ActionResult:
    """Create a safe repair proposal without writing files."""

    if not state.current_file:
        return ActionResult(
            action="propose_repair",
            success=False,
            message="No file is selected for repair.",
        )

    if state.current_file_content is None:
        return ActionResult(
            action="propose_repair",
            success=False,
            message="The selected file has not been read.",
        )

    content = state.current_file_content

    if (
        "assert False" in content
        and "test_" in content
    ):
        proposed_content = content.replace(
            'assert False, "Intentional failure for testing"',
            'assert True, "Intentional failure for testing"',
        )

        description = (
            "Replace the intentional failing assertion "
            "with a passing assertion."
        )

    else:
        return ActionResult(
            action="propose_repair",
            success=False,
            message=(
                "No safe automatic repair pattern was found. "
                "Manual review is required."
            ),
            data={
                "file": state.current_file,
                "next_step": (
                    "Review the file and create a repair manually."
                ),
            },
        )

    proposal = create_change_proposal(
        file=state.current_file,
        original_content=content,
        proposed_content=proposed_content,
        description=description,
    )

    if not proposal.changed:
        return ActionResult(
            action="propose_repair",
            success=False,
            message="The repair proposal contains no changes.",
            data={
                "file": state.current_file,
            },
        )

    state.repair_description = proposal.description
    state.proposed_content = proposal.proposed_content
    state.repair_proposed = True
    state.repair_approved = False
    state.repair_applied = False
    state.user_input_needed = True

    return ActionResult(
        action="propose_repair",
        success=True,
        message="Repair proposal created. No file was modified.",
        data={
            "file": proposal.file,
            "description": proposal.description,
            "original_content": proposal.original_content,
            "proposed_content": proposal.proposed_content,
            "diff": build_change_diff(proposal),
            "requires_approval": True,
        },
    )
