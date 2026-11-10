"""Repair proposal actions for FLY-CODER."""

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.changes import (
    build_change_diff,
    create_change_proposal,
)
from flycoder.tools.filesystem import Workspace
from flycoder.tools.proposals import (
    compute_proposal_fingerprint,
    record_proposal,
)
from flycoder.tools.human_input import (
    HUMAN_INPUT_NO_SAFE_REPAIR,
    HUMAN_INPUT_REPAIR_APPROVAL,
    request_human_input,
)


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
        # Insufficient evidence for a safe repair: never manufacture one.
        # Stop at the human boundary so the attempt is not repeated.
        request_human_input(state, HUMAN_INPUT_NO_SAFE_REPAIR)

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
        request_human_input(state, HUMAN_INPUT_NO_SAFE_REPAIR)

        return ActionResult(
            action="propose_repair",
            success=False,
            message="The repair proposal contains no changes.",
            data={
                "file": state.current_file,
            },
        )

    fingerprint = compute_proposal_fingerprint(
        proposal.file,
        proposal.original_content,
        proposal.proposed_content,
    )

    if fingerprint in state.rejected_proposals:
        # A human already rejected exactly this proposal. Re-offering it
        # without new evidence would just repeat the same decision.
        request_human_input(state, HUMAN_INPUT_NO_SAFE_REPAIR)

        return ActionResult(
            action="propose_repair",
            success=False,
            message=(
                "The only safe automatic repair was already rejected. "
                "Manual review is required."
            ),
            data={
                "file": state.current_file,
                "rejected_fingerprint": fingerprint,
            },
        )

    # Binds the exact file, original, and proposed content (and the code
    # version) so a later approval can only apply this proposal.
    record_proposal(
        state,
        file=proposal.file,
        original_content=proposal.original_content,
        proposed_content=proposal.proposed_content,
        description=proposal.description,
    )

    # The proposal stops at the approval boundary. Nothing was written.
    request_human_input(state, HUMAN_INPUT_REPAIR_APPROVAL)

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
            "fingerprint": fingerprint,
            "code_version": state.code_version,
            "requires_approval": True,
        },
    )
