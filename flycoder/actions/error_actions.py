"""Error-handling actions for FLY-CODER."""

import re

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace


def fix_error_action(
    workspace: Workspace,
    state: CodingState,
) -> ActionResult:
    """Inspect the latest test failure."""

    if not state.last_error:
        return ActionResult(
            action="fix_error",
            success=False,
            message=(
        "Test failure inspected. "
        "Ready to read the failing file "
        "and propose a repair."
        ),
        data={
        "error": error_text[-3000:],
        "failing_file": failing_file,
        "failing_line": failing_line,
        "failing_test": failing_test,
        "workspace": str(workspace.root),
        "next_step": (
            "Read the failing file before modifying code."
        ),
    },
)

    error_text = state.last_error

    file_match = re.search(
        r"([A-Za-z0-9_./-]+\.py):(\d+):",
        error_text,
    )

    failing_file = None
    failing_line = None

    if file_match:
        raw_file = file_match.group(1)
        failing_line = int(file_match.group(2))

        workspace_name = workspace.root.name

        if raw_file.startswith(f"{workspace_name}/"):
            raw_file = raw_file[
                len(workspace_name) + 1:
            ]

        failing_file = raw_file
        state.current_file = failing_file
        state.current_file_content = None

    test_match = re.search(
        r"FAILED\s+(.+?)\s+-",
        error_text,
    )

    failing_test = (
        test_match.group(1)
        if test_match
        else None
    )

    # Prevent the agent from repeatedly inspecting the same error.
    state.error_inspected = True
    state.user_input_needed = True

    return ActionResult(
        action="fix_error",
        success=False,
        message=(
            "Test failure inspected. "
            "Ready to read the failing file and propose a repair."
        ),
        data={
            "error": error_text[-3000:],
            "failing_file": failing_file,
            "failing_line": failing_line,
            "failing_test": failing_test,
            "workspace": str(workspace.root),
            "next_step": (
                "Read the failing file before modifying code."
            ),
        },
    )
