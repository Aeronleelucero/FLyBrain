"""Error-handling actions for FLY-CODER."""

import re

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace
from flycoder.tools.human_input import (
    HUMAN_INPUT_APPLIED_REPAIR_FAILED,
    HUMAN_INPUT_INSUFFICIENT_DIAGNOSIS,
    request_human_input,
)


def _workspace_file_exists(workspace: Workspace, file_path: str) -> bool:
    """Return whether a diagnosed path is a readable workspace file."""

    try:
        workspace.read_file(file_path)
    except (FileNotFoundError, ValueError, OSError, UnicodeDecodeError):
        return False

    return True


def fix_error_action(
    workspace: Workspace,
    state: CodingState,
) -> ActionResult:
    """Diagnose the latest recorded failure.

    Diagnosis is read-only with respect to the workspace. Its outcome:

    * A failing workspace file was identified: diagnosis continues. The
      file becomes ``current_file`` so it can be read next; no human input
      is requested, because reading and proposing do not modify code.
    * No failing file could be identified: human input is required
      (``insufficient_diagnosis``).
    * An approved repair is already applied and the failure persists:
      human input is required (``applied_repair_failed``), because only a
      human may choose between rollback and a new proposal.

    A diagnosis never authorizes, proposes, or applies a repair.
    """

    if not state.last_error:
        # Nothing to diagnose. Do not claim an inspection happened.
        return ActionResult(
            action="fix_error",
            success=False,
            message="No recorded error is available to diagnose.",
            data={
                "error": None,
                "failing_file": None,
                "failing_line": None,
                "failing_test": None,
                "workspace": str(workspace.root),
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

    test_match = re.search(
        r"FAILED\s+(.+?)\s+-",
        error_text,
    )

    failing_test = (
        test_match.group(1)
        if test_match
        else None
    )

    if failing_file is not None and not _workspace_file_exists(
        workspace,
        failing_file,
    ):
        failing_file = None

    # Phase 10.7: a human may have identified the relevant file after an
    # insufficient diagnosis. It is validated like any diagnosed path.
    hint_used = False

    if (
        failing_file is None
        and state.diagnosis_hint_file
        and _workspace_file_exists(workspace, state.diagnosis_hint_file)
    ):
        failing_file = state.diagnosis_hint_file
        hint_used = True

    # Prevent the agent from repeatedly inspecting the same error.
    state.error_inspected = True

    data = {
        "error": error_text[-3000:],
        "failing_file": failing_file,
        "failing_line": failing_line,
        "failing_test": failing_test,
        "workspace": str(workspace.root),
        "human_hint_used": hint_used,
    }

    if failing_file is None:
        request_human_input(state, HUMAN_INPUT_INSUFFICIENT_DIAGNOSIS)

        return ActionResult(
            action="fix_error",
            success=False,
            message=(
                "Error inspected, but no failing workspace file could be "
                "identified. Human input is required."
            ),
            data={
                **data,
                "next_step": "Investigate the failure manually.",
            },
        )

    state.current_file = failing_file
    state.current_file_content = None

    if state.repair_applied:
        request_human_input(state, HUMAN_INPUT_APPLIED_REPAIR_FAILED)

        return ActionResult(
            action="fix_error",
            success=True,
            message=(
                "Failure diagnosed after an applied repair. A human must "
                "decide between rollback and a new repair proposal."
            ),
            data={
                **data,
                "next_step": "Review the applied repair.",
            },
        )

    return ActionResult(
        action="fix_error",
        success=True,
        message=(
            "Test failure diagnosed. Failing file identified; continuing "
            "the investigation."
        ),
        data={
            **data,
            "next_step": "Read the failing file before proposing a repair.",
        },
    )
