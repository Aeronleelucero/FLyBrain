"""Filesystem actions for FLY-CODER."""

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace


def inspect_files_action(
    workspace: Workspace,
    state: CodingState,
) -> ActionResult:
    """Inspect files in the workspace."""

    files = workspace.list_files()
    state.files = files

    return ActionResult(
        action="inspect_files",
        success=True,
        message=f"Found {len(files)} file(s).",
        data={
            "files": files,
        },
    )


def read_file_action(
    workspace: Workspace,
    state: CodingState,
    file_path: str | None = None,
) -> ActionResult:
    """Read a file from the workspace."""

    selected_file = file_path or state.current_file

    if not selected_file:
        if not state.files:
            return ActionResult(
                action="read_file",
                success=False,
                message="No files available to read.",
            )

        selected_file = state.files[0]

    content = workspace.read_file(selected_file)

    state.current_file = selected_file
    state.current_file_content = content

    return ActionResult(
        action="read_file",
        success=True,
        message=f"Read file: {selected_file}",
        data={
            "file": selected_file,
            "content": content,
        },
    )


def write_file_action(
    workspace: Workspace,
    state: CodingState,
    file_path: str,
    content: str,
) -> ActionResult:
    """Write code to a file."""

    workspace.write_file(
        relative_path=file_path,
        content=content,
    )

    state.current_file = file_path
    state.current_file_content = content

    return ActionResult(
        action="write_code",
        success=True,
        message=f"Wrote file: {file_path}",
        data={
            "file": file_path,
        },
    )
