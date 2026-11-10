"""Testing actions for FLY-CODER."""

import os
import subprocess
import sys

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace
from flycoder.tools.freshness import record_test_run


def run_tests_action(
    workspace: Workspace,
    state: CodingState,
) -> ActionResult:
    """Run pytest inside the workspace."""

    environment = os.environ.copy()

    project_root = str(workspace.root.parent)

    existing_python_path = environment.get("PYTHONPATH")

    if existing_python_path:
        environment["PYTHONPATH"] = (
            f"{project_root}{os.pathsep}"
            f"{existing_python_path}"
        )
    else:
        environment["PYTHONPATH"] = project_root

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
    ]

    result = subprocess.run(
        command,
        cwd=workspace.root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    output = "\n".join(
        part
        for part in [
            result.stdout,
            result.stderr,
        ]
        if part
    ).strip()

    # Records the result for the current code version and resets the
    # error-inspection cycle (see flycoder.tools.freshness).
    record_test_run(
        state,
        passed=result.returncode == 0,
        output=output,
    )

    return ActionResult(
        action="run_tests",
        success=state.tests_passed,
        message=(
            "Tests passed."
            if state.tests_passed
            else "Tests failed."
        ),
        data={
            "return_code": result.returncode,
            "output": output,
        },
    )