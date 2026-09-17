"""FLY-CODER agent with executable actions."""

from pathlib import Path

from flycoder.actions import create_action_registry
from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace


class FlyCoderAgent:
    """Coding agent that selects and executes registered actions."""

    def __init__(self, workspace: str | Path):
        self.workspace = Workspace(workspace)
        self.actions = create_action_registry()

    def decide(self, state: CodingState) -> str:
        """Decide which action to take based on the current state."""

        if state.finished:
            return "finish"

        if not state.files:
            return "inspect_files"

        if state.last_error and not state.error_inspected:
            return "fix_error"

        if (
            state.error_inspected
            and state.current_file
            and state.current_file_content is None
        ):
            return "read_file"

        if (
            state.error_inspected
            and state.current_file_content is not None
            and not state.repair_proposed
        ):
            return "propose_repair"

        if state.user_input_needed:
            return "finish"

        if not state.current_file:
            return "read_file"

        if not state.tests_run:
            return "run_tests"

        if not state.tests_passed:
            state.finished = True
            return "finish"

        state.finished = True
        return "finish"

    def run_once(self, state: CodingState) -> ActionResult:
        """Choose and execute one action."""

        action = self.decide(state)

        if action == "finish":
            state.finished = True

            if state.repair_proposed and not state.repair_applied:
                message = (
                    "Agent stopped safely. "
                    "Review and approve the repair proposal."
                )
            elif state.tests_passed:
                message = (
                    "Agent finished successfully. "
                    "All tests passed."
                )
            else:
                message = (
                    "Agent stopped safely. "
                    "Manual review is required."
                )

            return ActionResult(
                action="finish",
                success=True,
                message=message,
            )

        return self.actions.execute(
            action,
            workspace=self.workspace,
            state=state,
        )