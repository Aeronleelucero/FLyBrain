"""First FLY-CODER coding-agent prototype."""

from pathlib import Path

from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace


class FlyCoderAgent:
    """Simple rule-based coding agent."""

    def __init__(self, workspace: str | Path):
        self.workspace = Workspace(workspace)

    def inspect(self, state: CodingState) -> CodingState:
        """Inspect workspace files."""

        state.files = self.workspace.list_files()
        return state

    def read_current_file(self, state: CodingState) -> CodingState:
        """Read the selected file."""

        if not state.current_file:
            if not state.files:
                state.last_error = "No files available to read."
                return state

            state.current_file = state.files[0]

        try:
            content = self.workspace.read_file(state.current_file)

            print()
            print(f"--- Reading: {state.current_file} ---")
            print(content)
            print("--- End file ---")
            print()

        except Exception as error:
            state.last_error = str(error)

        return state

    def decide(self, state: CodingState) -> str:
        """Choose the next coding action."""

        if not state.files:
            return "inspect_files"

        if not state.current_file:
            return "read_file"

        if state.last_error:
            return "fix_error"

        if not state.tests_run:
            return "run_tests"

        if not state.tests_passed:
            return "fix_error"

        return "finish"

    def run_once(self, state: CodingState) -> str:
        """Run one agent decision step."""

        action = self.decide(state)

        if action == "inspect_files":
            self.inspect(state)

        elif action == "read_file":
            self.read_current_file(state)

        elif action == "finish":
            state.finished = True

        return action