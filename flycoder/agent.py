"""Task-aware FLY-CODER agent."""

from pathlib import Path

from flycoder.actions import create_action_registry
from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace


class FlyCoderAgent:
    """Coding agent that selects actions based on task intent."""

    def __init__(self, workspace: str | Path):
        self.workspace = Workspace(workspace)
        self.actions = create_action_registry()

    @staticmethod
    def classify_task(task: str) -> str:
        """Classify the user's task into an execution intent."""

        text = task.lower().strip()

        if any(
            phrase in text
            for phrase in (
                "explain the error",
                "explain error",
                "why does",
                "why is",
                "what is wrong",
            )
        ):
            return "explain"

        if any(
            phrase in text
            for phrase in (
                "fix",
                "repair",
                "debug",
                "failing test",
                "broken test",
            )
        ):
            return "repair"

        if any(
            phrase in text
            for phrase in (
                "improve",
                "refactor",
                "optimize",
                "enhance",
            )
        ):
            return "improve"

        if any(
            phrase in text
            for phrase in (
                "review",
                "code review",
                "review the code",
            )
        ):
            return "review"

        if any(
            phrase in text
            for phrase in (
                "inspect",
                "list files",
                "show files",
                "explore project",
            )
        ):
            return "inspect"

        if any(
            phrase in text
            for phrase in (
                "test",
                "tests",
                "pytest",
                "verify",
                "run all",
            )
        ):
            return "test"

        return "inspect"

    def initialize_task(self, state: CodingState) -> None:
        """Classify the task once before execution."""

        if not state.task_intent:
            state.task_intent = self.classify_task(state.task)

    def decide(self, state: CodingState) -> str:
        """Choose the next action based on the current state."""

        self.initialize_task(state)

        if state.finished:
            return "finish"

        if not state.files:
            return "inspect_files"

        # Inspection-only tasks finish after listing project files.
        if state.task_intent == "inspect":
            state.finished = True
            return "finish"

        # Explain tasks inspect the available error without modifying files.
        if state.task_intent == "explain":
            if state.last_error and not state.error_inspected:
                return "fix_error"

            if state.last_error and state.error_inspected:
                state.finished = True
                return "finish"

            if not state.tests_run:
                return "run_tests"

            state.finished = True
            return "finish"

        # Repair, test, review, and improve tasks begin with tests.
        if state.last_error and not state.error_inspected:
            return "fix_error"

        if (
            state.error_inspected
            and state.current_file
            and state.current_file_content is None
        ):
            return "read_file"

        if (
            state.task_intent == "repair"
            and state.error_inspected
            and state.current_file_content is not None
            and not state.repair_proposed
        ):
            return "propose_repair"

        if state.user_input_needed:
            return "finish"

        if not state.tests_run:
            return "run_tests"

        if not state.tests_passed:
            state.finished = True
            return "finish"

        # Review and improve currently stop safely after verification.
        # Actual code modification logic can be added next.
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
                    f"Task intent '{state.task_intent}' completed. "
                    "All tests passed."
                )
            elif state.task_intent == "inspect":
                message = "Project inspection completed."
            elif state.task_intent == "explain":
                message = (
                    "Error inspection completed. "
                    "No files were modified."
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
                data={
                    "task": state.task,
                    "task_intent": state.task_intent,
                },
            )

        return self.actions.execute(
            action,
            workspace=self.workspace,
            state=state,
        )