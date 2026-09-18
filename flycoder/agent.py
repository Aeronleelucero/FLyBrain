"""Task-aware FLY-CODER agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from flycoder.actions import create_action_registry
from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace


class FlyCoderAgent:
    """Coding agent that selects actions based on task intent."""

    ANALYSIS_ACTIONS = {
        "explain_error",
        "review_code",
        "improve_code",
    }

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
                "what went wrong",
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
                "audit",
            )
        ):
            return "review"

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

        return "inspect"

    def initialize_task(self, state: CodingState) -> None:
        """Classify the task once before execution."""

        if not state.task_intent:
            state.task_intent = self.classify_task(state.task)

    @staticmethod
    def select_relevant_files(
        task: str,
        files: list[str],
        max_files: int = 5,
    ) -> list[str]:
        """Select the files most relevant to the user's task."""

        if not files:
            return []

        task_words = {
            word.strip(".,:;!?()[]{}\"'")
            for word in task.lower().split()
            if len(word.strip(".,:;!?()[]{}\"'")) >= 3
        }

        ignored_words = {
            "the",
            "this",
            "that",
            "with",
            "from",
            "into",
            "code",
            "file",
            "files",
            "project",
            "review",
            "improve",
            "explain",
            "inspect",
            "fix",
            "debug",
        }

        task_words -= ignored_words

        if not task_words:
            return files[:max_files]

        scored_files: list[tuple[int, str]] = []

        for file_path in files:
            normalized_path = file_path.lower()

            path_words = set(
                normalized_path
                .replace("/", " ")
                .replace("_", " ")
                .replace("-", " ")
                .replace(".", " ")
                .split()
            )

            score = 0

            for word in task_words:
                if word in path_words:
                    score += 5
                elif word in normalized_path:
                    score += 2

            # Prioritize FLY-CODER's own implementation.
            if normalized_path.startswith("flycoder/"):
                score += 2

            # Prefer Python source files.
            if normalized_path.endswith(".py"):
                score += 2

            # Tests get additional relevance when the task mentions tests.
            if (
                "test" in task_words
                and "/tests/" in f"/{normalized_path}/"
            ):
                score += 4

            # Don't let experiments outrank core implementation
            # for generic coding-agent tasks.
            if normalized_path.startswith("experiments/"):
                score -= 2

            scored_files.append((score, file_path))

        scored_files.sort(
            key=lambda item: (-item[0], item[1])
        )

        return [
            file_path
            for score, file_path in scored_files[:max_files]
            if score > 0
        ]

    def decide(self, state: CodingState) -> str:
        """Choose the next action based on the current state."""

        self.initialize_task(state)

        if state.finished:
            return "finish"

        if not state.files:
            return "inspect_files"

        if state.task_intent == "inspect":
            return "finish"

        if state.task_intent in {
            "explain",
            "review",
            "improve",
        }:
            if not state.current_file:
                selected_files = self.select_relevant_files(
                    state.task,
                    state.files,
                    max_files=5,
                )

                if selected_files:
                    state.current_file = selected_files[0]

            if state.current_file_content is None:
                return "read_file"

            if state.task_intent == "explain":
                return "explain_error"

            if state.task_intent == "review":
                return "review_code"

            if state.task_intent == "improve":
                return "improve_code"

        # Repair workflow.
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

        # Test workflow.
        if not state.tests_run:
            return "run_tests"

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
            elif state.task_intent == "inspect":
                message = "Project inspection completed."
            elif state.task_intent == "explain":
                message = "Error explanation completed."
            elif state.task_intent == "review":
                message = "Code review completed."
            elif state.task_intent == "improve":
                message = (
                    "Improvement proposal created. "
                    "No file was modified."
                )
            elif state.tests_passed:
                message = (
                    f"Task intent '{state.task_intent}' completed. "
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
                data={
                    "task": state.task,
                    "task_intent": state.task_intent,
                },
            )

        result = self.actions.execute(
            action,
            workspace=self.workspace,
            state=state,
        )

        if action == "explain_error" and result.data:
            print()
            print("Error explanation:")
            print(result.data.get("explanation", ""))

        elif action == "review_code" and result.data:
            print()
            print("Code review:")
            print(result.data.get("review", ""))

        elif action == "improve_code" and result.data:
            print()
            print("Improvement proposal:")

            improvements = result.data.get("improvements", [])

            if improvements:
                for item in improvements:
                    print(f"  - {item}")
            else:
                print("  No specific improvements were identified.")

        if action in self.ANALYSIS_ACTIONS:
            state.finished = True

        return result


def cli() -> None:
    """Run FLY-CODER from the command line."""

    parser = argparse.ArgumentParser(
        description="Task-aware FLY-CODER agent"
    )

    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="Workspace directory. Defaults to the current directory.",
    )

    parser.add_argument(
        "--task",
        default="Inspect the project",
        help="Task to perform.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum number of agent steps.",
    )

    parser.add_argument(
        "--approve",
        action="store_true",
        help="Automatically approve a proposed repair.",
    )

    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")

    agent = FlyCoderAgent(args.workspace)
    state = CodingState(task=args.task)

    print(f"Task: {args.task}")
    print(f"Task Intent: {agent.classify_task(args.task)}")
    print()

    for step in range(args.max_steps):
        if state.finished:
            break

        result = agent.run_once(state)

        print(
            f"[Step {step}] "
            f"{result.action} | "
            f"success={result.success}"
        )

        if result.message:
            print(f"  {result.message}")

        if (
            result.data
            and result.action not in FlyCoderAgent.ANALYSIS_ACTIONS
        ):
            print(f"  data={result.data}")

        if (
            args.approve
            and state.repair_proposed
            and not state.repair_applied
            and state.user_input_needed
        ):
            approval_result = agent.actions.execute(
                "approve_repair",
                workspace=agent.workspace,
                state=state,
            )

            print(
                f"[Step {step}] "
                f"{approval_result.action} | "
                f"success={approval_result.success}"
            )

            if approval_result.message:
                print(f"  {approval_result.message}")

    print()

    if state.finished:
        print("Agent finished.")
    else:
        print("Maximum steps reached.")

    if state.repair_proposed and not state.repair_applied:
        print("A repair proposal is waiting for approval.")


if __name__ == "__main__":
    cli()
