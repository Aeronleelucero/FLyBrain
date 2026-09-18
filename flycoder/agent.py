"""Task-aware FLY-CODER agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from flycoder.actions import create_action_registry
from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.dependencies import build_dependency_graph
from flycoder.tools.filesystem import Workspace
from flycoder.tools.symbols import analyze_symbols


class FlyCoderAgent:
    """Coding agent that selects and executes actions based on task intent."""

    ANALYSIS_ACTIONS = {
        "explain_error",
        "review_code",
        "improve_code",
    }

    MAX_REVIEW_FILES = 8

    def __init__(self, workspace: str | Path):
        self.workspace = Workspace(workspace)
        self.actions = create_action_registry()

    # ==============================================================
    # TASK CLASSIFICATION
    # ==============================================================

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
            state.task_intent = self.classify_task(
                state.task
            )

    # ==============================================================
    # TASK WORDS
    # ==============================================================

    @staticmethod
    def _extract_task_words(task: str) -> set[str]:
        """Extract meaningful words from a task description."""

        task_words = {
            word.strip(
                ".,:;!?()[]{}\"'"
            )
            for word in task.lower().split()
            if len(
                word.strip(
                    ".,:;!?()[]{}\"'"
                )
            ) >= 3
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

        return task_words - ignored_words

    # ==============================================================
    # SYMBOL-AWARE FILE SCORING
    # ==============================================================

    @staticmethod
    def _score_file_symbols(
        task_words: set[str],
        file_path: str,
        workspace: Workspace,
    ) -> int:
        """Score a Python file according to its actual symbols."""

        if not file_path.lower().endswith(".py"):
            return 0

        try:
            content = workspace.read_file(
                file_path
            )
        except (
            FileNotFoundError,
            UnicodeDecodeError,
        ):
            return 0

        symbols = analyze_symbols(
            content,
            file_path=file_path,
        )

        if not symbols:
            return 0

        score = 0

        symbol_names = {
            symbol.name.lower()
            for symbol in symbols
        }

        for word in task_words:
            # Exact symbol match.
            if word in symbol_names:
                score += 12
                continue

            # Partial symbol match.
            for symbol_name in symbol_names:
                if (
                    word in symbol_name
                    or symbol_name in word
                ):
                    score += 4
                    break

        return score

    # ==============================================================
    # FILE SELECTION
    # ==============================================================

    @staticmethod
    def select_relevant_files(
        task: str,
        files: list[str],
        max_files: int = 5,
        workspace: Workspace | None = None,
    ) -> list[str]:
        """Select the files most relevant to the user's task."""

        if not files:
            return []

        task_words = (
            FlyCoderAgent._extract_task_words(
                task
            )
        )

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

            # ------------------------------------------------------
            # Task/path matching
            # ------------------------------------------------------

            for word in task_words:
                if word in path_words:
                    score += 5
                elif word in normalized_path:
                    score += 2

            # ------------------------------------------------------
            # Source preference
            # ------------------------------------------------------

            if normalized_path.startswith(
                "flycoder/"
            ):
                score += 4

            if normalized_path.endswith(".py"):
                score += 2

            # ------------------------------------------------------
            # Testing preference
            # ------------------------------------------------------

            if "test" in task_words:
                if (
                    "/tests/" in f"/{normalized_path}/"
                    or "test_" in normalized_path
                    or normalized_path.startswith(
                        "tests/"
                    )
                ):
                    score += 6

            # ------------------------------------------------------
            # Directory priorities
            # ------------------------------------------------------

            if normalized_path.startswith(
                "experiments/"
            ):
                score -= 3

            if normalized_path.endswith(
                "/__init__.py"
            ):
                score -= 3

            # ------------------------------------------------------
            # Agent-specific priorities
            # ------------------------------------------------------

            if "agent" in task_words:

                if normalized_path.endswith(
                    "agent.py"
                ):
                    score += 10

                if normalized_path.endswith(
                    "state.py"
                ):
                    score += 7

                if "/core/" in (
                    f"/{normalized_path}/"
                ):
                    score += 5

                if "/actions/" in (
                    f"/{normalized_path}/"
                ):
                    score += 4

                if "/tools/" in (
                    f"/{normalized_path}/"
                ):
                    score += 3

            # ------------------------------------------------------
            # Phase 3 symbol intelligence
            # ------------------------------------------------------

            if workspace is not None:
                score += (
                    FlyCoderAgent._score_file_symbols(
                        task_words,
                        file_path,
                        workspace,
                    )
                )

            scored_files.append(
                (score, file_path)
            )

        scored_files.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        return [
            file_path
            for score, file_path in scored_files[
                :max_files
            ]
            if score > 0
        ]

    # ==============================================================
    # DEPENDENCY ANALYSIS
    # ==============================================================

    def expand_dependencies(
        self,
        state: CodingState,
    ) -> None:
        """Build and apply a recursive dependency graph."""

        if not state.relevant_files:
            return

        graph = build_dependency_graph(
            self.workspace,
            state.files,
            state.relevant_files,
            max_depth=3,
        )

        state.dependency_graph = graph

        expanded: list[str] = []

        def add_file(file_path: str) -> None:
            """Add a valid, non-empty file once."""

            if file_path in expanded:
                return

            try:
                content = self.workspace.read_file(
                    file_path
                )
            except (
                FileNotFoundError,
                UnicodeDecodeError,
            ):
                return

            if not content.strip():
                return

            expanded.append(file_path)

        # Keep originally selected files first.
        for file_path in state.relevant_files:
            add_file(file_path)

        # Add recursively discovered dependencies.
        for dependencies in graph.values():
            for dependency in dependencies:
                add_file(dependency)

        state.relevant_files = expanded[
            :self.MAX_REVIEW_FILES
        ]

    # ==============================================================
    # DECISION ENGINE
    # ==============================================================

    def decide(
        self,
        state: CodingState,
    ) -> str:
        """Choose the next action based on the current state."""

        self.initialize_task(state)

        # ----------------------------------------------------------
        # Finished
        # ----------------------------------------------------------

        if state.finished:
            return "finish"

        # ----------------------------------------------------------
        # Discover files
        # ----------------------------------------------------------

        if not state.files:
            return "inspect_files"

        # ----------------------------------------------------------
        # Select relevant files
        # ----------------------------------------------------------

        if (
            state.task_intent
            in {
                "explain",
                "review",
                "improve",
            }
            and not state.relevant_files
        ):
            state.relevant_files = (
                self.select_relevant_files(
                    state.task,
                    state.files,
                    max_files=5,
                    workspace=self.workspace,
                )
            )

            if not state.relevant_files:
                state.relevant_files = (
                    state.files[:5]
                )

            if state.task_intent in {
                "review",
                "improve",
            }:
                self.expand_dependencies(
                    state
                )

        # ----------------------------------------------------------
        # Inspect
        # ----------------------------------------------------------

        if state.task_intent == "inspect":
            return "finish"

        # ----------------------------------------------------------
        # Analysis workflows
        # ----------------------------------------------------------

        if state.task_intent in {
            "explain",
            "review",
            "improve",
        }:

            remaining_files = [
                file_path
                for file_path in state.relevant_files
                if file_path
                not in state.files_analyzed
            ]

            if not remaining_files:
                return "finish"

            next_file = remaining_files[0]

            if state.current_file != next_file:
                state.current_file = next_file
                state.current_file_content = None

            if state.current_file_content is None:
                return "read_file"

            if state.task_intent == "explain":
                return "explain_error"

            if state.task_intent == "review":
                return "review_code"

            if state.task_intent == "improve":
                return "improve_code"

        # ----------------------------------------------------------
        # Repair workflow
        # ----------------------------------------------------------

        if (
            state.last_error
            and not state.error_inspected
        ):
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

        # ----------------------------------------------------------
        # Test workflow
        # ----------------------------------------------------------

        if not state.tests_run:
            return "run_tests"

        return "finish"

    # ==============================================================
    # REVIEW REPORT
    # ==============================================================

    def print_review_report(
        self,
        state: CodingState,
    ) -> None:
        """Print an aggregated project-level code review."""

        print()
        print("=" * 60)
        print("FLY-CODER REVIEW REPORT")
        print("=" * 60)
        print()

        print(
            f"Task: {state.task}"
        )

        print(
            f"Files reviewed: "
            f"{len(state.review_findings)}"
        )

        print()

        total_issues = 0

        for result in state.review_findings:

            print("-" * 60)
            print(result["file"])

            print(
                f"Lines: {result['lines']}"
            )

            findings = result["findings"]

            real_findings = [
                finding
                for finding in findings
                if not finding.startswith(
                    "No basic static-review"
                )
            ]

            if real_findings:

                total_issues += len(
                    real_findings
                )

                for finding in real_findings:
                    print(
                        f"  ⚠ {finding}"
                    )

            else:
                print(
                    "  ✓ No basic issues detected."
                )

        print()
        print("-" * 60)

        print(
            f"Total issues found: "
            f"{total_issues}"
        )

        print("=" * 60)
        print()

    # ==============================================================
    # DEPENDENCY REPORT
    # ==============================================================

    def print_dependency_graph(
        self,
        state: CodingState,
    ) -> None:
        """Print the discovered local dependency graph."""

        if not state.dependency_graph:
            return

        print()
        print("=" * 60)
        print("FLY-CODER DEPENDENCY GRAPH")
        print("=" * 60)

        for source, dependencies in (
            state.dependency_graph.items()
        ):
            print()
            print(source)

            if dependencies:

                for dependency in dependencies:
                    print(
                        f"  -> {dependency}"
                    )

            else:
                print(
                    "  -> no local dependencies"
                )

        print()
        print("=" * 60)
        print()

    # ==============================================================
    # EXECUTION
    # ==============================================================

    def run_once(
        self,
        state: CodingState,
    ) -> ActionResult:
        """Choose and execute one action."""

        action = self.decide(state)

        # ----------------------------------------------------------
        # Finish
        # ----------------------------------------------------------

        if action == "finish":

            state.finished = True

            # Dependency graph is displayed before
            # the review report.
            if state.dependency_graph:
                self.print_dependency_graph(
                    state
                )

            if state.task_intent == "review":
                self.print_review_report(
                    state
                )

            # ------------------------------------------------------
            # Determine finish message
            # ------------------------------------------------------

            if (
                state.repair_proposed
                and not state.repair_applied
            ):
                message = (
                    "Agent stopped safely. "
                    "Review and approve the repair "
                    "proposal."
                )

            elif state.task_intent == "inspect":
                message = (
                    "Project inspection completed."
                )

            elif state.task_intent == "explain":
                message = (
                    "Error explanation completed."
                )

            elif state.task_intent == "review":
                message = (
                    "Code review completed."
                )

            elif state.task_intent == "improve":
                message = (
                    "Improvement proposal created. "
                    "No file was modified."
                )

            elif state.tests_passed:
                message = (
                    f"Task intent "
                    f"'{state.task_intent}' "
                    "completed. All tests passed."
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
                    "files_reviewed": len(
                        state.files_analyzed
                    ),
                    "review_findings": (
                        state.review_findings
                    ),
                    "dependency_graph": (
                        state.dependency_graph
                    ),
                },
            )

        # ----------------------------------------------------------
        # Execute action
        # ----------------------------------------------------------

        result = self.actions.execute(
            action,
            workspace=self.workspace,
            state=state,
        )

        # ----------------------------------------------------------
        # Display analysis output
        # ----------------------------------------------------------

        if (
            action == "explain_error"
            and result.data
        ):
            print()
            print("Error explanation:")
            print(
                result.data.get(
                    "explanation",
                    "",
                )
            )

        elif (
            action == "review_code"
            and result.data
        ):
            print()
            print("Code review:")
            print(
                result.data.get(
                    "review",
                    "",
                )
            )

        elif (
            action == "improve_code"
            and result.data
        ):
            print()
            print("Improvement proposal:")

            improvements = result.data.get(
                "improvements",
                [],
            )

            if improvements:

                for item in improvements:
                    print(
                        f"  - {item}"
                    )

            else:
                print(
                    "  No specific improvements "
                    "were identified."
                )

        # ----------------------------------------------------------
        # Update analysis state
        # ----------------------------------------------------------

        if action in self.ANALYSIS_ACTIONS:

            if (
                state.current_file
                and state.current_file
                not in state.files_analyzed
            ):
                state.files_analyzed.append(
                    state.current_file
                )

            if action == "explain_error":

                state.finished = True

            elif action == "review_code":

                # Clear the current file so the next
                # file will be read.
                state.current_file_content = None

                # IMPORTANT:
                # Do NOT set state.finished here.
                #
                # The next agent loop must call decide()
                # again so it can return "finish" and print
                # the final dependency graph and review report.

            elif action == "improve_code":

                state.finished = True

        return result


# ==================================================================
# CLI
# ==================================================================


def cli() -> None:
    """Run FLY-CODER from the command line."""

    parser = argparse.ArgumentParser(
        description="Task-aware FLY-CODER agent"
    )

    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help=(
            "Workspace directory. "
            "Defaults to the current directory."
        ),
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
        help=(
            "Maximum number of agent steps."
        ),
    )

    parser.add_argument(
        "--approve",
        action="store_true",
        help=(
            "Automatically approve "
            "a proposed repair."
        ),
    )

    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error(
            "--max-steps must be at least 1"
        )

    agent = FlyCoderAgent(
        args.workspace
    )

    state = CodingState(
        task=args.task
    )

    print(
        f"Task: {args.task}"
    )

    print(
        "Task Intent: "
        f"{agent.classify_task(args.task)}"
    )

    print()

    for step in range(
        args.max_steps
    ):

        if state.finished:
            break

        result = agent.run_once(
            state
        )

        print(
            f"[Step {step}] "
            f"{result.action} | "
            f"success={result.success}"
        )

        if result.message:
            print(
                f"  {result.message}"
            )

        if (
            result.data
            and result.action
            not in FlyCoderAgent.ANALYSIS_ACTIONS
        ):
            print(
                f"  data={result.data}"
            )

        if (
            args.approve
            and state.repair_proposed
            and not state.repair_applied
            and state.user_input_needed
        ):

            approval_result = (
                agent.actions.execute(
                    "approve_repair",
                    workspace=agent.workspace,
                    state=state,
                )
            )

            print(
                f"[Step {step}] "
                f"{approval_result.action} | "
                f"success="
                f"{approval_result.success}"
            )

            if approval_result.message:
                print(
                    f"  "
                    f"{approval_result.message}"
                )

    print()

    if state.finished:
        print(
            "Agent finished."
        )

    else:
        print(
            "Maximum steps reached."
        )

    if (
        state.repair_proposed
        and not state.repair_applied
    ):
        print(
            "A repair proposal is waiting "
            "for approval."
        )


if __name__ == "__main__":
    cli()
