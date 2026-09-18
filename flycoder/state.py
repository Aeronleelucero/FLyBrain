"""State definitions for the FLY-CODER agent."""

from dataclasses import dataclass, field


@dataclass
class CodingState:
    """Current state of a coding task."""

    task: str

    # ----------------------------------------------------------
    # Task
    # ----------------------------------------------------------

    # The agent classifies the task during initialization.
    task_intent: str | None = None

    # ----------------------------------------------------------
    # File discovery and selection
    # ----------------------------------------------------------

    # All files discovered in the workspace.
    files: list[str] = field(default_factory=list)

    # Files selected as relevant to the current task.
    relevant_files: list[str] = field(default_factory=list)

    # Files that have already been analyzed.
    files_analyzed: list[str] = field(
        default_factory=list
    )

    # ----------------------------------------------------------
    # Code review
    # ----------------------------------------------------------

    # Aggregated results from code reviews.
    review_findings: list[dict] = field(
        default_factory=list
    )

    # ----------------------------------------------------------
    # Dependency analysis
    # ----------------------------------------------------------

    # Local Python dependency graph.
    #
    # Example:
    #
    # {
    #     "flycoder/agent.py": [
    #         "flycoder/state.py",
    #         "flycoder/tools/filesystem.py",
    #     ]
    # }
    dependency_graph: dict[str, list[str]] = field(
        default_factory=dict
    )

    # ----------------------------------------------------------
    # Current file
    # ----------------------------------------------------------

    current_file: str | None = None

    current_file_content: str | None = None

    # ----------------------------------------------------------
    # Errors and testing
    # ----------------------------------------------------------

    last_error: str | None = None

    tests_passed: bool = False

    tests_run: bool = False

    error_inspected: bool = False

    # ----------------------------------------------------------
    # Repair workflow
    # ----------------------------------------------------------

    repair_proposed: bool = False

    repair_approved: bool = False

    repair_applied: bool = False

    repair_description: str | None = None

    proposed_content: str | None = None

    # ----------------------------------------------------------
    # Agent control
    # ----------------------------------------------------------

    user_input_needed: bool = False

    finished: bool = False