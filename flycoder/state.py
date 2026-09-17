"""State definitions for the FLY-CODER agent."""

from dataclasses import dataclass, field


@dataclass
class CodingState:
    """Current state of a coding task."""

    task: str

    # The agent will classify the task during initialization.
    task_intent: str | None = None

    files: list[str] = field(default_factory=list)

    current_file: str | None = None
    current_file_content: str | None = None

    last_error: str | None = None

    tests_passed: bool = False
    tests_run: bool = False

    error_inspected: bool = False

    repair_proposed: bool = False
    repair_approved: bool = False
    repair_applied: bool = False

    repair_description: str | None = None
    proposed_content: str | None = None

    user_input_needed: bool = False
    finished: bool = False
