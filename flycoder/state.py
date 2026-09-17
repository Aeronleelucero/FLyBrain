"""State definitions for the FLY-CODER agent."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CodingState:
    """Current state of a coding task."""

    task: str
    files: List[str] = field(default_factory=list)
    current_file: str | None = None
    last_error: str | None = None
    tests_passed: bool = False
    tests_run: bool = False
    user_input_needed: bool = False
    finished: bool = False