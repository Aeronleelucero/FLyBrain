"""Task lifecycle state machine for FLY-CODER Phase 10.2."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


TASK_PENDING = "PENDING"
TASK_INITIALIZING = "INITIALIZING"
TASK_RUNNING = "RUNNING"
TASK_VERIFYING = "VERIFYING"
TASK_COMPLETED = "COMPLETED"
TASK_FAILED = "FAILED"
TASK_BLOCKED = "BLOCKED"
TASK_NEEDS_HUMAN = "NEEDS_HUMAN"


TERMINAL_STATES = {
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_BLOCKED,
    TASK_NEEDS_HUMAN,
}


ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    TASK_PENDING: {
        TASK_INITIALIZING,
    },
    TASK_INITIALIZING: {
        TASK_RUNNING,
        TASK_FAILED,
    },
    TASK_RUNNING: {
        TASK_RUNNING,
        TASK_VERIFYING,
        TASK_FAILED,
        TASK_BLOCKED,
        TASK_NEEDS_HUMAN,
    },
    TASK_VERIFYING: {
        TASK_COMPLETED,
        TASK_FAILED,
    },
    TASK_COMPLETED: set(),
    TASK_FAILED: set(),
    TASK_BLOCKED: set(),
    TASK_NEEDS_HUMAN: set(),
}


@dataclass
class TaskTransition:
    """Record one lifecycle state transition."""

    from_state: str
    to_state: str
    reason: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class TaskLifecycle:
    """Track the lifecycle of one FLY-CODER task."""

    task: str
    status: str = TASK_PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    current_iteration: int = 0
    actions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    transitions: list[TaskTransition] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        """Return whether the task has reached a terminal state."""

        return self.status in TERMINAL_STATES

    @property
    def completed(self) -> bool:
        """Return whether the task completed successfully."""

        return self.status == TASK_COMPLETED

    @property
    def failed(self) -> bool:
        """Return whether the task failed."""

        return self.status == TASK_FAILED

    @property
    def blocked(self) -> bool:
        """Return whether the task was blocked."""

        return self.status == TASK_BLOCKED

    @property
    def needs_human(self) -> bool:
        """Return whether human intervention is required."""

        return self.status == TASK_NEEDS_HUMAN

    def transition(self, new_status: str, reason: str = "") -> TaskTransition:
        """Move the task to another valid lifecycle state."""

        if new_status not in ALLOWED_TRANSITIONS.get(self.status, set()):
            raise ValueError(
                f"Invalid task lifecycle transition: "
                f"{self.status} -> {new_status}"
            )

        old_status = self.status

        self.status = new_status

        transition = TaskTransition(
            from_state=old_status,
            to_state=new_status,
            reason=reason,
        )

        self.transitions.append(transition)

        if new_status == TASK_INITIALIZING and self.started_at is None:
            self.started_at = transition.timestamp

        if new_status in TERMINAL_STATES:
            self.completed_at = transition.timestamp

        return transition

    def record_action(self, action: str) -> None:
        """Record an executed action."""

        if not action:
            return

        self.actions.append(action)

    def record_error(self, error: str) -> None:
        """Record an error encountered during the task."""

        if not error:
            return

        self.errors.append(error)

    def record_iteration(self, iteration: int) -> None:
        """Record the current autonomous iteration."""

        if iteration < 0:
            raise ValueError("iteration must be non-negative")

        self.current_iteration = iteration
