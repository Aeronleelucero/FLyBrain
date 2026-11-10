"""Autonomous iteration tracking for FLY-CODER Phase 10.3."""

from __future__ import annotations

from dataclasses import dataclass, field

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.execution import ExecutionObservation


@dataclass
class IterationObservation:
    """Record what happened during one autonomous iteration."""

    iteration: int
    action: str
    success: bool
    message: str
    state_signature: str
    next_action: str | None = None
    progress: bool = False
    notes: list[str] = field(default_factory=list)


def build_state_signature(state: CodingState) -> str:
    """Build a deterministic signature representing relevant task state."""

    values = (
        state.task,
        state.current_file,
        state.current_file_content,
        state.last_error,
        state.tests_run,
        state.tests_passed,
        state.error_inspected,
        state.repair_proposed,
        state.repair_approved,
        state.repair_applied,
        state.user_input_needed,
        state.finished,
    )

    return repr(values)


def observe_iteration(
    iteration: int,
    result: ActionResult,
    state: CodingState,
    observation: ExecutionObservation | None = None,
    *,
    previous_signature: str | None = None,
) -> IterationObservation:
    """Create an observational record for one autonomous iteration."""

    state_signature = build_state_signature(state)

    progress = (
        previous_signature is not None
        and state_signature != previous_signature
    )

    next_action = None

    if observation is not None:
        next_action = observation.next_action

    notes: list[str] = []

    if result.success:
        notes.append("Action completed successfully.")
    else:
        notes.append("Action failed.")

    if progress:
        notes.append("Task state changed during the iteration.")
    else:
        notes.append("No relevant task-state change was detected.")

    if state.finished:
        notes.append("Task is marked finished.")

    if state.user_input_needed:
        notes.append("Human input is required.")

    return IterationObservation(
        iteration=iteration,
        action=result.action,
        success=result.success,
        message=result.message,
        state_signature=state_signature,
        next_action=next_action,
        progress=progress,
        notes=notes,
    )
