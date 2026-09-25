"""Learning feedback for FLY-CODER Phase 9.7."""

from __future__ import annotations

from dataclasses import dataclass, field

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.memory import (
    Experience,
    MemoryStore,
    record_failure,
    record_solution,
)


@dataclass
class LearningOutcome:
    """Structured outcome of an agent execution step."""

    task: str
    task_intent: str | None
    action: str
    success: bool

    strategy: str | None = None

    tests_run: bool = False
    tests_passed: bool = False

    error_present: bool = False
    error_inspected: bool = False

    repair_proposed: bool = False
    repair_approved: bool = False
    repair_applied: bool = False

    guardrail_blocked: bool = False
    human_input_required: bool = False

    confidence_score: float | None = None

    recovery_action: str | None = None

    message: str = ""

    evidence: list[str] = field(
        default_factory=list
    )

    lessons: list[str] = field(
        default_factory=list
    )


def _extract_guardrail_data(
    result: ActionResult,
) -> tuple[bool, bool]:
    """Extract guardrail status from an action result."""

    if not isinstance(result.data, dict):
        return False, False

    guardrail = result.data.get(
        "guardrail"
    )

    if not isinstance(guardrail, dict):
        return False, False

    allowed = guardrail.get(
        "allowed",
        True,
    )

    requires_human = guardrail.get(
        "requires_human_approval",
        False,
    )

    return (
        not allowed,
        bool(requires_human),
    )


def build_learning_outcome(
    result: ActionResult,
    state: CodingState,
    *,
    strategy: str | None = None,
    confidence_score: float | None = None,
    recovery_action: str | None = None,
) -> LearningOutcome:
    """Build structured learning feedback from execution state.

    This function is observational only. It never executes an action,
    changes workspace files, grants approval, or mutates the state.
    """

    guardrail_blocked, human_input_required = (
        _extract_guardrail_data(result)
    )

    outcome = LearningOutcome(
        task=state.task,
        task_intent=state.task_intent,
        action=result.action,
        success=result.success,
        strategy=strategy,
        tests_run=state.tests_run,
        tests_passed=state.tests_passed,
        error_present=bool(state.last_error),
        error_inspected=state.error_inspected,
        repair_proposed=state.repair_proposed,
        repair_approved=state.repair_approved,
        repair_applied=state.repair_applied,
        guardrail_blocked=guardrail_blocked,
        human_input_required=human_input_required,
        confidence_score=confidence_score,
        recovery_action=recovery_action,
        message=result.message,
    )

    # Evidence
    if result.success:
        outcome.evidence.append(
            f"Action '{result.action}' succeeded."
        )
    else:
        outcome.evidence.append(
            f"Action '{result.action}' failed."
        )

    if state.tests_run:
        if state.tests_passed:
            outcome.evidence.append(
                "Tests were run and passed."
            )
        else:
            outcome.evidence.append(
                "Tests were run but did not pass."
            )

    if state.error_inspected:
        outcome.evidence.append(
            "The recorded error was inspected."
        )

    if state.repair_proposed:
        outcome.evidence.append(
            "A repair proposal exists."
        )

    if state.repair_approved:
        outcome.evidence.append(
            "The repair received explicit approval."
        )

    if state.repair_applied:
        outcome.evidence.append(
            "The repair was applied."
        )

    if guardrail_blocked:
        outcome.evidence.append(
            "Execution was blocked by a guardrail."
        )

    if human_input_required:
        outcome.evidence.append(
            "Human input or approval was required."
        )

    if recovery_action:
        outcome.evidence.append(
            f"Recovery action selected: "
            f"{recovery_action}."
        )

    # Lessons
    if result.success:
        outcome.lessons.append(
            f"Action '{result.action}' produced a successful result."
        )
    else:
        outcome.lessons.append(
            f"Action '{result.action}' did not produce a successful result."
        )

    if guardrail_blocked:
        outcome.lessons.append(
            "The action reached a safety boundary and must "
            "not be treated as an execution success."
        )

    if state.tests_run and state.tests_passed:
        outcome.lessons.append(
            "The current execution path reached a passing test state."
        )

    if state.tests_run and not state.tests_passed:
        outcome.lessons.append(
            "The current execution path still requires correction "
            "or recovery."
        )

    if recovery_action:
        outcome.lessons.append(
            f"Future similar failures may require "
            f"recovery action '{recovery_action}'."
        )

    return outcome


def record_learning_outcome(
    store: MemoryStore,
    outcome: LearningOutcome,
    *,
    files: list[str] | None = None,
) -> Experience:
    """Record a learning outcome in the Phase 8 memory store.

    Successful outcomes are recorded through ``record_solution``.
    Failed outcomes are recorded through ``record_failure``.

    This function only records historical experience. It does not
    execute actions, modify workspace files, mutate CodingState,
    or grant approval.
    """

    notes = [
        *outcome.evidence,
        *outcome.lessons,
    ]

    if outcome.strategy:
        notes.append(
            f"Strategy used: {outcome.strategy}."
        )

    if outcome.task_intent:
        notes.append(
            f"Task intent: {outcome.task_intent}."
        )

    if outcome.confidence_score is not None:
        notes.append(
            "Confidence score: "
            f"{outcome.confidence_score:.3f}."
        )

    if outcome.recovery_action:
        notes.append(
            f"Recovery action: {outcome.recovery_action}."
        )

    if outcome.human_input_required:
        notes.append(
            "Human input was required."
        )

    if outcome.guardrail_blocked:
        notes.append(
            "Guardrail blocked execution."
        )

    normalized_files = list(files or [])

    if outcome.success:
        return record_solution(
            store=store,
            task=outcome.task,
            action=outcome.action,
            outcome=outcome.message,
            files=normalized_files,
            notes=notes,
            verified=(
                outcome.tests_run
                and outcome.tests_passed
                and outcome.repair_applied
            ),
        )

    error = outcome.message or None

    return record_failure(
        store=store,
        task=outcome.task,
        action=outcome.action,
        outcome=outcome.message,
        error=error,
        files=normalized_files,
        notes=notes,
    )
