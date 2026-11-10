"""Task-level progress tracking for FLY-CODER Phase 10.4.

Two levels of progress exist in FLY-CODER and must stay distinct:

* Iteration progress (``flycoder.tools.progress``) answers "did the
  last iteration change the task state?". It drives stall detection.
* Task progress (this module) answers "how far is the task from being
  done?". It is measured with intent-specific milestones evaluated
  directly from ``CodingState``.

The milestone model (Phase 10.4.1) is a pure function of state. It
keeps no counters of its own, so it cannot drift from the real task
state, and a milestone that regresses (for example, tests that must be
re-run after an approved repair) is reported as pending again.

This module is observational only. It never executes actions, mutates
state, grants approval, or bypasses guardrails.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from flycoder.state import CodingState
from flycoder.tools.freshness import tests_current, tests_stale
from flycoder.tools.iteration import IterationObservation


# ==================================================================
# PHASE 10.4.1 — PROGRESS MODEL
# ==================================================================

MILESTONE_PENDING = "PENDING"
MILESTONE_ACHIEVED = "ACHIEVED"
MILESTONE_NOT_APPLICABLE = "NOT_APPLICABLE"

MILESTONE_STATUSES = {
    MILESTONE_PENDING,
    MILESTONE_ACHIEVED,
    MILESTONE_NOT_APPLICABLE,
}

WORKSPACE_INSPECTED = "workspace_inspected"
FILES_SELECTED = "files_selected"
FILES_ANALYZED = "files_analyzed"
TESTS_RUN = "tests_run"
FAILURE_DIAGNOSED = "failure_diagnosed"
REPAIR_PROPOSED = "repair_proposed"
REPAIR_APPLIED = "repair_applied"
TESTS_PASSED = "tests_passed"


@dataclass(frozen=True)
class MilestoneDefinition:
    """Describe one milestone a task must reach to be complete."""

    key: str
    description: str
    requires_human: bool = False


@dataclass(frozen=True)
class MilestoneStatus:
    """Evaluated state of one milestone for the current task state."""

    key: str
    description: str
    status: str
    fraction: float
    detail: str
    requires_human: bool = False

    @property
    def achieved(self) -> bool:
        """Return whether the milestone has been reached."""

        return self.status == MILESTONE_ACHIEVED

    @property
    def applicable(self) -> bool:
        """Return whether the milestone applies to this task."""

        return self.status != MILESTONE_NOT_APPLICABLE

    @property
    def pending(self) -> bool:
        """Return whether the milestone still has to be reached."""

        return self.status == MILESTONE_PENDING


MILESTONE_DEFINITIONS: dict[str, MilestoneDefinition] = {
    WORKSPACE_INSPECTED: MilestoneDefinition(
        key=WORKSPACE_INSPECTED,
        description="Workspace files have been discovered.",
    ),
    FILES_SELECTED: MilestoneDefinition(
        key=FILES_SELECTED,
        description="Files relevant to the task have been selected.",
    ),
    FILES_ANALYZED: MilestoneDefinition(
        key=FILES_ANALYZED,
        description="Relevant files have been analyzed.",
    ),
    TESTS_RUN: MilestoneDefinition(
        key=TESTS_RUN,
        description="The test suite has been run on the current code.",
    ),
    FAILURE_DIAGNOSED: MilestoneDefinition(
        key=FAILURE_DIAGNOSED,
        description="The observed failure has been diagnosed.",
    ),
    REPAIR_PROPOSED: MilestoneDefinition(
        key=REPAIR_PROPOSED,
        description="A repair proposal has been created.",
    ),
    REPAIR_APPLIED: MilestoneDefinition(
        key=REPAIR_APPLIED,
        description=(
            "The repair has been explicitly approved by a human "
            "and applied."
        ),
        requires_human=True,
    ),
    TESTS_PASSED: MilestoneDefinition(
        key=TESTS_PASSED,
        description="The test suite passes on the current code.",
    ),
}


INTENT_MILESTONES: dict[str, tuple[str, ...]] = {
    "inspect": (
        WORKSPACE_INSPECTED,
    ),
    "explain": (
        WORKSPACE_INSPECTED,
        FILES_SELECTED,
        FILES_ANALYZED,
    ),
    "review": (
        WORKSPACE_INSPECTED,
        FILES_SELECTED,
        FILES_ANALYZED,
    ),
    "improve": (
        WORKSPACE_INSPECTED,
        FILES_SELECTED,
        FILES_ANALYZED,
    ),
    "test": (
        WORKSPACE_INSPECTED,
        TESTS_RUN,
        TESTS_PASSED,
    ),
    "repair": (
        WORKSPACE_INSPECTED,
        TESTS_RUN,
        FAILURE_DIAGNOSED,
        REPAIR_PROPOSED,
        REPAIR_APPLIED,
        TESTS_PASSED,
    ),
}


_Evaluation = tuple[str, float, str]


def _achieved(detail: str) -> _Evaluation:
    return MILESTONE_ACHIEVED, 1.0, detail


def _pending(detail: str, fraction: float = 0.0) -> _Evaluation:
    return MILESTONE_PENDING, fraction, detail


def _not_applicable(detail: str) -> _Evaluation:
    return MILESTONE_NOT_APPLICABLE, 0.0, detail


def _failure_observed(state: CodingState) -> bool:
    """Return whether the current task has observed a failure.

    The repair state chain is used as evidence too, because applying an
    approved repair clears ``last_error`` and ``error_inspected``.
    """

    return bool(
        (tests_current(state) and not state.tests_passed)
        or state.last_error
        or state.error_inspected
        or state.repair_proposed
    )


def _repair_not_needed(state: CodingState) -> bool:
    """Return whether tests pass without any repair being involved."""

    return (
        tests_current(state)
        and state.tests_passed
        and not _failure_observed(state)
    )


def _eval_workspace_inspected(
    state: CodingState,
    intent: str,
) -> _Evaluation:
    if state.files:
        return _achieved(f"{len(state.files)} file(s) discovered.")

    return _pending("Workspace files have not been discovered.")


def _eval_files_selected(
    state: CodingState,
    intent: str,
) -> _Evaluation:
    if state.relevant_files:
        return _achieved(
            f"{len(state.relevant_files)} relevant file(s) selected."
        )

    return _pending("No relevant files have been selected.")


def _eval_files_analyzed(
    state: CodingState,
    intent: str,
) -> _Evaluation:
    relevant = list(dict.fromkeys(state.relevant_files))
    analyzed = [
        file_path
        for file_path in relevant
        if file_path in state.files_analyzed
    ]

    if not relevant:
        return _pending("No relevant files are available to analyze.")

    # A review covers every relevant file. Explain and improve
    # workflows complete after the first analyzed file.
    if intent == "review":
        fraction = len(analyzed) / len(relevant)
        detail = f"{len(analyzed)} of {len(relevant)} file(s) analyzed."

        if len(analyzed) == len(relevant):
            return _achieved(detail)

        return _pending(detail, fraction)

    if analyzed:
        return _achieved(f"{len(analyzed)} file(s) analyzed.")

    return _pending("No relevant file has been analyzed.")


def _eval_tests_run(
    state: CodingState,
    intent: str,
) -> _Evaluation:
    if tests_current(state):
        return _achieved("Tests have been run on the current code.")

    if tests_stale(state):
        return _pending(
            "Recorded test results are stale; the code changed after "
            "they ran."
        )

    if state.repair_applied:
        return _pending(
            "Tests must be re-run after the applied repair."
        )

    return _pending("Tests have not been run.")


def _eval_failure_diagnosed(
    state: CodingState,
    intent: str,
) -> _Evaluation:
    if state.error_inspected or state.repair_proposed:
        return _achieved("The failure has been diagnosed.")

    if _repair_not_needed(state):
        return _not_applicable("No failure was observed.")

    if _failure_observed(state):
        return _pending("A failure was observed but not diagnosed.")

    return _pending("No failure information is available yet.")


def _eval_repair_proposed(
    state: CodingState,
    intent: str,
) -> _Evaluation:
    if state.repair_proposed:
        return _achieved("A repair proposal exists.")

    if _repair_not_needed(state):
        return _not_applicable("No repair is needed.")

    return _pending("No repair proposal exists.")


def _eval_repair_applied(
    state: CodingState,
    intent: str,
) -> _Evaluation:
    # Both flags are required. Guardrails already treat an applied but
    # unapproved repair as inconsistent; progress never counts it.
    if state.repair_applied and state.repair_approved:
        return _achieved("The approved repair has been applied.")

    if _repair_not_needed(state):
        return _not_applicable("No repair is needed.")

    if state.repair_proposed:
        return _pending(
            "The repair proposal is waiting for explicit human "
            "approval."
        )

    return _pending("No repair has been applied.")


def _eval_tests_passed(
    state: CodingState,
    intent: str,
) -> _Evaluation:
    if tests_current(state) and state.tests_passed:
        return _achieved("Tests pass on the current code.")

    if tests_current(state):
        return _pending("Tests ran and failed.")

    if tests_stale(state):
        return _pending("Recorded test results are stale.")

    return _pending("Tests have not been run on the current code.")


_EVALUATORS: dict[str, Callable[[CodingState, str], _Evaluation]] = {
    WORKSPACE_INSPECTED: _eval_workspace_inspected,
    FILES_SELECTED: _eval_files_selected,
    FILES_ANALYZED: _eval_files_analyzed,
    TESTS_RUN: _eval_tests_run,
    FAILURE_DIAGNOSED: _eval_failure_diagnosed,
    REPAIR_PROPOSED: _eval_repair_proposed,
    REPAIR_APPLIED: _eval_repair_applied,
    TESTS_PASSED: _eval_tests_passed,
}


@dataclass(frozen=True)
class TaskProgressModel:
    """Ordered milestones that define progress for one task intent."""

    intent: str
    milestones: tuple[MilestoneDefinition, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        """Return milestone keys in their natural order."""

        return tuple(milestone.key for milestone in self.milestones)

    def evaluate(
        self,
        state: CodingState,
    ) -> tuple[MilestoneStatus, ...]:
        """Evaluate every milestone against the current task state.

        Evaluation reads ``state`` only. It never mutates it.
        """

        statuses: list[MilestoneStatus] = []

        for milestone in self.milestones:
            status, fraction, detail = _EVALUATORS[milestone.key](
                state,
                self.intent,
            )

            statuses.append(
                MilestoneStatus(
                    key=milestone.key,
                    description=milestone.description,
                    status=status,
                    fraction=fraction,
                    detail=detail,
                    requires_human=milestone.requires_human,
                )
            )

        return tuple(statuses)


def build_progress_model(intent: str | None) -> TaskProgressModel:
    """Build the milestone model for a classified task intent.

    Raises:
        ValueError: if the intent is missing or unknown. Task intent
            must be classified before progress can be measured.
    """

    if not intent:
        raise ValueError(
            "Task intent must be classified before building a "
            "progress model."
        )

    if intent not in INTENT_MILESTONES:
        raise ValueError(f"Unknown task intent: {intent!r}")

    return TaskProgressModel(
        intent=intent,
        milestones=tuple(
            MILESTONE_DEFINITIONS[key]
            for key in INTENT_MILESTONES[intent]
        ),
    )


def evaluate_milestones(
    state: CodingState,
    intent: str | None = None,
) -> tuple[MilestoneStatus, ...]:
    """Evaluate milestones for ``state`` using its classified intent."""

    return build_progress_model(intent or state.task_intent).evaluate(
        state
    )


# ==================================================================
# PHASE 10.4.2 — PROGRESS METRICS
# ==================================================================


@dataclass(frozen=True)
class TaskProgressMetrics:
    """Explainable task-level progress derived from milestone statuses.

    The completion fraction measures how far the task has advanced
    through its applicable milestones. It is NOT a completion verdict:
    a high fraction, or even ``all_milestones_achieved``, never means
    the task is verified complete. Completion requires evidence and is
    decided separately (Phases 10.5 and 10.8). ``state.finished`` is
    not an input to these metrics at all.
    """

    intent: str
    milestones: tuple[MilestoneStatus, ...]

    @property
    def applicable(self) -> tuple[MilestoneStatus, ...]:
        """Return milestones that apply to this task."""

        return tuple(m for m in self.milestones if m.applicable)

    @property
    def applicable_count(self) -> int:
        return len(self.applicable)

    @property
    def achieved_count(self) -> int:
        return sum(1 for m in self.milestones if m.achieved)

    @property
    def pending_count(self) -> int:
        return sum(1 for m in self.milestones if m.pending)

    @property
    def not_applicable_count(self) -> int:
        return len(self.milestones) - self.applicable_count

    @property
    def completion_fraction(self) -> float:
        """Return progress through applicable milestones, 0.0 to 1.0.

        Not-applicable milestones are excluded from the denominator.
        Pending milestones contribute their partial fraction, so a
        review with half its files analyzed counts half a milestone.
        """

        applicable = self.applicable

        if not applicable:
            return 0.0

        return sum(m.fraction for m in applicable) / len(applicable)

    @property
    def percent(self) -> int:
        """Return a display percentage.

        100 is reserved for every applicable milestone being achieved,
        so rounding can never make an unfinished task look finished.
        """

        if self.all_milestones_achieved:
            return 100

        return min(99, round(self.completion_fraction * 100))

    @property
    def all_milestones_achieved(self) -> bool:
        """Return whether every applicable milestone is achieved.

        This is a progress statement, not verified completion.
        """

        return (
            self.applicable_count > 0
            and self.achieved_count == self.applicable_count
        )

    @property
    def next_milestone(self) -> MilestoneStatus | None:
        """Return the first pending milestone in natural order."""

        return next((m for m in self.milestones if m.pending), None)

    @property
    def awaiting_human(self) -> bool:
        """Return whether the next milestone needs explicit human action."""

        milestone = self.next_milestone

        return milestone is not None and milestone.requires_human

    def explain(self) -> list[str]:
        """Return human-readable lines describing every milestone."""

        lines = [
            f"Task progress ({self.intent}): {self.percent}% — "
            f"{self.achieved_count}/{self.applicable_count} applicable "
            f"milestone(s) achieved."
        ]

        for milestone in self.milestones:
            lines.append(
                f"  [{milestone.status}] {milestone.key}: "
                f"{milestone.detail}"
            )

        return lines


def measure_task_progress(
    state: CodingState,
    intent: str | None = None,
) -> TaskProgressMetrics:
    """Measure task-level progress for the current state (read-only)."""

    model = build_progress_model(intent or state.task_intent)

    return TaskProgressMetrics(
        intent=model.intent,
        milestones=model.evaluate(state),
    )


# ==================================================================
# ITERATION ACTIVITY STATISTICS
# ==================================================================


@dataclass
class TaskProgress:
    """Aggregate iteration activity across one task.

    This records execution activity: how many iterations ran, whether
    their actions succeeded, and whether each changed the task state.
    It deliberately says nothing about how close the task is to
    completion. Use ``measure_task_progress`` for that.

    Successful actions are not progress: an iteration can succeed
    without advancing any milestone, and a milestone can regress after
    a successful action (for example, an approved repair invalidates
    earlier test results).
    """

    total_iterations: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0
    progress_iterations: int = 0
    no_progress_iterations: int = 0
    history: list[bool] = field(default_factory=list)

    @property
    def progress_rate(self) -> float:
        """Return the fraction of iterations that made progress."""

        if self.total_iterations == 0:
            return 0.0

        return self.progress_iterations / self.total_iterations

    @property
    def has_progress(self) -> bool:
        """Return whether any iteration has made progress."""

        return self.progress_iterations > 0

    @property
    def latest_progress(self) -> bool | None:
        """Return the latest progress state."""

        if not self.history:
            return None

        return self.history[-1]

    @property
    def all_iterations_progressed(self) -> bool:
        """Return whether every recorded iteration made progress."""

        return (
            self.total_iterations > 0
            and self.progress_iterations == self.total_iterations
        )

    def record(self, observation: IterationObservation) -> None:
        """Record one iteration observation."""

        self.total_iterations += 1
        self.history.append(observation.progress)

        if observation.progress:
            self.progress_iterations += 1
        else:
            self.no_progress_iterations += 1

        if observation.success:
            self.successful_iterations += 1
        else:
            self.failed_iterations += 1

    def reset(self) -> None:
        """Reset all accumulated task progress."""

        self.total_iterations = 0
        self.successful_iterations = 0
        self.failed_iterations = 0
        self.progress_iterations = 0
        self.no_progress_iterations = 0
        self.history.clear()
