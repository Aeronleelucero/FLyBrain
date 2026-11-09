"""Unified task progress reporting for FLY-CODER Phase 10.4.5.

A progress report combines the Phase 10.4 components into one
immutable, deterministic, human-readable view:

* task progress  — ``measure_task_progress`` (10.4.1 / 10.4.2)
* history        — ``ProgressHistory`` values (10.4.3)
* evidence       — a supplied ``CompletionEvidence`` (10.4.4)
* lifecycle      — a supplied ``TaskLifecycle`` (10.2)

Design decisions
----------------
* Nothing is recalculated. Frozen inputs (metrics, evidence, task-level
  snapshots and changes) are stored as they are. ``ProgressHistory``
  and ``TaskLifecycle`` are mutable, so the report captures the values
  they compute into frozen summaries. A later append or transition
  therefore cannot change a report that was already built.
* Unavailable is not negative. A section that was not supplied is
  ``None`` and is described as unavailable, never as failed.
* Completion is only claimed when the supplied evidence is VERIFIED.
  100% progress, ``state.finished`` and a COMPLETED lifecycle are all
  reported, but never treated as verification.
* Contradictions between the sources are reported side by side rather
  than resolved, so nothing is hidden.
* History here is task-level (``progress_history``). The iteration-level
  ``flycoder.tools.progress.ProgressTracker``/``ProgressSnapshot`` are a
  different model and are rejected.

Reporting is observational only. It never mutates state, executes
actions, runs tests, approves repairs, or controls execution.
"""

from __future__ import annotations

from dataclasses import dataclass

from flycoder.state import CodingState
from flycoder.tools.completion_evidence import (
    EVIDENCE_INCONSISTENT,
    CompletionEvidence,
    build_code_signature,
)
from flycoder.tools.progress_history import (
    FRACTION_TOLERANCE,
    PROGRESS_DECREASED,
    PROGRESS_INCREASED,
    ProgressChange,
    ProgressHistory,
    ProgressSnapshot,
)
from flycoder.tools.task_lifecycle import (
    TASK_COMPLETED,
    TASK_NEEDS_HUMAN,
    TaskLifecycle,
)
from flycoder.tools.task_progress import (
    TaskProgressMetrics,
    measure_task_progress,
)


RECENT_CHANGE_LIMIT = 3


def _percent_points(fraction: float) -> str:
    return f"{fraction * 100:+.0f} points"


# ==================================================================
# FROZEN SECTION SUMMARIES
# ==================================================================


@dataclass(frozen=True)
class HistorySummary:
    """Frozen capture of values computed by a task ``ProgressHistory``."""

    snapshot_count: int
    initial: ProgressSnapshot | None
    latest: ProgressSnapshot | None
    peak: ProgressSnapshot | None
    net_progress: float
    increase_count: int
    regression_count: int
    below_peak: bool
    recent_changes: tuple[ProgressChange, ...]

    @classmethod
    def from_history(
        cls,
        history: ProgressHistory,
        recent_limit: int = RECENT_CHANGE_LIMIT,
    ) -> HistorySummary:
        """Capture history values using ProgressHistory's own logic."""

        if not isinstance(history, ProgressHistory):
            raise TypeError(
                "history must be a task-level ProgressHistory "
                "(flycoder.tools.progress_history), not an iteration-level "
                "ProgressTracker"
            )

        if recent_limit < 0:
            raise ValueError("recent_limit must be non-negative")

        changes = history.changes

        return cls(
            snapshot_count=len(history),
            initial=history.first,
            latest=history.latest,
            peak=history.peak,
            net_progress=history.net_progress,
            increase_count=history.increase_count,
            regression_count=history.regression_count,
            below_peak=history.below_peak,
            recent_changes=changes[-recent_limit:] if recent_limit else (),
        )

    @property
    def is_empty(self) -> bool:
        return self.snapshot_count == 0

    @property
    def latest_change(self) -> ProgressChange | None:
        return self.recent_changes[-1] if self.recent_changes else None


@dataclass(frozen=True)
class LifecycleSummary:
    """Frozen capture of a ``TaskLifecycle``.

    Timestamps are deliberately excluded so identical task behaviour
    produces identical reports.
    """

    task: str
    status: str
    is_terminal: bool
    current_iteration: int
    action_count: int
    error_count: int
    latest_error: str | None
    transition_count: int
    latest_transition_reason: str | None

    @classmethod
    def from_lifecycle(cls, lifecycle: TaskLifecycle) -> LifecycleSummary:
        if not isinstance(lifecycle, TaskLifecycle):
            raise TypeError("task_lifecycle must be a TaskLifecycle")

        latest_transition = (
            lifecycle.transitions[-1] if lifecycle.transitions else None
        )

        return cls(
            task=lifecycle.task,
            status=lifecycle.status,
            is_terminal=lifecycle.is_terminal,
            current_iteration=lifecycle.current_iteration,
            action_count=len(lifecycle.actions),
            error_count=len(lifecycle.errors),
            latest_error=lifecycle.errors[-1] if lifecycle.errors else None,
            transition_count=len(lifecycle.transitions),
            latest_transition_reason=(
                latest_transition.reason if latest_transition else None
            ),
        )


# ==================================================================
# PROGRESS REPORT
# ==================================================================


@dataclass(frozen=True)
class ProgressReport:
    """Immutable, deterministic report of task progress and verification.

    ``summary`` and every convenience property are derived from the
    stored sections, so they cannot disagree with them.
    """

    task: str
    progress: TaskProgressMetrics
    history: HistorySummary | None
    evidence: CompletionEvidence | None
    lifecycle: LifecycleSummary | None
    agent_marked_finished: bool
    human_intervention_reasons: tuple[str, ...]
    inconsistencies: tuple[str, ...]

    # --------------------------------------------------------------
    # A. Task progress
    # --------------------------------------------------------------

    @property
    def intent(self) -> str:
        return self.progress.intent

    @property
    def completion_fraction(self) -> float:
        return self.progress.completion_fraction

    @property
    def percent(self) -> int:
        return self.progress.percent

    @property
    def achieved_count(self) -> int:
        return self.progress.achieved_count

    @property
    def applicable_count(self) -> int:
        return self.progress.applicable_count

    @property
    def achieved_milestones(self) -> tuple[str, ...]:
        return tuple(m.key for m in self.progress.milestones if m.achieved)

    @property
    def pending_milestones(self) -> tuple[str, ...]:
        return tuple(m.key for m in self.progress.milestones if m.pending)

    @property
    def next_milestone(self) -> str | None:
        milestone = self.progress.next_milestone

        return milestone.key if milestone is not None else None

    # --------------------------------------------------------------
    # B–D. Availability and verification
    # --------------------------------------------------------------

    @property
    def history_available(self) -> bool:
        return self.history is not None

    @property
    def evidence_available(self) -> bool:
        return self.evidence is not None

    @property
    def lifecycle_available(self) -> bool:
        return self.lifecycle is not None

    @property
    def verification_status(self) -> str | None:
        """Return the evidence status, or None when evidence is unavailable."""

        return self.evidence.status if self.evidence is not None else None

    @property
    def verified(self) -> bool:
        """True only when supplied evidence explicitly reports VERIFIED."""

        return self.evidence is not None and self.evidence.verified

    @property
    def human_intervention_required(self) -> bool:
        return bool(self.human_intervention_reasons)

    @property
    def consistent(self) -> bool:
        return not self.inconsistencies

    # --------------------------------------------------------------
    # E. Human-readable summary
    # --------------------------------------------------------------

    @property
    def summary(self) -> tuple[str, ...]:
        """Answer the six reporting questions in deterministic wording."""

        return (
            self._accomplished_line(),
            self._remaining_line(),
            self._progressing_line(),
            self._regression_line(),
            self._verification_line(),
            self._human_line(),
        )

    def _accomplished_line(self) -> str:
        if self.applicable_count == 0:
            return (
                "Accomplished: no applicable milestones are defined for "
                "this task."
            )

        if not self.achieved_milestones:
            return "Accomplished: no milestones achieved yet."

        return (
            f"Accomplished: {self.achieved_count}/{self.applicable_count} "
            f"applicable milestone(s) — {', '.join(self.achieved_milestones)}."
        )

    def _remaining_line(self) -> str:
        if self.applicable_count == 0:
            return "Remaining: unknown; no applicable milestones exist."

        if not self.pending_milestones:
            return "Remaining: no pending milestones."

        return (
            f"Remaining: {', '.join(self.pending_milestones)} "
            f"(next: {self.next_milestone})."
        )

    def _progressing_line(self) -> str:
        if self.history is None:
            return "Progressing: unknown; no progress history was supplied."

        if self.history.is_empty:
            return "Progressing: unknown; progress history is empty."

        change = self.history.latest_change

        if change is None:
            return (
                "Progressing: unknown; only one snapshot "
                f"(iteration {self.history.latest.iteration}) is recorded."
            )

        span = f"iteration {change.before.iteration} → {change.after.iteration}"

        if change.direction == PROGRESS_INCREASED:
            return (
                f"Progressing: yes; {_percent_points(change.delta)} "
                f"({span})."
            )

        if change.direction == PROGRESS_DECREASED:
            return (
                f"Progressing: no; progress fell "
                f"{_percent_points(change.delta)} ({span})."
            )

        return f"Progressing: no change ({span})."

    def _regression_line(self) -> str:
        if self.history is None:
            return "Regressions: unknown; no progress history was supplied."

        if self.history.is_empty:
            return "Regressions: unknown; progress history is empty."

        net = _percent_points(self.history.net_progress)

        if self.history.regression_count == 0:
            return f"Regressions: none recorded; net progress {net}."

        line = (
            f"Regressions: {self.history.regression_count} recorded; "
            f"net progress {net}"
        )

        if self.history.below_peak:
            peak = self.history.peak
            line += (
                f"; current progress is below the peak of {peak.percent}% "
                f"(iteration {peak.iteration})"
            )

        return line + "."

    def _verification_line(self) -> str:
        if self.evidence is None:
            return (
                "Verified: unknown; no completion evidence was supplied, "
                "so completion cannot be claimed."
            )

        if self.evidence.verified:
            return (
                "Verified: yes, by "
                f"{self.evidence.verification_method}."
            )

        reason = (
            f" {self.evidence.reasons[-1]}" if self.evidence.reasons else ""
        )

        return f"Verified: no ({self.evidence.status}).{reason}"

    def _human_line(self) -> str:
        if self.human_intervention_reasons:
            return (
                "Human intervention: required — "
                + "; ".join(self.human_intervention_reasons)
                + "."
            )

        return (
            "Human intervention: not indicated by the available "
            "information."
        )

    def render(self) -> str:
        """Return the full report as deterministic plain text."""

        if self.evidence is None:
            headline = "VERIFICATION UNAVAILABLE"
        elif self.evidence.verified:
            headline = "VERIFIED"
        else:
            headline = f"NOT VERIFIED ({self.evidence.status})"

        lines = [
            f"FLY-CODER progress report — {self.task}",
            f"Intent: {self.intent} | Progress: {self.percent}% | "
            f"Status: {headline}",
            "",
            *self.summary,
            "",
            *self.progress.explain(),
        ]

        if self.history is not None and self.history.recent_changes:
            lines.append("")
            lines.append("Recent progress changes:")

            for change in self.history.recent_changes:
                lines.append(
                    f"  iteration {change.before.iteration} → "
                    f"{change.after.iteration}: {change.direction} "
                    f"({change.before.percent}% → {change.after.percent}%)"
                )

        if self.evidence is not None:
            lines.append("")
            lines.append(f"Completion evidence: {self.evidence.status}")
            lines.extend(f"  - {r}" for r in self.evidence.reasons)
            lines.extend(
                f"  ! unresolved: {e}" for e in self.evidence.unresolved_errors
            )

        if self.lifecycle is not None:
            lines.append("")
            lines.append(
                f"Lifecycle: {self.lifecycle.status} | iteration "
                f"{self.lifecycle.current_iteration} | "
                f"{self.lifecycle.action_count} action(s) | "
                f"{self.lifecycle.error_count} error(s)"
            )

        if self.agent_marked_finished:
            lines.append("")
            lines.append(
                "Note: the agent marked the task finished; this is not "
                "verification."
            )

        if self.inconsistencies:
            lines.append("")
            lines.append("Inconsistencies:")
            lines.extend(f"  ! {item}" for item in self.inconsistencies)

        return "\n".join(lines)


# ==================================================================
# REPORT CONSTRUCTION
# ==================================================================


def _human_intervention_reasons(
    state: CodingState,
    progress: TaskProgressMetrics,
    evidence: CompletionEvidence | None,
    lifecycle: LifecycleSummary | None,
) -> tuple[str, ...]:
    reasons: list[str] = []

    if state.user_input_needed:
        reasons.append("the task state is waiting for human input")

    if progress.awaiting_human:
        reasons.append(
            f"milestone '{progress.next_milestone.key}' requires explicit "
            "human approval"
        )

    if evidence is not None and evidence.needs_approval:
        reasons.append("completion evidence requires repair approval")

    if evidence is not None and evidence.status == EVIDENCE_INCONSISTENT:
        reasons.append("completion evidence is inconsistent and needs review")

    if lifecycle is not None and lifecycle.status == TASK_NEEDS_HUMAN:
        reasons.append(f"the lifecycle status is {TASK_NEEDS_HUMAN}")

    return tuple(reasons)


def _inconsistencies(
    state: CodingState,
    progress: TaskProgressMetrics,
    history: HistorySummary | None,
    evidence: CompletionEvidence | None,
    lifecycle: LifecycleSummary | None,
) -> tuple[str, ...]:
    items: list[str] = []

    if evidence is not None:
        if evidence.intent != progress.intent:
            items.append(
                f"Completion evidence is for intent '{evidence.intent}' but "
                f"progress is for '{progress.intent}'."
            )

        if evidence.code_signature != build_code_signature(state):
            items.append(
                "Completion evidence was evaluated for different code than "
                "the current state; re-evaluate before relying on it."
            )

        if progress.all_milestones_achieved and not evidence.verified:
            items.append(
                "All applicable milestones are achieved, but completion "
                f"evidence is {evidence.status}; progress is not proof."
            )

        if evidence.verified and not progress.all_milestones_achieved:
            items.append(
                "Completion evidence is VERIFIED, but "
                f"{progress.pending_count} milestone(s) are still pending "
                "in current progress."
            )

    if history is not None and history.latest is not None:
        if (
            abs(history.latest.completion_fraction
                - progress.completion_fraction)
            > FRACTION_TOLERANCE
        ):
            items.append(
                f"Latest history snapshot (iteration "
                f"{history.latest.iteration}, {history.latest.percent}%) "
                f"does not match current progress ({progress.percent}%)."
            )

    if lifecycle is not None:
        if lifecycle.task != state.task:
            items.append(
                f"Lifecycle task {lifecycle.task!r} does not match state "
                f"task {state.task!r}."
            )

        if lifecycle.status == TASK_COMPLETED and not (
            evidence is not None and evidence.verified
        ):
            items.append(
                "Lifecycle reports COMPLETED, but no completion evidence "
                "verifies the task."
            )

        if (
            lifecycle.is_terminal
            and lifecycle.status != TASK_COMPLETED
            and evidence is not None
            and evidence.verified
        ):
            items.append(
                f"Completion evidence is VERIFIED, but the lifecycle ended "
                f"as {lifecycle.status}."
            )

    if state.finished and not (evidence is not None and evidence.verified):
        items.append(
            "The agent marked the task finished without verified "
            "completion evidence."
        )

    return tuple(items)


def build_progress_report(
    state: CodingState,
    history: ProgressHistory | None = None,
    completion_evidence: CompletionEvidence | None = None,
    task_lifecycle: TaskLifecycle | None = None,
    *,
    intent: str | None = None,
) -> ProgressReport:
    """Build a read-only progress report for the current task.

    Args:
        state: Current task state; read, never modified.
        history: Optional task-level ``ProgressHistory``.
        completion_evidence: Optional evidence from
            ``evaluate_completion_evidence``. It is reported as supplied
            and never recalculated or replaced.
        task_lifecycle: Optional ``TaskLifecycle``.
        intent: Task intent; defaults to ``state.task_intent``. Callers
            that observe a task before the agent classifies it (the
            autonomous loop) pass the intent explicitly.

    Raises:
        ValueError: if the task intent is missing or unknown.
        TypeError: if an optional input has the wrong type, including an
            iteration-level ProgressTracker passed as history.
    """

    if not isinstance(state, CodingState):
        raise TypeError("state must be a CodingState")

    if completion_evidence is not None and not isinstance(
        completion_evidence,
        CompletionEvidence,
    ):
        raise TypeError("completion_evidence must be CompletionEvidence")

    progress = measure_task_progress(state, intent)

    history_summary = (
        HistorySummary.from_history(history) if history is not None else None
    )
    lifecycle_summary = (
        LifecycleSummary.from_lifecycle(task_lifecycle)
        if task_lifecycle is not None
        else None
    )

    return ProgressReport(
        task=state.task,
        progress=progress,
        history=history_summary,
        evidence=completion_evidence,
        lifecycle=lifecycle_summary,
        agent_marked_finished=state.finished,
        human_intervention_reasons=_human_intervention_reasons(
            state,
            progress,
            completion_evidence,
            lifecycle_summary,
        ),
        inconsistencies=_inconsistencies(
            state,
            progress,
            history_summary,
            completion_evidence,
            lifecycle_summary,
        ),
    )
