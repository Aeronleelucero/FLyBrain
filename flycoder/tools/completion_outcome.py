"""Fine-grained completion outcomes for FLY-CODER Phase 10.8.

Three layers describe how a task ended. They answer different questions
and are deliberately kept separate:

* ``TaskLifecycle`` status (``task_lifecycle``): where the task is in its
  state machine (RUNNING, COMPLETED, FAILED, NEEDS_HUMAN, ...). FAILED is
  coarse: it covers every terminal end that is not success, block, or
  cancellation.
* Stop status (``AUTONOMOUS_*`` below): why one autonomous run stopped
  (the agent finished, an action failed, the iteration budget ran out,
  a guardrail blocked an action, ...).
* Completion outcome (``OUTCOME_*`` below): what the run achieved,
  judged from evidence. It is derived from the two layers above plus
  completion evidence, task progress, verification runs, and recovery
  history, and it never changes any of them.

``classify_completion_outcome`` is the single place that decides the
outcome. ``success_blockers`` is the single place that decides whether
success may be claimed; the autonomous loop uses it to gate COMPLETED,
so the loop and the classifier cannot disagree.

Safety decisions
----------------
* Only VERIFIED_COMPLETE is successful. It requires VERIFIED completion
  evidence, and the latest verification run (if any) must have passed:
  a failed verification is never turned into success without a newer
  verification that passed.
* Cancellation is checked first and can never become success.
* A pending human decision (AWAITING_HUMAN) is not terminal and stays
  distinct from the terminal BLOCKED outcome.
* Budget exhaustion is its own outcome even when evidence happens to be
  verified: the agent did not conclude the task.
* Success is never inferred from the absence of errors; missing or stale
  evidence is reported as such.

Classification is deterministic and read-only: it never mutates state,
executes actions, runs tests, approves repairs, or changes the lifecycle.

Limitation: the autonomous loop has an iteration budget only. There is
no wall-clock time budget, so BUDGET_EXHAUSTED always means iterations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.completion_evidence import (
    EVIDENCE_INCONSISTENT,
    EVIDENCE_VERIFIED,
    CompletionEvidence,
    evaluate_completion_evidence,
)
from flycoder.tools.human_input import describe_human_input
from flycoder.tools.task_lifecycle import (
    TASK_BLOCKED,
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_INITIALIZING,
    TASK_NEEDS_HUMAN,
    TASK_PENDING,
    TASK_RUNNING,
    TASK_VERIFYING,
)
from flycoder.tools.task_progress import (
    FILES_SELECTED,
    REPAIR_APPLIED,
    WORKSPACE_INSPECTED,
    MilestoneStatus,
    measure_task_progress,
)


# ==================================================================
# STOP STATUSES (why one autonomous run stopped)
#
# Defined here so the classifier can use them without importing the
# loop; ``flycoder.tools.autonomous`` re-exports them unchanged.
# ==================================================================

AUTONOMOUS_COMPLETED = "COMPLETED"
AUTONOMOUS_FAILED = "FAILED"
AUTONOMOUS_BLOCKED = "BLOCKED"
AUTONOMOUS_NEEDS_HUMAN = "NEEDS_HUMAN"
AUTONOMOUS_MAX_ITERATIONS = "MAX_ITERATIONS"

# The agent stopped (``state.finished``) but completion evidence does not
# verify the task. ``AutonomousTaskResult.completion_evidence.status``
# says why (FAILED, MISSING, STALE, INCOMPLETE or INCONSISTENT).
AUTONOMOUS_UNVERIFIED = "UNVERIFIED"

# A human cancelled the task during an intervention (Phase 10.7).
AUTONOMOUS_CANCELLED = "CANCELLED"

STOP_STATUSES = frozenset(
    {
        AUTONOMOUS_COMPLETED,
        AUTONOMOUS_FAILED,
        AUTONOMOUS_BLOCKED,
        AUTONOMOUS_NEEDS_HUMAN,
        AUTONOMOUS_MAX_ITERATIONS,
        AUTONOMOUS_UNVERIFIED,
        AUTONOMOUS_CANCELLED,
    }
)

# Agent action that runs the Phase 7 verification pipeline (Phase 10.5).
VERIFICATION_ACTION = "verify_changes"

LIFECYCLE_STATUSES = frozenset(
    {
        TASK_PENDING,
        TASK_INITIALIZING,
        TASK_RUNNING,
        TASK_VERIFYING,
        TASK_COMPLETED,
        TASK_FAILED,
        TASK_BLOCKED,
        TASK_NEEDS_HUMAN,
        TASK_CANCELLED,
    }
)


# ==================================================================
# OUTCOME TAXONOMY
# ==================================================================

OUTCOME_VERIFIED_COMPLETE = "VERIFIED_COMPLETE"
OUTCOME_PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
OUTCOME_BLOCKED = "BLOCKED"
OUTCOME_EXECUTION_FAILED = "EXECUTION_FAILED"
OUTCOME_VERIFICATION_FAILED = "VERIFICATION_FAILED"
OUTCOME_CANCELLED = "CANCELLED"
OUTCOME_AWAITING_HUMAN = "AWAITING_HUMAN"
OUTCOME_BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
OUTCOME_IN_PROGRESS = "IN_PROGRESS"


@dataclass(frozen=True)
class OutcomeDefinition:
    """Static properties of one completion outcome."""

    identifier: str
    description: str
    terminal: bool
    resumable: bool
    verification_required: bool
    required_evidence: str
    successful: bool
    human_action_required: bool
    expected_lifecycle: frozenset[str]
    next_step: str


OUTCOME_DEFINITIONS: dict[str, OutcomeDefinition] = {
    definition.identifier: definition
    for definition in (
        OutcomeDefinition(
            identifier=OUTCOME_VERIFIED_COMPLETE,
            description="The requested work was completed and verified.",
            terminal=True,
            resumable=False,
            verification_required=True,
            required_evidence=(
                "VERIFIED completion evidence for the current code, and "
                "no failed verification run without a newer passing one."
            ),
            successful=True,
            human_action_required=False,
            expected_lifecycle=frozenset({TASK_COMPLETED}),
            next_step="No action required.",
        ),
        OutcomeDefinition(
            identifier=OUTCOME_PARTIALLY_COMPLETED,
            description=(
                "Some of the requested work was done, but the task was not "
                "completed and verified."
            ),
            terminal=True,
            resumable=False,
            verification_required=False,
            required_evidence=(
                "Meaningful progress toward the requested work — an "
                "analyzed relevant file, a test run on the current code, a "
                "diagnosed failure, a repair proposal, or an applied "
                "approved change; workspace discovery and automatic file "
                "selection alone do not count — and completion evidence "
                "that is not VERIFIED and shows no failed verification of "
                "the task's work."
            ),
            successful=False,
            human_action_required=False,
            expected_lifecycle=frozenset({TASK_FAILED}),
            next_step=(
                "Review the remaining work and run the task again to "
                "finish and verify it."
            ),
        ),
        OutcomeDefinition(
            identifier=OUTCOME_BLOCKED,
            description=(
                "Execution cannot continue until an unresolved condition is "
                "fixed (a guardrail blocked an action, or the task state or "
                "evidence is inconsistent)."
            ),
            terminal=True,
            resumable=False,
            verification_required=False,
            required_evidence=(
                "A guardrail-blocked action, INCONSISTENT completion "
                "evidence, or a contradictory completion record."
            ),
            successful=False,
            human_action_required=True,
            expected_lifecycle=frozenset({TASK_BLOCKED, TASK_FAILED}),
            next_step=(
                "Resolve the blocking condition listed under unresolved "
                "issues, then run the task again."
            ),
        ),
        OutcomeDefinition(
            identifier=OUTCOME_EXECUTION_FAILED,
            description=(
                "Execution ended without producing the requested work: an "
                "action failed with no recovery, execution stopped making "
                "progress, or the agent stopped without meaningful progress "
                "toward the requested work."
            ),
            terminal=True,
            resumable=False,
            verification_required=False,
            required_evidence=(
                "A failed run (failed action without recovery, or a stop "
                "decision), or an unverified finish without meaningful "
                "progress (workspace discovery and file selection alone "
                "are not progress)."
            ),
            successful=False,
            human_action_required=False,
            expected_lifecycle=frozenset({TASK_FAILED}),
            next_step="Inspect the failure, then run the task again.",
        ),
        OutcomeDefinition(
            identifier=OUTCOME_VERIFICATION_FAILED,
            description=(
                "The task's work was verified and verification failed."
            ),
            terminal=True,
            resumable=False,
            verification_required=True,
            required_evidence=(
                "FAILED completion evidence for the current code after the "
                "task's work (an applied change, or the requested test run)."
            ),
            successful=False,
            human_action_required=False,
            expected_lifecycle=frozenset({TASK_FAILED}),
            next_step=(
                "Inspect the failing verification; the work is not "
                "verified."
            ),
        ),
        OutcomeDefinition(
            identifier=OUTCOME_CANCELLED,
            description="A human cancelled the task.",
            terminal=True,
            resumable=False,
            verification_required=False,
            required_evidence="A recorded human cancellation.",
            successful=False,
            human_action_required=False,
            expected_lifecycle=frozenset({TASK_CANCELLED}),
            next_step="None; start a new task to continue the work.",
        ),
        OutcomeDefinition(
            identifier=OUTCOME_AWAITING_HUMAN,
            description=(
                "Execution is paused until a human records a decision."
            ),
            terminal=False,
            resumable=True,
            verification_required=False,
            required_evidence=(
                "The task is paused at the human boundary "
                "(NEEDS_HUMAN)."
            ),
            successful=False,
            human_action_required=True,
            expected_lifecycle=frozenset({TASK_NEEDS_HUMAN}),
            next_step=(
                "Review the pending intervention and record a decision; "
                "the task can then resume."
            ),
        ),
        OutcomeDefinition(
            identifier=OUTCOME_BUDGET_EXHAUSTED,
            description=(
                "The iteration budget ran out before the agent concluded "
                "the task."
            ),
            terminal=True,
            resumable=False,
            verification_required=False,
            required_evidence="The run reached its iteration limit.",
            successful=False,
            human_action_required=False,
            expected_lifecycle=frozenset({TASK_FAILED}),
            next_step="Run the task again with a larger iteration budget.",
        ),
        OutcomeDefinition(
            identifier=OUTCOME_IN_PROGRESS,
            description="The task is still running.",
            terminal=False,
            resumable=False,
            verification_required=False,
            required_evidence="A non-terminal lifecycle status.",
            successful=False,
            human_action_required=False,
            expected_lifecycle=frozenset(
                {TASK_PENDING, TASK_INITIALIZING, TASK_RUNNING, TASK_VERIFYING}
            ),
            next_step="Wait for the run to stop.",
        ),
    )
}

OUTCOMES = frozenset(OUTCOME_DEFINITIONS)


# ==================================================================
# VERIFICATION RUNS
# ==================================================================


def guardrail_blocked(result: ActionResult) -> bool:
    """Return whether a result was blocked by a guardrail."""

    if not isinstance(result.data, dict):
        return False

    guardrail = result.data.get("guardrail")

    if not isinstance(guardrail, dict):
        return False

    return guardrail.get("allowed") is False


@dataclass(frozen=True)
class VerificationAttempt:
    """One executed run of the verification pipeline."""

    iteration: int
    passed: bool
    evidence_status: str | None = None


def verification_attempt(
    iteration: int,
    result: ActionResult,
    evidence: CompletionEvidence | None = None,
) -> VerificationAttempt | None:
    """Return the verification run a result represents, if any.

    Only ``verify_changes`` results that actually ran (not blocked by a
    guardrail) count.
    """

    if result.action != VERIFICATION_ACTION or guardrail_blocked(result):
        return None

    return VerificationAttempt(
        iteration=iteration,
        passed=bool(result.success),
        evidence_status=evidence.status if evidence is not None else None,
    )


# ==================================================================
# SUCCESS GATE
# ==================================================================


def success_blockers(
    evidence: CompletionEvidence,
    verification_attempts: Sequence[VerificationAttempt] = (),
) -> tuple[str, ...]:
    """Return why success cannot be claimed; empty means it may be.

    This is the only success gate. The autonomous loop uses it before
    transitioning to COMPLETED, and the classifier uses it before
    returning VERIFIED_COMPLETE.
    """

    if not isinstance(evidence, CompletionEvidence):
        raise TypeError("evidence must be CompletionEvidence")

    blockers: list[str] = []

    if not evidence.verified:
        blockers.append(f"Completion evidence is {evidence.status}.")

    if verification_attempts and not verification_attempts[-1].passed:
        latest = verification_attempts[-1]
        blockers.append(
            f"The latest verification run (iteration {latest.iteration}) "
            "failed and no newer verification passed."
        )

    return tuple(blockers)


# ==================================================================
# COMPLETION OUTCOME
# ==================================================================


@dataclass(frozen=True)
class CompletionOutcome:
    """Immutable, evidence-based description of how a task ended."""

    outcome: str
    stop_status: str | None
    lifecycle_status: str | None
    evidence_status: str
    intent: str
    verified: bool
    completed_work: tuple[str, ...]
    remaining_work: tuple[str, ...]
    unresolved_issues: tuple[str, ...]
    verification_attempts: tuple[VerificationAttempt, ...]
    recovery_attempts: int
    human_action_required: bool
    can_resume: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"Unknown completion outcome: {self.outcome!r}")

        if self.stop_status is not None and self.stop_status not in STOP_STATUSES:
            raise ValueError(f"Unknown stop status: {self.stop_status!r}")

        if (
            self.lifecycle_status is not None
            and self.lifecycle_status not in LIFECYCLE_STATUSES
        ):
            raise ValueError(
                f"Unknown lifecycle status: {self.lifecycle_status!r}"
            )

        if self.verified != (self.evidence_status == EVIDENCE_VERIFIED):
            raise ValueError("verified must match the evidence status")

        if self.can_resume and not self.definition.resumable:
            raise ValueError(f"{self.outcome} cannot be resumed")

        if self.recovery_attempts < 0:
            raise ValueError("recovery_attempts must be non-negative")

        for name in (
            "completed_work",
            "remaining_work",
            "unresolved_issues",
            "verification_attempts",
            "reasons",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

        # Hard safety invariants: success is never claimed without
        # verified evidence, a passing latest verification run, and a
        # matching completion record.
        if self.successful:
            problems = []

            if not self.verified:
                problems.append("evidence is not VERIFIED")

            if (
                self.verification_attempts
                and not self.verification_attempts[-1].passed
            ):
                problems.append("the latest verification run failed")

            if self.lifecycle_status not in (None, TASK_COMPLETED):
                problems.append(
                    f"the lifecycle ended as {self.lifecycle_status}"
                )

            if self.stop_status not in (None, AUTONOMOUS_COMPLETED):
                problems.append(f"the run stopped as {self.stop_status}")

            if problems:
                raise ValueError(
                    "A successful outcome is contradictory: "
                    + "; ".join(problems)
                )

    @property
    def definition(self) -> OutcomeDefinition:
        return OUTCOME_DEFINITIONS[self.outcome]

    @property
    def successful(self) -> bool:
        return self.definition.successful

    @property
    def terminal(self) -> bool:
        return self.definition.terminal

    @property
    def description(self) -> str:
        return self.definition.description

    @property
    def next_step(self) -> str:
        """The recommended next step for this specific outcome.

        Presentation only: a resumable outcome that cannot be resumed
        here (no session, closed session, or no budget) recommends
        running the task again instead of resuming it.
        """

        if self.definition.resumable and not self.can_resume:
            return (
                "Run the task again; it cannot be resumed here. "
                "Review the pending intervention first."
            )

        return self.definition.next_step

    @property
    def verification_failures(self) -> int:
        return sum(1 for attempt in self.verification_attempts if not attempt.passed)

    def render_lines(self) -> tuple[str, ...]:
        """Return a concise, deterministic outcome report."""

        lines = [
            f"Outcome: {self.outcome} — {self.description}",
            "Successful: " + ("yes" if self.successful else "no"),
        ]

        status_parts = []

        if self.lifecycle_status is not None:
            status_parts.append(f"final lifecycle status {self.lifecycle_status}")

        if self.stop_status is not None:
            status_parts.append(f"run stopped as {self.stop_status}")

        if status_parts:
            lines.append("Status: " + "; ".join(status_parts) + ".")

        lines.append(
            "Work verified: "
            + ("yes" if self.verified else f"no (evidence {self.evidence_status})")
        )

        if self.verification_attempts:
            latest = self.verification_attempts[-1]
            lines.append(
                f"Verification runs: {len(self.verification_attempts)} "
                f"({self.verification_failures} failed); latest "
                f"{'passed' if latest.passed else 'failed'} at iteration "
                f"{latest.iteration}."
            )

        if self.recovery_attempts:
            lines.append(f"Recovery attempts: {self.recovery_attempts}")

        lines.append(
            "Completed: "
            + (", ".join(self.completed_work) if self.completed_work else "nothing")
        )
        lines.append(
            "Remaining: "
            + (", ".join(self.remaining_work) if self.remaining_work else "nothing")
        )

        if self.unresolved_issues:
            lines.append("Unresolved issues:")
            lines.extend(f"  ! {issue}" for issue in self.unresolved_issues)

        lines.append(
            "Human action required: "
            + ("yes" if self.human_action_required else "no")
        )

        if self.can_resume:
            lines.append("Can resume: yes, after a human decision.")
        elif self.definition.resumable:
            lines.append(
                "Can resume: no — no resumable session or iteration budget "
                "is available here."
            )
        else:
            lines.append("Can resume: no")

        lines.append(f"Next step: {self.next_step}")

        return tuple(lines)

    def render(self) -> str:
        return "\n".join(self.render_lines())


# ==================================================================
# CLASSIFICATION
# ==================================================================


# Milestones that are groundwork, not progress toward the requested work.
# ``workspace_inspected`` only records that ``inspect_files`` listed the
# workspace. ``files_selected`` is set inside the agent's ``decide()``
# without any executed action, and falls back to the first files listed
# when nothing matches. Both stay visible in ``completed_work``, but
# neither makes a stopped, unverified task "partially completed".
#
# For ``inspect`` tasks discovery *is* the defined completion criterion:
# with files discovered the evidence is VERIFIED (task observation), so an
# inspection never needs a partial outcome. The agent records no deeper
# inspection evidence, so no further inspect milestone is assumed here.
INCIDENTAL_MILESTONES = frozenset({WORKSPACE_INSPECTED, FILES_SELECTED})


def meaningful_progress(
    milestones: Sequence[MilestoneStatus],
    state: CodingState,
) -> tuple[str, ...]:
    """Return the evidence of meaningful progress toward the task.

    A milestone counts when it is not incidental and is achieved, or is
    partly achieved (a review that analyzed some but not all relevant
    files). Analysis only counts after a successful analysis action, and
    tests, diagnosis, proposals, and applied changes are all recorded by
    executed actions or a human-approved change. An applied repair
    counts even for intents whose milestones do not track it.
    """

    items = [
        milestone.key
        for milestone in milestones
        if milestone.key not in INCIDENTAL_MILESTONES
        and (
            milestone.achieved
            or (milestone.pending and milestone.fraction > 0)
        )
    ]

    if state.repair_applied and REPAIR_APPLIED not in items:
        items.append(REPAIR_APPLIED)

    return tuple(items)


def _verification_failed_for_work(
    state: CodingState,
    evidence: CompletionEvidence,
    attempts: tuple[VerificationAttempt, ...],
) -> bool:
    """Return whether the task's work was checked and the check failed.

    True when the latest verification run failed (no newer one passed),
    or when fresh evidence fails after the work: for a repair, failing
    tests before any change is applied are the original problem, not a
    failed verification; for a test task, the test run is the requested
    work, so its failure is a verification failure.
    """

    if attempts and not attempts[-1].passed:
        return True

    if not evidence.failed:
        return False

    return state.repair_applied or evidence.intent == "test"


def infer_stop_status(
    state: CodingState,
    evidence: CompletionEvidence,
    *,
    budget_exhausted: bool,
    verification_attempts: Sequence[VerificationAttempt] = (),
) -> str:
    """Infer a stop status for a run that did not use the autonomous loop.

    The command-line interfaces drive ``agent.run_once`` themselves; this
    maps where they stopped onto the loop's stop statuses with the same
    success gate.
    """

    if state.user_input_needed:
        return AUTONOMOUS_NEEDS_HUMAN

    if state.finished:
        if success_blockers(evidence, verification_attempts):
            return AUTONOMOUS_UNVERIFIED

        return AUTONOMOUS_COMPLETED

    if budget_exhausted:
        return AUTONOMOUS_MAX_ITERATIONS

    # Stopped early (for example after a human rejected the proposal)
    # without the agent concluding the task.
    return AUTONOMOUS_UNVERIFIED


def classify_stopped_run(
    state: CodingState,
    *,
    intent: str,
    budget_exhausted: bool,
    verification_attempts: Sequence[VerificationAttempt] = (),
    final_result: ActionResult | None = None,
) -> CompletionOutcome:
    """Classify a run that drove ``agent.run_once`` without the loop.

    Used by the command-line interfaces. They keep no lifecycle and no
    resumable session, so the outcome is classified from the state, its
    evidence, and where the run stopped, and resumption is unavailable.
    """

    evidence = evaluate_completion_evidence(state, intent=intent)
    attempts = tuple(verification_attempts)

    return classify_completion_outcome(
        state,
        evidence,
        stop_status=infer_stop_status(
            state,
            evidence,
            budget_exhausted=budget_exhausted,
            verification_attempts=attempts,
        ),
        verification_attempts=attempts,
        final_error=(
            final_result.message
            if final_result is not None and not final_result.success
            else None
        ),
        resume_available=False,
    )


def classify_completion_outcome(
    state: CodingState,
    evidence: CompletionEvidence,
    *,
    stop_status: str | None,
    lifecycle_status: str | None = None,
    verification_attempts: Sequence[VerificationAttempt] = (),
    recovery_attempts: int = 0,
    final_error: str | None = None,
    resume_available: bool = True,
) -> CompletionOutcome:
    """Classify how a task ended from its actual execution evidence.

    Args:
        state: Current task state; read, never modified.
        evidence: Completion evidence evaluated for ``state``.
        stop_status: Why the run stopped (``AUTONOMOUS_*``), or None for
            a run that has not stopped.
        lifecycle_status: Current ``TaskLifecycle`` status, if known.
        verification_attempts: Every verification run of the task so
            far, in order (across resumptions).
        recovery_attempts: Failures recorded in the recovery history.
        final_error: Message of the final failed action, if any.
        resume_available: Whether the caller can actually resume a
            paused task (a session with iterations left, and not
            restricted to cancellation).

    Raises:
        TypeError / ValueError: for invalid inputs.
    """

    if not isinstance(state, CodingState):
        raise TypeError("state must be a CodingState")

    if not isinstance(evidence, CompletionEvidence):
        raise TypeError("evidence must be CompletionEvidence")

    if stop_status is not None and stop_status not in STOP_STATUSES:
        raise ValueError(f"Unknown stop status: {stop_status!r}")

    if lifecycle_status is not None and lifecycle_status not in LIFECYCLE_STATUSES:
        raise ValueError(f"Unknown lifecycle status: {lifecycle_status!r}")

    attempts = tuple(verification_attempts)

    for attempt in attempts:
        if not isinstance(attempt, VerificationAttempt):
            raise TypeError("verification_attempts must be VerificationAttempts")

    progress = measure_task_progress(state, evidence.intent)
    achieved = [m for m in progress.milestones if m.achieved]

    completed_work = [m.key for m in achieved]

    if state.repair_applied and state.current_file:
        completed_work.append(f"approved change applied to {state.current_file}")

    remaining_work = [m.key for m in progress.milestones if m.pending]

    unresolved: list[str] = list(evidence.unresolved_errors)

    if final_error and final_error not in unresolved:
        unresolved.append(final_error)

    reasons: list[str] = []
    blockers = success_blockers(evidence, attempts)

    # --------------------------------------------------------------
    # Decide the outcome. Order matters: cancellation and pending human
    # decisions are explicit facts that no evidence can override.
    # --------------------------------------------------------------

    if lifecycle_status == TASK_CANCELLED or stop_status == AUTONOMOUS_CANCELLED:
        outcome = OUTCOME_CANCELLED
        reasons.append("A human cancelled the task; cancellation is never success.")

    elif (
        lifecycle_status == TASK_NEEDS_HUMAN
        or stop_status == AUTONOMOUS_NEEDS_HUMAN
    ):
        outcome = OUTCOME_AWAITING_HUMAN
        reasons.append("Execution is paused for an explicit human decision.")

        pending = describe_human_input(state)

        if pending and pending not in unresolved:
            unresolved.append(pending)

    elif (
        stop_status == AUTONOMOUS_COMPLETED
        or lifecycle_status == TASK_COMPLETED
    ):
        mismatched = []

        if stop_status not in (None, AUTONOMOUS_COMPLETED):
            mismatched.append(f"run stop status {stop_status}")

        if lifecycle_status not in (None, TASK_COMPLETED):
            mismatched.append(f"lifecycle status {lifecycle_status}")

        if blockers or mismatched:
            outcome = OUTCOME_BLOCKED
            reasons.append(
                "The completion record is contradictory, so success cannot "
                "be claimed."
            )
            reasons.extend(blockers)
            reasons.extend(
                f"Completion disagrees with the {item}." for item in mismatched
            )
            unresolved.append("The completion record must be reviewed.")
        else:
            outcome = OUTCOME_VERIFIED_COMPLETE
            reasons.append(
                "Completion evidence is VERIFIED "
                f"({evidence.verification_method})."
            )

    elif stop_status == AUTONOMOUS_BLOCKED or lifecycle_status == TASK_BLOCKED:
        outcome = OUTCOME_BLOCKED
        reasons.append("A safety guardrail blocked the next action.")

    elif stop_status == AUTONOMOUS_MAX_ITERATIONS:
        outcome = OUTCOME_BUDGET_EXHAUSTED
        reasons.append("The run reached its iteration limit.")

        if evidence.verified:
            reasons.append(
                "Evidence is VERIFIED, but the agent did not conclude the "
                "task within its budget; this is not reported as success."
            )

    elif evidence.status == EVIDENCE_INCONSISTENT:
        outcome = OUTCOME_BLOCKED
        reasons.append("Completion evidence is inconsistent and needs review.")
        unresolved.extend(
            reason
            for reason in evidence.reasons
            if reason.startswith("Inconsistent evidence")
            and reason not in unresolved
        )

    elif _verification_failed_for_work(state, evidence, attempts):
        outcome = OUTCOME_VERIFICATION_FAILED
        reasons.append(
            "Verification of the task's work failed on the current code."
        )

    elif stop_status == AUTONOMOUS_FAILED:
        outcome = OUTCOME_EXECUTION_FAILED
        reasons.append("Execution failed and could not continue.")

    elif stop_status == AUTONOMOUS_UNVERIFIED:
        reasons.append(
            "The agent stopped, but completion evidence is "
            f"{evidence.status}."
        )

        meaningful = meaningful_progress(progress.milestones, state)

        if meaningful:
            outcome = OUTCOME_PARTIALLY_COMPLETED
            reasons.append("Meaningful progress: " + ", ".join(meaningful) + ".")
        else:
            outcome = OUTCOME_EXECUTION_FAILED
            reasons.append(
                "No meaningful progress toward the requested work was made; "
                "workspace discovery or file selection alone is not partial "
                "completion."
            )

    else:
        outcome = OUTCOME_IN_PROGRESS
        reasons.append("The task has not stopped.")

    if progress.applicable_count:
        reasons.append(
            f"{progress.achieved_count}/{progress.applicable_count} "
            "applicable milestone(s) achieved."
        )

    definition = OUTCOME_DEFINITIONS[outcome]

    return CompletionOutcome(
        outcome=outcome,
        stop_status=stop_status,
        lifecycle_status=lifecycle_status,
        evidence_status=evidence.status,
        intent=evidence.intent,
        verified=evidence.verified,
        completed_work=tuple(completed_work),
        remaining_work=tuple(remaining_work),
        unresolved_issues=tuple(unresolved),
        verification_attempts=attempts,
        recovery_attempts=recovery_attempts,
        human_action_required=(
            definition.human_action_required or state.user_input_needed
        ),
        can_resume=definition.resumable and resume_available,
        reasons=tuple(reasons),
    )


def outcome_consistency_problems(
    outcome: CompletionOutcome,
    evidence: CompletionEvidence | None = None,
    lifecycle_status: str | None = None,
) -> tuple[str, ...]:
    """Return contradictions between an outcome and its sources.

    ``evidence`` and ``lifecycle_status`` are the values a report shows
    next to the outcome; they must be the ones it was classified from.
    """

    problems: list[str] = []

    if (
        lifecycle_status is not None
        and outcome.lifecycle_status is not None
        and lifecycle_status != outcome.lifecycle_status
    ):
        problems.append(
            f"Outcome was classified for lifecycle status "
            f"{outcome.lifecycle_status}, but the lifecycle is now "
            f"{lifecycle_status}."
        )

    if (
        outcome.lifecycle_status is not None
        and outcome.lifecycle_status not in outcome.definition.expected_lifecycle
    ):
        problems.append(
            f"Outcome {outcome.outcome} does not match lifecycle status "
            f"{outcome.lifecycle_status}."
        )

    if evidence is not None and evidence.status != outcome.evidence_status:
        problems.append(
            f"Outcome was classified with evidence {outcome.evidence_status}, "
            f"but the reported evidence is {evidence.status}."
        )

    return tuple(problems)
