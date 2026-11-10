"""Autonomous task orchestration for FLY-CODER Phase 10."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.confidence import (
    ConfidenceAssessment,
    assess_confidence,
)
from flycoder.tools.completion_evidence import (
    CompletionEvidence,
    VerificationRecord,
    evaluate_completion_evidence,
)
from flycoder.tools.completion_outcome import (
    AUTONOMOUS_BLOCKED,
    AUTONOMOUS_CANCELLED,
    AUTONOMOUS_COMPLETED,
    AUTONOMOUS_FAILED,
    AUTONOMOUS_MAX_ITERATIONS,
    AUTONOMOUS_NEEDS_HUMAN,
    AUTONOMOUS_UNVERIFIED,
    VERIFICATION_ACTION,
    CompletionOutcome,
    VerificationAttempt,
    classify_completion_outcome,
    guardrail_blocked,
    success_blockers,
    verification_attempt,
)
from flycoder.tools.execution import ExecutionObservation, observe_action
from flycoder.tools.iteration import (
    IterationObservation,
    observe_iteration,
)
from flycoder.tools.iteration_decision import (
    IterationDecision,
    decide_iteration,
)
from flycoder.tools.progress import ProgressTracker
from flycoder.tools.progress_history import (
    ProgressHistory,
    ProgressSnapshot as TaskProgressSnapshot,
)
from flycoder.tools.progress_report import (
    ProgressReport,
    build_progress_report,
)
from flycoder.tools.repetition import ActionRepetitionTracker
from flycoder.tools.recovery import (
    RecoveryAssessment,
    RecoveryDecision,
    RecoveryHistory,
    recover_from_failure,
)
from flycoder.tools.stall import StallAssessment, assess_stall
from flycoder.tools.task_lifecycle import (
    ALLOWED_TRANSITIONS,
    TASK_BLOCKED,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_INITIALIZING,
    TASK_NEEDS_HUMAN,
    TASK_RUNNING,
    TASK_VERIFYING,
    TaskLifecycle,
)

if TYPE_CHECKING:
    from flycoder.agent import FlyCoderAgent


# The stop statuses (AUTONOMOUS_*) and VERIFICATION_ACTION are defined in
# ``flycoder.tools.completion_outcome`` since Phase 10.8 and re-exported
# here unchanged. A stop status says why a run stopped; the completion
# outcome (``AutonomousTaskResult.outcome``) says what it achieved.

# Phase 10.9: action name of the synthetic step that records an iteration
# in which a component raised. It is never executed or registered.
COMPONENT_ERROR_ACTION = "component_error"

# Lifecycle states in which a run is still active. A run never returns or
# raises while its lifecycle is in one of them.
_ACTIVE_LIFECYCLE_STATES = frozenset(
    {TASK_INITIALIZING, TASK_RUNNING, TASK_VERIFYING}
)

# Attribute under which an interrupted run's result is attached to the
# original KeyboardInterrupt/SystemExit object before it is re-raised.
_ATTACHED_RESULT_ATTRIBUTE = "flycoder_autonomous_result"


@dataclass(frozen=True)
class ComponentError:
    """Diagnostic record of an exception raised during a run (Phase 10.9).

    Only the type name, message, and location are kept; the exception
    and its traceback are not stored. ``interrupted`` marks
    ``KeyboardInterrupt`` and ``SystemExit``.
    """

    type_name: str
    message: str
    stage: str
    iteration: int | None = None
    interrupted: bool = False

    @classmethod
    def from_exception(
        cls,
        error: BaseException,
        *,
        stage: str,
        iteration: int | None = None,
        interrupted: bool = False,
    ) -> ComponentError:
        return cls(
            type_name=type(error).__name__,
            message=str(error),
            stage=stage,
            iteration=iteration,
            interrupted=interrupted,
        )

    def describe(self) -> str:
        where = f"stage '{self.stage}'"

        if self.iteration is not None:
            where += f", iteration {self.iteration}"

        message = f": {self.message}" if self.message else ""

        return f"{self.type_name}{message} ({where})"


@dataclass
class ResumeContext:
    """Task context carried across a human intervention (Phase 10.7).

    Resuming continues the same lifecycle, progress history, and recovery
    history, and numbers iterations after ``iteration_offset`` so the
    histories stay chronological. Execution still happens only through
    ``run_autonomous_task`` and ``agent.run_once``.
    """

    lifecycle: TaskLifecycle
    progress_history: ProgressHistory
    recovery_history: RecoveryHistory
    iteration_offset: int
    reason: str = "Resumed after an explicit human decision."
    # Phase 10.8: steps of the earlier runs of this task, so verification
    # runs before the pause still count when the outcome is classified
    # (a failed verification is not forgotten by resuming).
    prior_steps: tuple[AutonomousStep, ...] = ()


@dataclass
class AutonomousStep:
    """Record one autonomous execution step."""

    iteration: int
    result: ActionResult
    observation: ExecutionObservation | None = None
    recovery: RecoveryDecision | None = None
    iteration_observation: IterationObservation | None = None
    stall: StallAssessment | None = None
    iteration_decision: IterationDecision | None = None
    # Phase 10.4.6: task-level progress and evidence after this step.
    progress_snapshot: TaskProgressSnapshot | None = None
    completion_evidence: CompletionEvidence | None = None
    # Phase 10.6: recovery history assessment (failures only) and the
    # Phase 9 decision confidence, both observational.
    recovery_assessment: RecoveryAssessment | None = None
    confidence: ConfidenceAssessment | None = None


@dataclass
class AutonomousTaskResult:
    """Final result of one autonomous task."""

    task: str
    status: str
    iterations: int
    steps: list[AutonomousStep] = field(default_factory=list)
    final_result: ActionResult | None = None
    stopped_reason: str = ""
    lifecycle: TaskLifecycle | None = None
    progress_history: ProgressHistory | None = None
    completion_evidence: CompletionEvidence | None = None
    progress_report: ProgressReport | None = None
    recovery_history: RecoveryHistory | None = None
    # Phase 10.8: evidence-based completion outcome of this result.
    # Phase 10.9: None only when evidence could not be evaluated for the
    # final state (a degraded result); such a result is never successful.
    outcome: CompletionOutcome | None = None
    # Phase 10.9: the exception that ended the run, if any, and
    # non-fatal diagnostics (observer failures, report failures).
    error: ComponentError | None = None
    diagnostics: tuple[str, ...] = ()

    @property
    def interrupted(self) -> bool:
        """Return whether KeyboardInterrupt/SystemExit ended the run."""

        return self.error is not None and self.error.interrupted

    @property
    def classified(self) -> bool:
        """Return whether an outcome could be classified from evidence."""

        return self.outcome is not None

    @property
    def successful(self) -> bool:
        """Return whether the outcome is verified completion.

        Only ``OUTCOME_VERIFIED_COMPLETE`` is successful; without a
        classified outcome nothing is reported as successful.
        """

        return self.outcome is not None and self.outcome.successful

    @property
    def completed(self) -> bool:
        """Return whether the task completed with verified evidence."""

        return self.status == AUTONOMOUS_COMPLETED

    @property
    def unverified(self) -> bool:
        """Return whether the agent stopped without verified evidence."""

        return self.status == AUTONOMOUS_UNVERIFIED

    @property
    def cancelled(self) -> bool:
        """Return whether a human cancelled the task."""

        return self.status == AUTONOMOUS_CANCELLED

    @property
    def failed(self) -> bool:
        return self.status == AUTONOMOUS_FAILED

    @property
    def blocked(self) -> bool:
        return self.status == AUTONOMOUS_BLOCKED

    @property
    def needs_human(self) -> bool:
        return self.status == AUTONOMOUS_NEEDS_HUMAN

    @property
    def reached_iteration_limit(self) -> bool:
        return self.status == AUTONOMOUS_MAX_ITERATIONS


def _execution_observation(
    result: ActionResult,
) -> ExecutionObservation | None:
    """Recover the execution observation attached by the agent."""

    if not isinstance(result.data, dict):
        return None

    data = result.data.get("execution_observation")

    if not isinstance(data, dict):
        return None

    return ExecutionObservation(
        action=str(data.get("action", result.action)),
        success=bool(data.get("success", result.success)),
        message=str(data.get("message", result.message)),
        next_action=data.get("next_action"),
        reason=str(data.get("reason", "")),
        risks=list(data.get("risks", [])),
        requires_human_input=bool(
            data.get("requires_human_input", False)
        ),
    )


def _guardrail_blocked(result: ActionResult) -> bool:
    """Return whether a result was blocked by a guardrail."""

    return guardrail_blocked(result)


def _verification_attempts(
    steps: list[AutonomousStep] | tuple[AutonomousStep, ...],
) -> tuple[VerificationAttempt, ...]:
    """Return the verification runs recorded in ``steps``, in order."""

    attempts = (
        verification_attempt(
            step.iteration,
            step.result,
            step.completion_evidence,
        )
        for step in steps
    )

    return tuple(attempt for attempt in attempts if attempt is not None)


def _record_step(
    lifecycle: TaskLifecycle,
    iteration: int,
    result: ActionResult,
) -> None:
    """Record action and iteration information in the lifecycle."""

    lifecycle.record_iteration(iteration)

    if result.action:
        lifecycle.record_action(result.action)

    if not result.success and result.message:
        lifecycle.record_error(result.message)


def _classify_outcome(
    state: CodingState,
    *,
    status: str,
    lifecycle: TaskLifecycle,
    evidence: CompletionEvidence,
    steps: tuple[AutonomousStep, ...],
    recovery_history: RecoveryHistory | None,
    final_result: ActionResult | None,
    resume_available: bool = True,
    lifecycle_status: str | None = None,
) -> CompletionOutcome:
    """Classify a task's outcome from its complete execution record.

    ``steps`` must be every step of the task so far (across human
    interventions), so earlier verification runs are never forgotten.
    ``lifecycle_status`` classifies for a status the lifecycle is about
    to reach (Phase 10.9 finalization classifies before transitioning);
    it defaults to the lifecycle's current status.
    """

    return classify_completion_outcome(
        state,
        evidence,
        stop_status=status,
        lifecycle_status=lifecycle_status or lifecycle.status,
        verification_attempts=_verification_attempts(steps),
        recovery_attempts=(
            len(recovery_history) if recovery_history is not None else 0
        ),
        final_error=(
            final_result.message
            if final_result is not None and not final_result.success
            else None
        ),
        resume_available=resume_available,
    )


# ==================================================================
# PHASE 10.9 — FINALIZATION AND FAILURE ISOLATION
# ==================================================================


def attach_result(
    error: BaseException,
    result: AutonomousTaskResult,
) -> None:
    """Attach an interrupted run's result to the original exception.

    The exception object itself is re-raised unchanged; callers that
    catch it can recover the finalized result with ``attached_result``.
    """

    try:
        setattr(error, _ATTACHED_RESULT_ATTRIBUTE, result)
    except (AttributeError, TypeError):
        pass


def attached_result(error: BaseException) -> AutonomousTaskResult | None:
    """Return the result attached to an interrupt, if any."""

    result = getattr(error, _ATTACHED_RESULT_ATTRIBUTE, None)

    return result if isinstance(result, AutonomousTaskResult) else None


def _path_valid(
    start: str,
    path: tuple[tuple[str, str], ...],
) -> bool:
    """Return whether every transition in ``path`` is allowed."""

    status = start

    for to_state, _ in path:
        if to_state not in ALLOWED_TRANSITIONS.get(status, set()):
            return False

        status = to_state

    return True


def _running_path(
    lifecycle_status: str,
    *transitions: tuple[str, str],
) -> tuple[tuple[str, str], ...]:
    """Return the transitions that end a run from RUNNING.

    Every stop decision is taken while the lifecycle is RUNNING. Any
    other status here is an internal inconsistency; raising lets the
    loop finalize the run as FAILED instead of leaving it active.
    """

    if lifecycle_status != TASK_RUNNING:
        raise RuntimeError(
            f"A stop decision was reached with lifecycle {lifecycle_status}."
        )

    return tuple(transitions)


def _failed_path(
    lifecycle_status: str,
    reason: str,
) -> tuple[tuple[str, str], ...]:
    """Return the transition that fails an active run, if any."""

    if lifecycle_status in _ACTIVE_LIFECYCLE_STATES:
        return ((TASK_FAILED, reason),)

    return ()


def _force_terminal(lifecycle: TaskLifecycle, reason: str) -> None:
    """Fail a lifecycle that is still active. Never raises."""

    try:
        if lifecycle.status in _ACTIVE_LIFECYCLE_STATES:
            lifecycle.transition(TASK_FAILED, reason)
    except Exception:
        pass


def _component_error_step(
    iteration: int,
    error: ComponentError,
) -> AutonomousStep:
    """Record an iteration in which a component raised.

    The synthetic result is never executed; it keeps the failing
    iteration in the history and the budget.
    """

    return AutonomousStep(
        iteration=iteration,
        result=ActionResult(
            action=COMPONENT_ERROR_ACTION,
            success=False,
            message=error.describe(),
            data={
                "component_error": {
                    "type": error.type_name,
                    "message": error.message,
                    "stage": error.stage,
                    "iteration": error.iteration,
                    "interrupted": error.interrupted,
                },
            },
        ),
    )


def _notify_observer(
    on_step: Callable[[AutonomousStep], None] | None,
    step: AutonomousStep,
    diagnostics: list[str],
) -> None:
    """Call a step observer; its failures are diagnostics only.

    An observer (for example a CLI printer) never changes the run: an
    ordinary exception it raises is recorded and execution continues.
    KeyboardInterrupt and SystemExit still propagate as interrupts.
    """

    if on_step is None:
        return

    try:
        on_step(step)
    except Exception as error:
        failure = ComponentError.from_exception(
            error,
            stage="observer",
            iteration=step.iteration,
        )
        diagnostics.append(f"Step observer failed: {failure.describe()}")


def _degraded_result(
    *,
    state: CodingState,
    lifecycle: TaskLifecycle,
    path: tuple[tuple[str, str], ...],
    status: str,
    stopped_reason: str,
    steps: list[AutonomousStep],
    history: ProgressHistory,
    recovery_history: RecoveryHistory | None,
    final_result: ActionResult | None,
    error: ComponentError | None,
    diagnostics: list[str],
) -> AutonomousTaskResult:
    """Build a result without evidence or outcome.

    Used only when evidence cannot be evaluated or classified for the
    final state. Nothing is evaluated here, so it cannot fail; the
    lifecycle still reaches a terminal status, and the result is never
    successful (``outcome`` is None).
    """

    if _path_valid(lifecycle.status, path):
        for to_state, reason in path:
            lifecycle.transition(to_state, reason)
    else:
        _force_terminal(lifecycle, stopped_reason)

    return AutonomousTaskResult(
        task=state.task,
        status=status,
        iterations=len(steps),
        steps=steps,
        final_result=final_result,
        stopped_reason=stopped_reason,
        lifecycle=lifecycle,
        progress_history=history,
        completion_evidence=None,
        progress_report=None,
        recovery_history=recovery_history,
        outcome=None,
        error=error,
        diagnostics=tuple(diagnostics),
    )


def _finalize(
    *,
    state: CodingState,
    lifecycle: TaskLifecycle,
    path: tuple[tuple[str, str], ...],
    status: str,
    stopped_reason: str,
    steps: list[AutonomousStep],
    history: ProgressHistory,
    recovery_history: RecoveryHistory | None,
    intent: str | None,
    evaluate_evidence: Callable[[], CompletionEvidence],
    final_result: ActionResult | None,
    prior_steps: tuple[AutonomousStep, ...] = (),
    error: ComponentError | None = None,
    diagnostics: list[str] | tuple[str, ...] = (),
    resume_available: bool = True,
    fallback: bool = False,
) -> AutonomousTaskResult:
    """End a run: the single place a result is produced (Phase 10.9).

    1. Evidence is evaluated and the outcome classified for the *target*
       lifecycle status, before any transition. A classification
       failure therefore never leaves a lifecycle COMPLETED (or paused)
       without a result.
    2. The transition ``path`` is applied.
    3. The progress report is built; a report failure is a diagnostic,
       because the report is not authoritative.

    If evaluation or classification fails, the run is finalized as
    FAILED instead (cancellation stays CANCELLED). If that fails too, a
    degraded result without evidence or outcome is returned. This
    function never raises an ordinary exception, and it never reports
    success the shared success gate has not allowed.
    """

    notes = list(diagnostics)

    try:
        if not _path_valid(lifecycle.status, path):
            raise ValueError(
                f"Invalid lifecycle path from {lifecycle.status}: "
                + " -> ".join(to_state for to_state, _ in path)
            )

        target = path[-1][0] if path else lifecycle.status
        evidence = evaluate_evidence()
        outcome = _classify_outcome(
            state,
            status=status,
            lifecycle=lifecycle,
            lifecycle_status=target,
            evidence=evidence,
            steps=tuple(prior_steps) + tuple(steps),
            recovery_history=recovery_history,
            final_result=final_result,
            resume_available=resume_available,
        )
    except Exception as exc:
        failure = ComponentError.from_exception(
            exc,
            stage="finalization",
            iteration=lifecycle.current_iteration or None,
        )
        notes.append(f"Finalization failed: {failure.describe()}")

        if not fallback and status not in (
            AUTONOMOUS_FAILED,
            AUTONOMOUS_CANCELLED,
        ):
            reason = (
                f"Finalization failed, so the run cannot be reported as "
                f"{status}: {failure.describe()}"
            )

            return _finalize(
                state=state,
                lifecycle=lifecycle,
                path=_failed_path(lifecycle.status, reason),
                status=AUTONOMOUS_FAILED,
                stopped_reason=reason,
                steps=steps,
                history=history,
                recovery_history=recovery_history,
                intent=intent,
                evaluate_evidence=evaluate_evidence,
                final_result=final_result,
                prior_steps=prior_steps,
                error=error or failure,
                diagnostics=notes,
                resume_available=resume_available,
                fallback=True,
            )

        return _degraded_result(
            state=state,
            lifecycle=lifecycle,
            path=path,
            status=status,
            stopped_reason=stopped_reason,
            steps=steps,
            history=history,
            recovery_history=recovery_history,
            final_result=final_result,
            error=error or failure,
            diagnostics=notes,
        )

    for to_state, reason in path:
        lifecycle.transition(to_state, reason)

    try:
        report = build_progress_report(
            state,
            history,
            evidence,
            lifecycle,
            intent=state.task_intent or intent,
            outcome=outcome,
        )
    except Exception as exc:
        report = None
        failure = ComponentError.from_exception(exc, stage="report")
        notes.append(f"Progress report unavailable: {failure.describe()}")

    return AutonomousTaskResult(
        task=state.task,
        status=status,
        iterations=len(steps),
        steps=steps,
        final_result=final_result,
        stopped_reason=stopped_reason,
        lifecycle=lifecycle,
        progress_history=history,
        completion_evidence=evidence,
        recovery_history=recovery_history,
        outcome=outcome,
        progress_report=report,
        error=error,
        diagnostics=tuple(notes),
    )


def _record_verification_phase(
    lifecycle: TaskLifecycle,
    result: ActionResult,
    evidence: CompletionEvidence,
) -> None:
    """Bracket an executed verification in the lifecycle.

    Verification runs as an ordinary agent action, so it already passed
    the guardrails. The lifecycle records it as RUNNING -> VERIFYING ->
    RUNNING. Completion is never decided here: COMPLETED is only reached
    through ``_complete_finished_task`` with VERIFIED evidence, and a
    failed verification returns to execution, where the agent inspects
    the failure and the human approval boundary still applies.
    """

    if lifecycle.status != TASK_RUNNING:
        return

    lifecycle.transition(
        TASK_VERIFYING,
        "The verification pipeline ran for the applied change.",
    )

    outcome = "passed" if result.success else "failed"

    lifecycle.transition(
        TASK_RUNNING,
        f"Verification {outcome} (evidence {evidence.status}); "
        "returning to execution.",
    )


def _resolve_intent(agent: FlyCoderAgent, state: CodingState) -> str:
    """Return the task intent without mutating state.

    ``state.task_intent`` is only set by the agent's first ``decide()``.
    The baseline snapshot is taken before that, so the intent is
    classified here with the agent's own classifier instead.
    """

    if state.task_intent:
        return state.task_intent

    return agent.classify_task(state.task)


def _completion_decision(
    lifecycle_status: str,
    evidence: CompletionEvidence,
    finished_reason: str,
    verification_attempts: tuple[VerificationAttempt, ...] = (),
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """Decide how a finished task ends, without changing anything.

    ``state.finished`` only means the agent stopped choosing actions.
    The task is COMPLETED only when ``success_blockers`` finds nothing:
    VERIFIED evidence, and no failed verification run without a newer
    passing one (Phase 10.8). Pending approval escalates to a human.
    Anything else leaves the task UNVERIFIED; it is never reported as
    successful.

    Returns the autonomous status, the stopped reason, and the lifecycle
    transitions to apply.
    """

    if evidence.needs_approval:
        reason = (
            "The agent finished, but a repair proposal still requires "
            "explicit human approval."
        )
        path = (
            ((TASK_NEEDS_HUMAN, reason),)
            if lifecycle_status == TASK_RUNNING
            else ()
        )

        return AUTONOMOUS_NEEDS_HUMAN, reason, path

    transitions: list[tuple[str, str]] = []
    status_now = lifecycle_status

    if status_now == TASK_RUNNING:
        transitions.append((TASK_VERIFYING, finished_reason))
        status_now = TASK_VERIFYING

    blockers = success_blockers(evidence, verification_attempts)

    if not blockers:
        reason = (
            "Autonomous task completed with verified evidence "
            f"({evidence.verification_method})."
        )

        if status_now == TASK_VERIFYING:
            transitions.append((TASK_COMPLETED, reason))

        return AUTONOMOUS_COMPLETED, reason, tuple(transitions)

    if evidence.verified:
        detail = f" {blockers[-1]}"
    else:
        detail = f" {evidence.reasons[-1]}" if evidence.reasons else ""

    reason = (
        "The agent finished, but completion evidence is "
        f"{evidence.status}.{detail}"
    )

    if status_now == TASK_VERIFYING:
        transitions.append((TASK_FAILED, reason))

    return AUTONOMOUS_UNVERIFIED, reason, tuple(transitions)


def _complete_finished_task(
    lifecycle: TaskLifecycle,
    evidence: CompletionEvidence,
    finished_reason: str,
    verification_attempts: tuple[VerificationAttempt, ...] = (),
) -> tuple[str, str]:
    """Gate completion of a finished task and apply its transitions.

    Kept for compatibility; the decision is ``_completion_decision``.
    Returns the autonomous status and the stopped reason.
    """

    status, reason, path = _completion_decision(
        lifecycle.status,
        evidence,
        finished_reason,
        verification_attempts,
    )

    for to_state, transition_reason in path:
        lifecycle.transition(to_state, transition_reason)

    return status, reason


def _build_iteration_decision(
    *,
    state: CodingState,
    result: ActionResult,
    recovery: RecoveryDecision | None,
    stall: StallAssessment,
    repetition_tracker: ActionRepetitionTracker,
    recovery_assessment: RecoveryAssessment | None = None,
) -> IterationDecision:
    """Build the advisory decision for the current iteration."""

    repetition = repetition_tracker.assess()

    recovery_exhausted = (
        recovery_assessment is not None
        and recovery_assessment.should_escalate
    )

    recovery_available = (
        recovery is not None
        and recovery.recovery_action is not None
        and not recovery_exhausted
    )

    return decide_iteration(
        task_finished=state.finished,
        human_input_required=state.user_input_needed,
        action_success=result.success,
        recovery_available=recovery_available,
        stall=stall,
        repetition=repetition,
        recovery_exhausted=recovery_exhausted,
        recovery_reason=(
            recovery_assessment.reason if recovery_exhausted else ""
        ),
    )


def run_autonomous_task(
    agent: FlyCoderAgent,
    state: CodingState,
    *,
    max_iterations: int = 20,
    verification: VerificationRecord | None = None,
    resume: ResumeContext | None = None,
    on_step: Callable[[AutonomousStep], None] | None = None,
) -> AutonomousTaskResult:
    """Run one task through the autonomous orchestration loop.

    The orchestration layer delegates actual action selection and
    execution to the existing agent. It does not bypass guardrails,
    approval boundaries, recovery logic, or learning.

    Iteration analysis is observational and advisory. It does not
    execute a suggested action or grant repair approval.

    In particular, this function never automatically approves a
    repair.

    Phase 10.4.6 progress integration:

    * A task-level ``ProgressHistory`` gets a baseline snapshot at
      iteration 0 and one snapshot after every executed iteration.
    * Completion evidence is re-evaluated from the current state after
      every iteration, using ``verification`` when supplied. A record
      produced for different code is reported STALE, never as proof.
    * ``state.finished`` no longer implies COMPLETED. A finished task is
      COMPLETED only with VERIFIED evidence; otherwise it is
      UNVERIFIED, or NEEDS_HUMAN when approval is pending.
    * The final result carries the history, the latest evidence, and a
      ``ProgressReport``.

    Phase 10.5 verification:

    * The agent runs the Phase 7 pipeline as the ``verify_changes``
      action after an applied change, through the normal guardrails.
      The loop records it as RUNNING -> VERIFYING -> RUNNING.
    * The record that action stores on ``state.verification_record`` is
      the latest evidence and takes precedence over ``verification``.

    Phase 10.9 failure isolation:

    * Once the lifecycle has started, an ordinary exception from any
      component (the agent, observation, recovery, progress, evidence)
      ends the run as FAILED through ``_finalize``: the failing
      iteration is recorded as a ``component_error`` step, earlier steps
      and histories are kept, and ``result.error`` describes it. The
      outcome comes from the single classifier, so it is never success.
    * ``KeyboardInterrupt`` and ``SystemExit`` finalize the run the same
      way, attach the result to the original exception object
      (``attached_result``), and re-raise that same object.
    * A ``finally`` guard fails any lifecycle still active, so a run
      never returns or raises with its lifecycle RUNNING/VERIFYING.
    * Precondition errors (invalid arguments) are raised before the
      lifecycle changes, as before.

    Args:
        verification: Optional Phase 7 pipeline result already bound to
            a code signature with ``record_verification``, used when the
            state holds no verification record of its own.
        resume: Phase 10.7 context of a task paused at NEEDS_HUMAN whose
            human decision was already applied. The run continues that
            lifecycle and those histories; ``max_iterations`` is the
            number of iterations this run may use.
        on_step: Optional read-only observer called after each recorded
            step (for example a CLI printer). Its ordinary exceptions are
            recorded in ``result.diagnostics`` and never change the run.

    Progress tracking and evidence are observational. They never
    select or execute actions, mutate state, or approve repairs.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    if verification is not None and not isinstance(
        verification,
        VerificationRecord,
    ):
        raise TypeError("verification must be a VerificationRecord")

    if on_step is not None and not callable(on_step):
        raise TypeError("on_step must be callable")

    if resume is not None:
        if resume.lifecycle.status != TASK_NEEDS_HUMAN:
            raise ValueError(
                "Only a task paused for human intervention can be resumed."
            )

        if state.user_input_needed:
            raise ValueError(
                "The task is still waiting for a human decision."
            )

        if resume.iteration_offset < 0:
            raise ValueError("iteration_offset must be non-negative")

    intent = _resolve_intent(agent, state)

    def evaluate_evidence() -> CompletionEvidence:
        # The state's own record is produced by the latest
        # verify_changes run, so it is newer than any record supplied
        # when the run started.
        return evaluate_completion_evidence(
            state,
            state.verification_record or verification,
            state.task_intent or intent,
        )

    if resume is None:
        progress_history = ProgressHistory()

        # Baseline before any action. Iteration numbers then strictly
        # increase (1, 2, ...), and ProgressHistory.record rejects any
        # duplicate or out-of-order iteration explicitly.
        progress_history.capture(0, state, intent)

        lifecycle = TaskLifecycle(task=state.task)
        recovery_history = RecoveryHistory()
        first_iteration = 1
        prior_steps: tuple[AutonomousStep, ...] = ()
    else:
        # Continue the paused task: same lifecycle and histories, and
        # iteration numbers that follow the previous run.
        progress_history = resume.progress_history
        lifecycle = resume.lifecycle
        recovery_history = resume.recovery_history
        first_iteration = resume.iteration_offset + 1
        prior_steps = tuple(resume.prior_steps)

    steps: list[AutonomousStep] = []
    diagnostics: list[str] = []

    # Where the run is, for diagnostics of a component failure.
    stage = "start"
    current_iteration: int | None = None

    def finish(
        path: tuple[tuple[str, str], ...],
        status: str,
        stopped_reason: str,
        final_result: ActionResult | None,
        evidence_now: CompletionEvidence | None = None,
        error: ComponentError | None = None,
    ) -> AutonomousTaskResult:
        return _finalize(
            state=state,
            lifecycle=lifecycle,
            path=path,
            status=status,
            stopped_reason=stopped_reason,
            steps=steps,
            history=progress_history,
            recovery_history=recovery_history,
            intent=intent,
            # A stop decision reuses the evidence it was taken with; a
            # failure re-evaluates the state as it is now.
            evaluate_evidence=(
                evaluate_evidence
                if evidence_now is None
                else (lambda: evidence_now)
            ),
            final_result=final_result,
            prior_steps=prior_steps,
            error=error,
            diagnostics=diagnostics,
        )

    def fail_run(error: ComponentError) -> AutonomousTaskResult:
        """Record the failing iteration and finalize the run as FAILED."""

        if current_iteration is not None and not (
            steps and steps[-1].iteration == current_iteration
        ):
            failed_step = _component_error_step(current_iteration, error)
            steps.append(failed_step)

            try:
                _record_step(lifecycle, current_iteration, failed_step.result)
            except Exception:
                pass

            _notify_observer(on_step, failed_step, diagnostics)

        reason = (
            "Execution was interrupted" if error.interrupted
            else "A component failed"
        ) + f": {error.describe()}"

        return finish(
            _failed_path(lifecycle.status, reason),
            AUTONOMOUS_FAILED,
            reason,
            steps[-1].result if steps else None,
            error=error,
        )

    try:
        try:
            if resume is None:
                stage = "initialize"

                lifecycle.transition(
                    TASK_INITIALIZING,
                    "Autonomous task initialization started.",
                )

                lifecycle.transition(
                    TASK_RUNNING,
                    "Autonomous execution started.",
                )
            else:
                stage = "resume"
                lifecycle.transition(TASK_RUNNING, resume.reason)

            stage = "evidence"
            evidence = evaluate_evidence()

            progress_tracker = ProgressTracker()
            repetition_tracker = ActionRepetitionTracker()

            previous_signature: str | None = None

            for iteration in range(
                first_iteration,
                first_iteration + max_iterations,
            ):
                if state.finished:
                    # The state was finished before this iteration ran,
                    # so no step or snapshot is recorded for it; evidence
                    # reflects the state as it is now.
                    current_iteration = None
                    stage = "completion"

                    status, stopped_reason, path = _completion_decision(
                        lifecycle.status,
                        evidence,
                        "Agent had already marked the task finished.",
                        _verification_attempts(prior_steps + tuple(steps)),
                    )

                    return finish(
                        path,
                        status,
                        stopped_reason,
                        steps[-1].result if steps else None,
                        evidence,
                    )

                current_iteration = iteration

                stage = "run_once"
                result = agent.run_once(state)

                stage = "observation"
                observation = _execution_observation(result)

                if observation is None:
                    observation = observe_action(result, state)

                stage = "recovery"
                recovery = None

                if not result.success:
                    recovery = recover_from_failure(result, state)

                stage = "iteration_observation"
                iteration_observation = observe_iteration(
                    iteration,
                    result,
                    state,
                    observation,
                    previous_signature=previous_signature,
                )

                progress_tracker.record(iteration_observation)

                # Phase 10.6: failures are recorded across iterations so
                # a repeat without new information is escalated, not
                # retried.
                stage = "recovery_history"
                recovery_assessment = None

                if recovery is not None and not result.success:
                    recovery_assessment = recovery_history.record(
                        iteration,
                        result,
                        recovery,
                        iteration_observation.state_signature,
                    )

                stage = "iteration_decision"
                repetition_tracker.record(result.action)

                stall = assess_stall(progress_tracker)

                iteration_decision = _build_iteration_decision(
                    state=state,
                    result=result,
                    recovery=recovery,
                    stall=stall,
                    repetition_tracker=repetition_tracker,
                    recovery_assessment=recovery_assessment,
                )

                # Phase 10.4.6: observe task-level progress and
                # re-evaluate completion evidence after the action
                # changed (or did not change) the state.
                stage = "progress"
                progress_snapshot = progress_history.capture(
                    iteration,
                    state,
                    state.task_intent or intent,
                )

                stage = "evidence"
                evidence = evaluate_evidence()

                stage = "step"
                step = AutonomousStep(
                    iteration=iteration,
                    result=result,
                    observation=observation,
                    recovery=recovery,
                    iteration_observation=iteration_observation,
                    stall=stall,
                    iteration_decision=iteration_decision,
                    progress_snapshot=progress_snapshot,
                    completion_evidence=evidence,
                    recovery_assessment=recovery_assessment,
                    confidence=assess_confidence(result, state),
                )

                steps.append(step)

                _record_step(lifecycle, iteration, result)

                if (
                    result.action == VERIFICATION_ACTION
                    and not _guardrail_blocked(result)
                ):
                    stage = "verification_phase"
                    _record_verification_phase(lifecycle, result, evidence)

                _notify_observer(on_step, step, diagnostics)

                previous_signature = iteration_observation.state_signature

                stage = "stop_decision"

                if (
                    state.user_input_needed
                    or observation.requires_human_input
                    or (
                        recovery is not None
                        and recovery.requires_human_input
                    )
                ):
                    reason = (
                        "Human input is required before execution can "
                        "continue."
                    )

                    return finish(
                        _running_path(
                            lifecycle.status,
                            (TASK_NEEDS_HUMAN, reason),
                        ),
                        AUTONOMOUS_NEEDS_HUMAN,
                        reason,
                        result,
                        evidence,
                    )

                if _guardrail_blocked(result):
                    reason = "The action was blocked by a safety guardrail."

                    return finish(
                        _running_path(
                            lifecycle.status,
                            (TASK_BLOCKED, reason),
                        ),
                        AUTONOMOUS_BLOCKED,
                        reason,
                        result,
                        evidence,
                    )

                if state.finished:
                    stage = "completion"

                    status, stopped_reason, path = _completion_decision(
                        lifecycle.status,
                        evidence,
                        "The agent marked the task finished.",
                        _verification_attempts(prior_steps + tuple(steps)),
                    )

                    return finish(
                        path,
                        status,
                        stopped_reason,
                        result,
                        evidence,
                    )

                if iteration_decision.should_escalate:
                    return finish(
                        _running_path(
                            lifecycle.status,
                            (TASK_NEEDS_HUMAN, iteration_decision.reason),
                        ),
                        AUTONOMOUS_NEEDS_HUMAN,
                        iteration_decision.reason,
                        result,
                        evidence,
                    )

                if iteration_decision.should_stop:
                    return finish(
                        _running_path(
                            lifecycle.status,
                            (TASK_FAILED, iteration_decision.reason),
                        ),
                        AUTONOMOUS_FAILED,
                        iteration_decision.reason,
                        result,
                        evidence,
                    )

                if not result.success:
                    if recovery is None or recovery.recovery_action is None:
                        return finish(
                            _running_path(
                                lifecycle.status,
                                (
                                    TASK_FAILED,
                                    "No recovery action is available.",
                                ),
                            ),
                            AUTONOMOUS_FAILED,
                            "The task failed and no recovery action is "
                            "available.",
                            result,
                            evidence,
                        )

                # The autonomous layer intentionally does not execute
                # observation.next_action or recovery.recovery_action
                # directly.
                #
                # The next iteration returns to agent.run_once(),
                # allowing the existing decision engine to select the
                # next action.

            current_iteration = None
            stage = "iteration_limit"

            return finish(
                _running_path(
                    lifecycle.status,
                    (TASK_VERIFYING, "Maximum autonomous iterations reached."),
                    (
                        TASK_FAILED,
                        "Task did not complete within the iteration limit.",
                    ),
                ),
                AUTONOMOUS_MAX_ITERATIONS,
                "The task reached the maximum autonomous iteration limit.",
                steps[-1].result if steps else None,
                evidence,
            )

        except Exception as exc:
            return fail_run(
                ComponentError.from_exception(
                    exc,
                    stage=stage,
                    iteration=current_iteration,
                )
            )

        except (KeyboardInterrupt, SystemExit) as interrupt:
            try:
                interrupted = fail_run(
                    ComponentError.from_exception(
                        interrupt,
                        stage=stage,
                        iteration=current_iteration,
                        interrupted=True,
                    )
                )
            except Exception:
                interrupted = None

            if interrupted is not None:
                attach_result(interrupt, interrupted)

            # Re-raise the original interrupt object; it is never
            # swallowed or replaced.
            raise

    finally:
        _force_terminal(
            lifecycle,
            "The autonomous run ended abnormally.",
        )
