"""In-process task sessions with human intervention for FLY-CODER 10.7.

A ``TaskSession`` keeps one task's context across human interventions:
the ``CodingState``, lifecycle, progress history, recovery history, all
executed steps, the intervention requests shown, and the human decisions
applied. Execution always goes through ``run_autonomous_task`` (and so
``agent.run_once`` and the guardrails); resuming passes a
``ResumeContext`` instead of starting a new task.

Phase 10.8: the session keeps the task's completion outcome current.
After every run, decision, and cancellation it re-evaluates completion
evidence from the state and re-classifies the outcome from the complete
step history (``_classify_outcome``), including whether this session can
actually resume (iterations left, not restricted to cancellation).

Phase 10.9: the session stays consistent when a run fails. Component
exceptions end a run as FAILED inside the loop (``_finalize``); an
interrupted run's result is absorbed before the original
KeyboardInterrupt/SystemExit is re-raised; refreshing the outcome is
atomic; and every non-fatal problem is kept in ``diagnostics``.

Phase 10.9 entry point: ``TaskSession`` is the orchestration controller.
``run()`` executes the initial task once (``start()`` is construct plus
``run()``), ``resume()`` continues it after a human decision, and
``close()`` ends the controller's hold on the task: afterwards no run,
resume, or decision is accepted and the outcome reports that the task
cannot be resumed. Closing never changes the lifecycle.

Limitation: sessions live in memory only. Persisting a paused task
across processes would need a reviewed serialization format and is not
implemented. The task identifier is therefore only meaningful within
this process.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

from flycoder.state import CodingState
from flycoder.tools.autonomous import (
    AUTONOMOUS_CANCELLED,
    AUTONOMOUS_FAILED,
    AutonomousStep,
    AutonomousTaskResult,
    ComponentError,
    ResumeContext,
    _classify_outcome,
    _finalize,
    attached_result,
    run_autonomous_task,
)
from flycoder.tools.completion_outcome import CompletionOutcome
from flycoder.tools.completion_evidence import evaluate_completion_evidence
from flycoder.tools.human_intervention import (
    DECISION_CANCEL_TASK,
    DecisionRecord,
    HumanDecision,
    HumanDecisionError,
    InterventionRequest,
    apply_human_decision,
    build_intervention_request,
    restrict_to_cancellation,
)
from flycoder.tools.progress_report import ProgressReport, build_progress_report
from flycoder.tools.task_lifecycle import (
    TASK_CANCELLED,
    TASK_FAILED,
    TASK_NEEDS_HUMAN,
)

if TYPE_CHECKING:
    from flycoder.agent import FlyCoderAgent
    from flycoder.tools.orchestration import OrchestrationSnapshot


# The same intervention (same situation, same id) appearing this many
# times without progress only accepts cancellation.
MAX_REPEATED_INTERVENTIONS = 3


class TaskSession:
    """One task, possibly spanning several human interventions."""

    def __init__(
        self,
        agent: FlyCoderAgent,
        state: CodingState,
        *,
        max_iterations: int = 20,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        self.agent = agent
        self.state = state
        self.max_iterations = max_iterations
        # Process-local identifier for reports (sessions are not
        # persisted).
        self.task_id = uuid.uuid4().hex[:12]
        self.steps: list[AutonomousStep] = []
        self.result: AutonomousTaskResult | None = None
        self._decisions: list[DecisionRecord] = []
        self._intervention_ids: list[str] = []
        # Non-fatal problems across all runs (observer, report, and
        # refresh failures), in order.
        self._diagnostics: list[str] = []
        # Phase 10.9: the initial run happens at most once; a closed
        # session accepts no further run, resume, or decision.
        self._started = False
        self._running = False
        self._closed = False
        self._close_reason: str | None = None

    # --------------------------------------------------------------
    # Construction and execution
    # --------------------------------------------------------------

    @classmethod
    def start(
        cls,
        agent: FlyCoderAgent,
        task: str,
        *,
        max_iterations: int = 20,
        state: CodingState | None = None,
        on_step: Callable[[AutonomousStep], None] | None = None,
    ) -> TaskSession:
        """Create a session for ``task`` and run it until it stops.

        ``state`` may supply prepared task context (its task must match).
        ``on_step`` is a read-only step observer (see ``run``).

        If the run is interrupted, the interrupt propagates and this
        method returns nothing; construct the session and call ``run()``
        to keep a handle on it in that case.
        """

        if state is None:
            state = CodingState(task=task)
        elif state.task != task:
            raise ValueError("The supplied state belongs to a different task.")

        session = cls(agent, state, max_iterations=max_iterations)
        session.run(on_step=on_step)
        return session

    def run(
        self,
        *,
        on_step: Callable[[AutonomousStep], None] | None = None,
    ) -> AutonomousTaskResult:
        """Execute the session's initial run through the autonomous loop.

        The task intent is classified once, before anything runs. A
        session runs at most once, even if that run failed or was
        interrupted; continue a paused task with ``resume()``.

        ``on_step`` observes each recorded step (for example a CLI
        printer); its ordinary exceptions are diagnostics only.

        Raises:
            ValueError: if the session already ran or is closed.
        """

        if self._closed:
            raise ValueError("The session is closed; it cannot run.")

        if self._started:
            raise ValueError(
                "The session has already run; use resume() after a human "
                "decision."
            )

        if not self.state.task_intent:
            self.state.task_intent = self.agent.classify_task(self.state.task)

        self._started = True

        return self._execute(
            lambda: run_autonomous_task(
                self.agent,
                self.state,
                max_iterations=self.max_iterations,
                on_step=on_step,
            ),
            resuming=False,
        )

    def close(
        self,
        reason: str = "The session was closed by its controller.",
    ) -> None:
        """End the controller's hold on the task.

        Afterwards ``run()``, ``resume()``, and ``decide()`` are refused
        and the outcome reports ``can_resume=False``. The lifecycle,
        state, history, and outcome classification are not otherwise
        changed. Closing again keeps the first reason.

        Raises:
            ValueError: if ``reason`` is empty.
        """

        if self._closed:
            return

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")

        self._closed = True
        self._close_reason = reason

        # During a run the result is refreshed when the run is absorbed.
        if not self._running:
            self._refresh()

    def _execute(self, run, *, resuming: bool) -> AutonomousTaskResult:
        """Run the loop and absorb its result, whatever happens.

        Ordinary component exceptions never escape the loop once its
        lifecycle has started; they come back as a FAILED result. An
        ordinary exception here was therefore raised before anything
        changed (an invalid precondition) and is propagated unchanged.
        KeyboardInterrupt/SystemExit: the result attached to the
        original exception is absorbed, then that same object is
        re-raised.
        """

        previous_finished = self.state.finished

        if resuming:
            # A human decision reopens the task: whatever the agent had
            # concluded before the pause must be re-established.
            self.state.finished = False

        self._running = True

        try:
            result = run()
        except (KeyboardInterrupt, SystemExit) as interrupt:
            self._running = False
            self._after_interrupt(
                interrupt,
                resuming=resuming,
                previous_finished=previous_finished,
            )
            raise
        except Exception:
            self._running = False

            if resuming:
                self.state.finished = previous_finished
            raise

        self._running = False
        self._absorb(result)
        return result

    def _after_interrupt(
        self,
        interrupt: BaseException,
        *,
        resuming: bool,
        previous_finished: bool,
    ) -> None:
        """Bring the session in line with an interrupted run."""

        result = attached_result(interrupt)

        if result is not None:
            self._absorb(result)
            return

        previous = self.result

        if previous is None or previous.lifecycle.status == TASK_NEEDS_HUMAN:
            # The run never started: nothing but the reopened flag
            # changed, and that is restored.
            if resuming:
                self.state.finished = previous_finished
            return

        # The run started, the loop's guard made its lifecycle terminal,
        # but no result could be attached (for example a second
        # interrupt during finalization). Record that explicitly.
        note = (
            "Execution was interrupted before a result could be recorded; "
            "the steps of the interrupted run are unavailable."
        )
        self.result = AutonomousTaskResult(
            task=self.state.task,
            status=AUTONOMOUS_FAILED,
            iterations=0,
            stopped_reason=note,
            lifecycle=previous.lifecycle,
            progress_history=previous.progress_history,
            recovery_history=previous.recovery_history,
            error=ComponentError.from_exception(
                interrupt,
                stage="finalization",
                interrupted=True,
            ),
            diagnostics=(note,),
        )
        self._diagnostics.append(note)
        self._refresh()

    def _absorb(self, result: AutonomousTaskResult) -> None:
        self.steps.extend(result.steps)
        self.result = result
        self._diagnostics.extend(result.diagnostics)
        self._refresh()

    def _intent(self) -> str:
        return self.state.task_intent or self.agent.classify_task(self.state.task)

    def _resume_available(self) -> bool:
        if self._closed:
            return False

        request = self.intervention_request()

        cancellation_only = (
            request is not None
            and request.allowed_decisions == (DECISION_CANCEL_TASK,)
        )

        return self.remaining_iterations >= 1 and not cancellation_only

    def _refresh(self) -> None:
        """Re-evaluate evidence and the outcome from the current state.

        Called after every run, decision, and cancellation, so the
        result never reports evidence or an outcome the state no longer
        supports. Evaluation is read-only; earlier per-step evidence in
        ``self.steps`` is kept as it was recorded.

        Phase 10.9: evidence and outcome are replaced together or not at
        all. If either cannot be computed, the previous (consistent)
        values are kept and the failure is recorded in ``diagnostics``.
        The report is not authoritative; if only it fails, it becomes
        None with a diagnostic.
        """

        result = self.result

        if result is None:
            return

        try:
            evidence = evaluate_completion_evidence(
                self.state,
                intent=self._intent(),
            )
            outcome = _classify_outcome(
                self.state,
                status=result.status,
                lifecycle=result.lifecycle,
                evidence=evidence,
                steps=tuple(self.steps),
                recovery_history=result.recovery_history,
                final_result=self.steps[-1].result if self.steps else None,
                resume_available=self._resume_available(),
            )
        except Exception as error:
            failure = ComponentError.from_exception(error, stage="refresh")
            self._diagnostics.append(
                "Outcome refresh failed; the previous evidence and outcome "
                f"are kept: {failure.describe()}"
            )
            return

        result.completion_evidence = evidence
        result.outcome = outcome

        try:
            result.progress_report = self._build_report(evidence, outcome)
        except Exception as error:
            result.progress_report = None
            failure = ComponentError.from_exception(error, stage="report")
            self._diagnostics.append(
                f"Progress report unavailable: {failure.describe()}"
            )

    @property
    def iterations_used(self) -> int:
        return len(self.steps)

    @property
    def remaining_iterations(self) -> int:
        return max(0, self.max_iterations - self.iterations_used)

    @property
    def lifecycle(self):
        return self.result.lifecycle if self.result else None

    @property
    def status(self) -> str | None:
        return self.result.status if self.result else None

    @property
    def needs_human(self) -> bool:
        return (
            self.lifecycle is not None
            and self.lifecycle.status == TASK_NEEDS_HUMAN
        )

    @property
    def cancelled(self) -> bool:
        return self.status == AUTONOMOUS_CANCELLED

    @property
    def decisions(self) -> tuple[DecisionRecord, ...]:
        return tuple(self._decisions)

    @property
    def outcome(self) -> CompletionOutcome | None:
        """The task's current completion outcome (Phase 10.8)."""

        return self.result.outcome if self.result else None

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Non-fatal problems recorded across all runs (Phase 10.9)."""

        return tuple(self._diagnostics)

    @property
    def started(self) -> bool:
        """Whether the initial run has been started."""

        return self._started

    @property
    def running(self) -> bool:
        """Whether a run is executing right now (for example while an
        ``on_step`` observer is called)."""

        return self._running

    @property
    def closed(self) -> bool:
        """Whether the session no longer accepts run, resume, or decide."""

        return self._closed

    @property
    def close_reason(self) -> str | None:
        return self._close_reason

    # --------------------------------------------------------------
    # Human intervention
    # --------------------------------------------------------------

    def intervention_request(self) -> InterventionRequest | None:
        """Return the pending intervention, or None when none is pending.

        A request that keeps recurring for the same situation is
        restricted to cancellation, so human intervention cannot loop.
        """

        if not self.needs_human or not self.state.user_input_needed:
            return None

        request = build_intervention_request(
            self.state,
            task_status=self.lifecycle.status,
            evidence=self.result.completion_evidence,
        )

        if (
            self._intervention_ids.count(request.intervention_id) + 1
            >= MAX_REPEATED_INTERVENTIONS
        ):
            request = restrict_to_cancellation(request)

        return request

    def decide(self, decision: HumanDecision) -> DecisionRecord:
        """Validate and apply a human decision for the pending request.

        Raises:
            HumanDecisionError: if the session is closed, no
                intervention is pending, or the decision is invalid.
        """

        if self._closed:
            raise HumanDecisionError(
                "The session is closed; no further decisions are accepted."
            )

        request = self.intervention_request()

        if request is None:
            raise HumanDecisionError("No human intervention is pending.")

        record = apply_human_decision(self.state, request, decision)

        self._decisions.append(record)
        self._intervention_ids.append(request.intervention_id)

        if decision.kind == DECISION_CANCEL_TASK:
            self._cancel(record)
        else:
            # The decision changed the state (for example a discarded
            # proposal); evidence and outcome are recalculated.
            self._refresh()

        return record

    def _cancel(self, record: DecisionRecord) -> None:
        previous = self.result

        # The loop's single finalization path: the outcome is classified
        # for CANCELLED before the transition. If that fails, the task is
        # still cancelled (a degraded result); cancellation never becomes
        # FAILED or COMPLETED.
        self.result = _finalize(
            state=self.state,
            lifecycle=previous.lifecycle,
            path=((TASK_CANCELLED, record.outcome),),
            status=AUTONOMOUS_CANCELLED,
            stopped_reason="The task was cancelled by a human.",
            steps=list(self.steps),
            history=previous.progress_history,
            recovery_history=previous.recovery_history,
            intent=self.state.task_intent,
            evaluate_evidence=lambda: evaluate_completion_evidence(
                self.state,
                intent=self._intent(),
            ),
            final_result=self.steps[-1].result if self.steps else None,
            resume_available=False,
        )
        self._diagnostics.extend(self.result.diagnostics)
        self._refresh()

    def resume(
        self,
        *,
        on_step: Callable[[AutonomousStep], None] | None = None,
    ) -> AutonomousTaskResult:
        """Continue the task through the existing loop after a decision.

        Raises:
            ValueError: if the session is closed, the task is not
                paused, a decision is still pending, it was cancelled or
                failed, or no iterations remain.
        """

        if self._closed:
            raise ValueError("The session is closed; it cannot be resumed.")

        if self.cancelled:
            raise ValueError("A cancelled task cannot be resumed.")

        if self.lifecycle is not None and self.lifecycle.status == TASK_FAILED:
            raise ValueError("The task failed; it cannot be resumed.")

        if not self.needs_human:
            raise ValueError("Only a task paused for human input can resume.")

        if self.state.user_input_needed:
            raise ValueError(
                "A human decision is required before the task can resume."
            )

        if self.remaining_iterations < 1:
            raise ValueError("The task has no iterations left to resume.")

        previous = self.result
        latest_snapshot = previous.progress_history.latest
        iteration_offset = max(
            previous.lifecycle.current_iteration,
            latest_snapshot.iteration if latest_snapshot else 0,
        )
        resume_context = ResumeContext(
            lifecycle=previous.lifecycle,
            progress_history=previous.progress_history,
            recovery_history=previous.recovery_history,
            iteration_offset=iteration_offset,
            prior_steps=tuple(self.steps),
        )

        return self._execute(
            lambda: run_autonomous_task(
                self.agent,
                self.state,
                max_iterations=self.remaining_iterations,
                resume=resume_context,
                on_step=on_step,
            ),
            resuming=True,
        )

    # --------------------------------------------------------------
    # Reporting
    # --------------------------------------------------------------

    def snapshot(self) -> OrchestrationSnapshot:
        """Return a deeply immutable, point-in-time view of the session.

        The snapshot holds no reference to the session, its state, or
        any mutable internal object; later changes to the session never
        change it, and it cannot be used to change the session.
        """

        # Imported here: the snapshot module depends on this module's
        # dependencies, and only reads sessions.
        from flycoder.tools.orchestration import build_snapshot

        return build_snapshot(self)

    def report(self) -> ProgressReport:
        """Build a progress report including interventions and decisions."""

        return self._build_report(
            self.result.completion_evidence,
            self.outcome,
        )

    def _build_report(self, evidence, outcome) -> ProgressReport:
        return build_progress_report(
            self.state,
            self.result.progress_history,
            evidence,
            self.result.lifecycle,
            intent=self.state.task_intent,
            intervention=self.intervention_request(),
            human_decisions=self.decisions,
            outcome=outcome,
            task_id=self.task_id,
        )
