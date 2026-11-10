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

Limitation: sessions live in memory only. Persisting a paused task
across processes would need a reviewed serialization format and is not
implemented. The task identifier is therefore only meaningful within
this process.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from flycoder.state import CodingState
from flycoder.tools.autonomous import (
    AUTONOMOUS_CANCELLED,
    AutonomousStep,
    AutonomousTaskResult,
    ResumeContext,
    _classify_outcome,
    _finish_result,
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
from flycoder.tools.task_lifecycle import TASK_CANCELLED, TASK_NEEDS_HUMAN

if TYPE_CHECKING:
    from flycoder.agent import FlyCoderAgent


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
    ) -> TaskSession:
        """Create a session for ``task`` and run it until it stops.

        ``state`` may supply prepared task context (its task must match).
        """

        if state is None:
            state = CodingState(task=task)
        elif state.task != task:
            raise ValueError("The supplied state belongs to a different task.")

        session = cls(agent, state, max_iterations=max_iterations)
        session._absorb(
            run_autonomous_task(
                agent,
                session.state,
                max_iterations=max_iterations,
            )
        )
        return session

    def _absorb(self, result: AutonomousTaskResult) -> None:
        self.steps.extend(result.steps)
        self.result = result
        self._refresh()

    def _intent(self) -> str:
        return self.state.task_intent or self.agent.classify_task(self.state.task)

    def _resume_available(self) -> bool:
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
        """

        result = self.result
        evidence = evaluate_completion_evidence(
            self.state,
            intent=self._intent(),
        )

        result.completion_evidence = evidence
        result.outcome = _classify_outcome(
            self.state,
            status=result.status,
            lifecycle=result.lifecycle,
            evidence=evidence,
            steps=tuple(self.steps),
            recovery_history=result.recovery_history,
            final_result=self.steps[-1].result if self.steps else None,
            resume_available=self._resume_available(),
        )
        result.progress_report = self.report()

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
            HumanDecisionError: if no intervention is pending or the
                decision is invalid for it.
        """

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
        lifecycle = self.lifecycle
        lifecycle.transition(TASK_CANCELLED, record.outcome)

        evidence = evaluate_completion_evidence(
            self.state,
            intent=self._intent(),
        )

        # Reuse the loop's result builder so the final report has the
        # same shape; cancellation is never COMPLETED.
        self.result = _finish_result(
            state=self.state,
            status=AUTONOMOUS_CANCELLED,
            steps=list(self.steps),
            final_result=self.steps[-1].result if self.steps else None,
            stopped_reason="The task was cancelled by a human.",
            lifecycle=lifecycle,
            history=self.result.progress_history,
            evidence=evidence,
            intent=self._intent(),
            recovery_history=self.result.recovery_history,
            resume_available=False,
        )
        self._refresh()

    def resume(self) -> AutonomousTaskResult:
        """Continue the task through the existing loop after a decision.

        Raises:
            ValueError: if the task is not paused, a decision is still
                pending, it was cancelled, or no iterations remain.
        """

        if self.cancelled:
            raise ValueError("A cancelled task cannot be resumed.")

        if not self.needs_human:
            raise ValueError("Only a task paused for human input can resume.")

        if self.state.user_input_needed:
            raise ValueError(
                "A human decision is required before the task can resume."
            )

        if self.remaining_iterations < 1:
            raise ValueError("The task has no iterations left to resume.")

        # A human decision reopens the task: whatever the agent had
        # concluded before the pause must be re-established.
        self.state.finished = False

        previous = self.result
        latest_snapshot = previous.progress_history.latest
        iteration_offset = max(
            previous.lifecycle.current_iteration,
            latest_snapshot.iteration if latest_snapshot else 0,
        )

        result = run_autonomous_task(
            self.agent,
            self.state,
            max_iterations=self.remaining_iterations,
            resume=ResumeContext(
                lifecycle=previous.lifecycle,
                progress_history=previous.progress_history,
                recovery_history=previous.recovery_history,
                iteration_offset=iteration_offset,
                prior_steps=tuple(self.steps),
            ),
        )

        self._absorb(result)
        return result

    # --------------------------------------------------------------
    # Reporting
    # --------------------------------------------------------------

    def report(self) -> ProgressReport:
        """Build a progress report including interventions and decisions."""

        return build_progress_report(
            self.state,
            self.result.progress_history,
            self.result.completion_evidence,
            self.result.lifecycle,
            intent=self.state.task_intent,
            intervention=self.intervention_request(),
            human_decisions=self.decisions,
            outcome=self.outcome,
            task_id=self.task_id,
        )
