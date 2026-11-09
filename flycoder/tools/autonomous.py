"""Autonomous task orchestration for FLY-CODER Phase 10."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.completion_evidence import (
    CompletionEvidence,
    VerificationRecord,
    evaluate_completion_evidence,
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
from flycoder.tools.recovery import RecoveryDecision, recover_from_failure
from flycoder.tools.stall import StallAssessment, assess_stall
from flycoder.tools.task_lifecycle import (
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


AUTONOMOUS_COMPLETED = "COMPLETED"
AUTONOMOUS_FAILED = "FAILED"
AUTONOMOUS_BLOCKED = "BLOCKED"
AUTONOMOUS_NEEDS_HUMAN = "NEEDS_HUMAN"
AUTONOMOUS_MAX_ITERATIONS = "MAX_ITERATIONS"

# The agent stopped (``state.finished``) but completion evidence does not
# verify the task. ``AutonomousTaskResult.completion_evidence.status``
# says why (FAILED, MISSING, STALE, INCOMPLETE or INCONSISTENT).
AUTONOMOUS_UNVERIFIED = "UNVERIFIED"


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

    @property
    def completed(self) -> bool:
        """Return whether the task completed with verified evidence."""

        return self.status == AUTONOMOUS_COMPLETED

    @property
    def unverified(self) -> bool:
        """Return whether the agent stopped without verified evidence."""

        return self.status == AUTONOMOUS_UNVERIFIED

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

    if not isinstance(result.data, dict):
        return False

    guardrail = result.data.get("guardrail")

    if not isinstance(guardrail, dict):
        return False

    return guardrail.get("allowed") is False


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


def _finish_result(
    *,
    state: CodingState,
    status: str,
    steps: list[AutonomousStep],
    final_result: ActionResult | None,
    stopped_reason: str,
    lifecycle: TaskLifecycle,
    history: ProgressHistory,
    evidence: CompletionEvidence,
    intent: str,
) -> AutonomousTaskResult:
    """Build the final autonomous task result and its progress report.

    The report is built after the final lifecycle transition, so it
    describes the task exactly as the result reports it. The intent is
    passed explicitly because a task can end before the agent ever
    classified it (for example, a state that was already finished).
    """

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
        progress_report=build_progress_report(
            state,
            history,
            evidence,
            lifecycle,
            intent=state.task_intent or intent,
        ),
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


def _complete_finished_task(
    lifecycle: TaskLifecycle,
    evidence: CompletionEvidence,
    finished_reason: str,
) -> tuple[str, str]:
    """Gate completion of a finished task on completion evidence.

    ``state.finished`` only means the agent stopped choosing actions.
    The task is COMPLETED only when the evidence is VERIFIED. Pending
    approval escalates to a human. Any other evidence status leaves the
    task UNVERIFIED; it is never reported as successful.

    Returns the autonomous status and the stopped reason.
    """

    if evidence.needs_approval:
        reason = (
            "The agent finished, but a repair proposal still requires "
            "explicit human approval."
        )

        if lifecycle.status == TASK_RUNNING:
            lifecycle.transition(TASK_NEEDS_HUMAN, reason)

        return AUTONOMOUS_NEEDS_HUMAN, reason

    if lifecycle.status == TASK_RUNNING:
        lifecycle.transition(TASK_VERIFYING, finished_reason)

    if evidence.verified:
        reason = (
            "Autonomous task completed with verified evidence "
            f"({evidence.verification_method})."
        )

        if lifecycle.status == TASK_VERIFYING:
            lifecycle.transition(TASK_COMPLETED, reason)

        return AUTONOMOUS_COMPLETED, reason

    detail = f" {evidence.reasons[-1]}" if evidence.reasons else ""
    reason = (
        "The agent finished, but completion evidence is "
        f"{evidence.status}.{detail}"
    )

    if lifecycle.status == TASK_VERIFYING:
        lifecycle.transition(TASK_FAILED, reason)

    return AUTONOMOUS_UNVERIFIED, reason


def _build_iteration_decision(
    *,
    state: CodingState,
    result: ActionResult,
    recovery: RecoveryDecision | None,
    stall: StallAssessment,
    repetition_tracker: ActionRepetitionTracker,
) -> IterationDecision:
    """Build the advisory decision for the current iteration."""

    repetition = repetition_tracker.assess()

    recovery_available = (
        recovery is not None
        and recovery.recovery_action is not None
    )

    return decide_iteration(
        task_finished=state.finished,
        human_input_required=state.user_input_needed,
        action_success=result.success,
        recovery_available=recovery_available,
        stall=stall,
        repetition=repetition,
    )


def run_autonomous_task(
    agent: FlyCoderAgent,
    state: CodingState,
    *,
    max_iterations: int = 20,
    verification: VerificationRecord | None = None,
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

    Args:
        verification: Optional Phase 7 pipeline result already bound to
            a code signature with ``record_verification``. The loop does
            not run the pipeline itself.

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

    intent = _resolve_intent(agent, state)

    def evaluate_evidence() -> CompletionEvidence:
        return evaluate_completion_evidence(
            state,
            verification,
            state.task_intent or intent,
        )

    progress_history = ProgressHistory()

    # Baseline before any action. Iteration numbers then strictly
    # increase (1, 2, ...), and ProgressHistory.record rejects any
    # duplicate or out-of-order iteration explicitly.
    progress_history.capture(0, state, intent)

    evidence = evaluate_evidence()

    lifecycle = TaskLifecycle(task=state.task)
    progress_tracker = ProgressTracker()
    repetition_tracker = ActionRepetitionTracker()

    steps: list[AutonomousStep] = []

    lifecycle.transition(
        TASK_INITIALIZING,
        "Autonomous task initialization started.",
    )

    lifecycle.transition(
        TASK_RUNNING,
        "Autonomous execution started.",
    )

    previous_signature: str | None = None

    for iteration in range(1, max_iterations + 1):
        if state.finished:
            # The state was finished before this iteration ran, so no
            # new snapshot is captured; evidence reflects the state as
            # it is now.
            status, stopped_reason = _complete_finished_task(
                lifecycle,
                evidence,
                "Agent had already marked the task finished.",
            )

            return _finish_result(
                state=state,
                status=status,
                steps=steps,
                final_result=steps[-1].result if steps else None,
                stopped_reason=stopped_reason,
                lifecycle=lifecycle,
                history=progress_history,
                evidence=evidence,
                intent=intent,
            )

        result = agent.run_once(state)

        observation = _execution_observation(result)

        if observation is None:
            observation = observe_action(result, state)

        recovery = None

        if not result.success:
            recovery = recover_from_failure(result, state)

        iteration_observation = observe_iteration(
            iteration,
            result,
            state,
            observation,
            previous_signature=previous_signature,
        )

        progress_tracker.record(iteration_observation)

        repetition_tracker.record(result.action)

        stall = assess_stall(progress_tracker)

        iteration_decision = _build_iteration_decision(
            state=state,
            result=result,
            recovery=recovery,
            stall=stall,
            repetition_tracker=repetition_tracker,
        )

        # Phase 10.4.6: observe task-level progress and re-evaluate
        # completion evidence after the action changed (or did not
        # change) the state.
        progress_snapshot = progress_history.capture(
            iteration,
            state,
            state.task_intent or intent,
        )

        evidence = evaluate_evidence()

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
        )

        steps.append(step)

        _record_step(lifecycle, iteration, result)

        previous_signature = iteration_observation.state_signature

        if (
            state.user_input_needed
            or observation.requires_human_input
            or (
                recovery is not None
                and recovery.requires_human_input
            )
        ):
            if lifecycle.status == TASK_RUNNING:
                lifecycle.transition(
                    TASK_NEEDS_HUMAN,
                    "Human input is required before execution can continue.",
                )

            return _finish_result(
                state=state,
                status=AUTONOMOUS_NEEDS_HUMAN,
                steps=steps,
                final_result=result,
                stopped_reason=(
                    "Human input is required before execution can continue."
                ),
                lifecycle=lifecycle,
                history=progress_history,
                evidence=evidence,
                intent=intent,
            )

        if _guardrail_blocked(result):
            if lifecycle.status == TASK_RUNNING:
                lifecycle.transition(
                    TASK_BLOCKED,
                    "The action was blocked by a safety guardrail.",
                )

            return _finish_result(
                state=state,
                status=AUTONOMOUS_BLOCKED,
                steps=steps,
                final_result=result,
                stopped_reason=(
                    "The action was blocked by a safety guardrail."
                ),
                lifecycle=lifecycle,
                history=progress_history,
                evidence=evidence,
                intent=intent,
            )

        if state.finished:
            status, stopped_reason = _complete_finished_task(
                lifecycle,
                evidence,
                "The agent marked the task finished.",
            )

            return _finish_result(
                state=state,
                status=status,
                steps=steps,
                final_result=result,
                stopped_reason=stopped_reason,
                lifecycle=lifecycle,
                history=progress_history,
                evidence=evidence,
                intent=intent,
            )

        if iteration_decision.should_escalate:
            if lifecycle.status == TASK_RUNNING:
                lifecycle.transition(
                    TASK_NEEDS_HUMAN,
                    iteration_decision.reason,
                )

            return _finish_result(
                state=state,
                status=AUTONOMOUS_NEEDS_HUMAN,
                steps=steps,
                final_result=result,
                stopped_reason=iteration_decision.reason,
                lifecycle=lifecycle,
                history=progress_history,
                evidence=evidence,
                intent=intent,
            )

        if iteration_decision.should_stop:
            if lifecycle.status == TASK_RUNNING:
                lifecycle.transition(
                    TASK_FAILED,
                    iteration_decision.reason,
                )

            return _finish_result(
                state=state,
                status=AUTONOMOUS_FAILED,
                steps=steps,
                final_result=result,
                stopped_reason=iteration_decision.reason,
                lifecycle=lifecycle,
                history=progress_history,
                evidence=evidence,
                intent=intent,
            )

        if not result.success:
            if recovery is None or recovery.recovery_action is None:
                if lifecycle.status == TASK_RUNNING:
                    lifecycle.transition(
                        TASK_FAILED,
                        "No recovery action is available.",
                    )

                return _finish_result(
                    state=state,
                    status=AUTONOMOUS_FAILED,
                    steps=steps,
                    final_result=result,
                    stopped_reason=(
                        "The task failed and no recovery action is available."
                    ),
                    lifecycle=lifecycle,
                    history=progress_history,
                    evidence=evidence,
                    intent=intent,
                )

        # The autonomous layer intentionally does not execute
        # observation.next_action or recovery.recovery_action directly.
        #
        # The next iteration returns to agent.run_once(), allowing
        # the existing decision engine to select the next action.

    if lifecycle.status == TASK_RUNNING:
        lifecycle.transition(
            TASK_VERIFYING,
            "Maximum autonomous iterations reached.",
        )

        lifecycle.transition(
            TASK_FAILED,
            "Task did not complete within the iteration limit.",
        )

    return _finish_result(
        state=state,
        status=AUTONOMOUS_MAX_ITERATIONS,
        steps=steps,
        final_result=steps[-1].result if steps else None,
        stopped_reason=(
            "The task reached the maximum autonomous iteration limit."
        ),
        lifecycle=lifecycle,
        history=progress_history,
        evidence=evidence,
        intent=intent,
    )
