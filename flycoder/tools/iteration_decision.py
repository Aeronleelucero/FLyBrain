"""Autonomous iteration decision model for FLY-CODER Phase 10.3."""

from __future__ import annotations

from dataclasses import dataclass, field

from flycoder.tools.repetition import RepetitionAssessment
from flycoder.tools.stall import StallAssessment


ITERATION_CONTINUE = "CONTINUE"
ITERATION_VERIFY = "VERIFY"
ITERATION_STOP = "STOP"
ITERATION_ESCALATE = "ESCALATE"


@dataclass
class IterationDecision:
    """Describe what should happen after an autonomous iteration."""

    decision: str
    reason: str
    risks: list[str] = field(default_factory=list)

    @property
    def should_continue(self) -> bool:
        return self.decision == ITERATION_CONTINUE

    @property
    def should_verify(self) -> bool:
        return self.decision == ITERATION_VERIFY

    @property
    def should_stop(self) -> bool:
        return self.decision == ITERATION_STOP

    @property
    def should_escalate(self) -> bool:
        return self.decision == ITERATION_ESCALATE


def decide_iteration(
    *,
    task_finished: bool = False,
    human_input_required: bool = False,
    action_success: bool = True,
    recovery_available: bool = False,
    stall: StallAssessment | None = None,
    repetition: RepetitionAssessment | None = None,
) -> IterationDecision:
    """Determine whether another autonomous iteration is justified.

    This function is advisory only. It does not execute actions,
    modify state, approve repairs, or bypass guardrails.
    """

    risks: list[str] = []

    if human_input_required:
        return IterationDecision(
            decision=ITERATION_ESCALATE,
            reason="Human input is required before execution can continue.",
            risks=["Human approval or input is required."],
        )

    if task_finished:
        return IterationDecision(
            decision=ITERATION_VERIFY,
            reason="The task has been marked finished and should be verified.",
        )

    if stall is not None:
        if stall.critical:
            risks.append(
                "Critical no-progress condition detected."
            )

            return IterationDecision(
                decision=ITERATION_ESCALATE,
                reason=(
                    "Autonomous execution reached the critical "
                    "no-progress threshold."
                ),
                risks=risks,
            )

        if stall.stalled:
            risks.append(
                "Autonomous execution appears stalled."
            )

    if repetition is not None:
        if repetition.loop_risk:
            risks.append(
                "Repeated action indicates possible execution loop."
            )

            return IterationDecision(
                decision=ITERATION_ESCALATE,
                reason=(
                    "The same action has been repeated enough times "
                    "to indicate loop risk."
                ),
                risks=risks,
            )

        if repetition.repeated:
            risks.append(
                "The same action has been repeated consecutively."
            )

    if not action_success:
        if recovery_available:
            return IterationDecision(
                decision=ITERATION_CONTINUE,
                reason=(
                    "The action failed, but a recovery path is available."
                ),
                risks=risks,
            )

        return IterationDecision(
            decision=ITERATION_STOP,
            reason=(
                "The action failed and no recovery path is available."
            ),
            risks=risks,
        )

    if stall is not None and stall.stalled:
        return IterationDecision(
            decision=ITERATION_CONTINUE,
            reason=(
                "Execution may continue, but progress is currently stalled."
            ),
            risks=risks,
        )

    return IterationDecision(
        decision=ITERATION_CONTINUE,
        reason="Execution is progressing normally.",
        risks=risks,
    )
