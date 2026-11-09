"""Failure recovery tools for FLY-CODER Phase 9.4 and Phase 10.6.

Recovery is advisory. ``recover_from_failure`` classifies one failed
action and recommends a next step; ``RecoveryHistory`` (Phase 10.6)
tracks failures across iterations so repeated, uninformative recovery
can be detected and escalated.

Nothing here executes an action, writes files, or grants approval. A
recommended ``recovery_action`` is never run directly: the autonomous
loop always returns to ``agent.run_once(state)``, where the decision
engine and the guardrails stay authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.human_input import describe_human_input


# Recovery categories (Phase 10.6). They describe what kind of next step
# the failure calls for; none of them authorizes anything.
RECOVERY_NONE = "NONE"
RECOVERY_RETRY = "RETRY"
RECOVERY_INVESTIGATE = "INVESTIGATE"
RECOVERY_PROPOSE_REPAIR = "PROPOSE_REPAIR"
RECOVERY_UNRESOLVABLE = "UNRESOLVABLE"
RECOVERY_HUMAN_REQUIRED = "HUMAN_REQUIRED"

RECOVERY_CATEGORIES = {
    RECOVERY_NONE,
    RECOVERY_RETRY,
    RECOVERY_INVESTIGATE,
    RECOVERY_PROPOSE_REPAIR,
    RECOVERY_UNRESOLVABLE,
    RECOVERY_HUMAN_REQUIRED,
}


@dataclass
class RecoveryDecision:
    """Describe how the agent should recover from an action result."""

    action: str
    failed: bool
    recovery_action: str | None = None
    reason: str = ""
    risks: list[str] = field(default_factory=list)
    requires_human_input: bool = False
    retry_allowed: bool = False
    category: str = RECOVERY_NONE
    failure_type: str = "success"


def classify_failure(result: ActionResult) -> str:
    """Classify a failed action into a small set of recovery categories."""

    if result.success:
        return "success"

    action = result.action

    if action == "run_tests":
        if "timed out" in (result.message or "").lower():
            return "test_timeout"

        return "test_failure"

    if action == "verify_changes":
        return "verification_failure"

    if action == "inspect_files":
        return "inspection_failure"

    if action == "read_file":
        return "read_failure"

    if action == "write_code":
        return "write_failure"

    if action == "fix_error":
        return "diagnostic_failure"

    if action == "propose_repair":
        return "repair_proposal_failure"

    return "unknown_failure"


def repair_evidence_sufficient(state: CodingState) -> bool:
    """Return whether diagnosis supports preparing a repair proposal.

    Sufficient evidence means: a repair task, an inspected error, the
    diagnosed file read into state, no existing proposal, and no pending
    human input. This is the single definition used by the agent's
    decision engine and by recovery; it never authorizes applying
    anything.
    """

    return (
        state.task_intent == "repair"
        and state.error_inspected
        and bool(state.current_file)
        and state.current_file_content is not None
        and not state.repair_proposed
        and not state.user_input_needed
    )


def _decision(
    result: ActionResult,
    failure_type: str,
    category: str,
    reason: str,
    *,
    recovery_action: str | None = None,
    risks: list[str] | None = None,
    requires_human_input: bool = False,
    retry_allowed: bool = False,
) -> RecoveryDecision:
    return RecoveryDecision(
        action=result.action,
        failed=not result.success,
        recovery_action=recovery_action,
        reason=reason,
        risks=list(risks or []),
        requires_human_input=requires_human_input,
        retry_allowed=retry_allowed,
        category=category,
        failure_type=failure_type,
    )


def recover_from_failure(
    result: ActionResult,
    state: CodingState,
) -> RecoveryDecision:
    """Determine a safe recovery recommendation.

    This function is advisory only. It never executes an action and
    never grants permission to modify code or approve a repair.
    """

    if result.success:
        return _decision(
            result,
            "success",
            RECOVERY_NONE,
            "The action succeeded, so failure recovery is not required.",
        )

    failure_type = classify_failure(result)

    if state.user_input_needed:
        boundary = describe_human_input(state)

        return _decision(
            result,
            failure_type,
            RECOVERY_HUMAN_REQUIRED,
            (
                "The workflow requires user input or approval. "
                "Failure recovery must stop at the human boundary. "
                f"{boundary}"
            ),
            risks=[
                "Do not bypass explicit user approval.",
                "Do not automatically execute approve_repair.",
            ],
            requires_human_input=True,
        )

    if failure_type in {"test_failure", "diagnostic_failure"} and (
        repair_evidence_sufficient(state)
    ):
        return _decision(
            result,
            failure_type,
            RECOVERY_PROPOSE_REPAIR,
            (
                "The failure is diagnosed and the failing file is loaded, "
                "so a repair proposal can be prepared. Proposing does not "
                "modify files and still requires explicit approval."
            ),
            recovery_action="propose_repair",
            risks=[
                "A proposal must stop at the human approval boundary.",
            ],
        )

    if failure_type == "test_timeout":
        return _decision(
            result,
            failure_type,
            RECOVERY_RETRY,
            (
                "The test run timed out. Re-running tests is read-only and "
                "may succeed; a repeated timeout without new information "
                "is escalated."
            ),
            recovery_action="run_tests",
            risks=["The test suite may hang or be too slow."],
            retry_allowed=True,
        )

    if failure_type == "inspection_failure":
        return _decision(
            result,
            failure_type,
            RECOVERY_RETRY,
            (
                "Workspace inspection failed. Inspection is read-only and "
                "idempotent, so one retry is safe."
            ),
            recovery_action="inspect_files",
            risks=["The workspace may be unavailable."],
            retry_allowed=True,
        )

    if failure_type == "test_failure":
        return _decision(
            result,
            failure_type,
            RECOVERY_INVESTIGATE,
            (
                "The test suite failed. Inspect the failure before "
                "attempting a repair."
            ),
            recovery_action="fix_error",
            risks=[
                "The failure may be caused by the test environment.",
                "The root cause should be identified before modifying code.",
            ],
        )

    if failure_type == "verification_failure":
        if not state.last_error:
            return _decision(
                result,
                failure_type,
                RECOVERY_UNRESOLVABLE,
                (
                    "Verification could not run and produced no failure "
                    "details to inspect."
                ),
                risks=[
                    "Verification preconditions may not be satisfied.",
                ],
            )

        return _decision(
            result,
            failure_type,
            RECOVERY_INVESTIGATE,
            (
                "Verification of the applied change failed. Inspect the "
                "failure; any further repair still needs explicit approval."
            ),
            recovery_action="fix_error",
            risks=[
                "Do not retry verification without a code change.",
                "Do not propose or apply a repair without human approval.",
            ],
        )

    if failure_type == "read_failure":
        return _decision(
            result,
            failure_type,
            RECOVERY_INVESTIGATE,
            (
                "Reading the selected file failed. Reinspect the workspace "
                "before attempting another file read."
            ),
            recovery_action="inspect_files",
            risks=[
                "The selected path may be invalid or unavailable.",
                "Do not repeatedly retry an unresolved path.",
            ],
        )

    if failure_type == "write_failure":
        return _decision(
            result,
            failure_type,
            RECOVERY_UNRESOLVABLE,
            (
                "Writing code failed. The cause must be understood before "
                "attempting another modification."
            ),
            risks=[
                "The workspace may be partially affected.",
                "Blindly retrying a write could create additional changes.",
            ],
        )

    if failure_type == "diagnostic_failure":
        return _decision(
            result,
            failure_type,
            RECOVERY_INVESTIGATE,
            (
                "Error diagnosis failed. Gather a clearer explanation "
                "before continuing with repair."
            ),
            recovery_action="explain_error",
            risks=[
                "The underlying failure may not be understood yet.",
            ],
        )

    if failure_type == "repair_proposal_failure":
        return _decision(
            result,
            failure_type,
            RECOVERY_HUMAN_REQUIRED,
            (
                "The repair proposal could not be generated. "
                "Stop rather than attempting an unreviewed modification."
            ),
            risks=[
                "No safe repair proposal is currently available.",
                "Human investigation may be required.",
            ],
            requires_human_input=True,
        )

    return _decision(
        result,
        failure_type,
        RECOVERY_UNRESOLVABLE,
        (
            "The failure type is unknown. Stop automatic recovery until "
            "the failure can be understood."
        ),
        risks=[
            "Unknown failures must not trigger blind retries.",
        ],
    )


# ==================================================================
# PHASE 10.6 — RECOVERY HISTORY AND LOOP PREVENTION
# ==================================================================

RECOVERY_CONFIDENCE_HIGH = "high"
RECOVERY_CONFIDENCE_MEDIUM = "medium"
RECOVERY_CONFIDENCE_LOW = "low"

# A failure of the same kind may be attempted this many times in total,
# provided every repeat follows new information.
MAX_RECOVERY_ATTEMPTS = 3

# The same (failed action, recommended recovery) pair this many times
# in a row indicates a recovery loop.
REPEATED_DECISION_THRESHOLD = 3


@dataclass(frozen=True)
class RecoveryAttempt:
    """Immutable record of one failure and the recovery recommended."""

    iteration: int
    action: str
    failure_type: str
    category: str
    recovery_action: str | None
    state_signature: str
    message: str

    @property
    def failure_key(self) -> tuple[str, str]:
        return (self.action, self.failure_type)

    @property
    def decision_key(self) -> tuple[str, str | None]:
        return (self.action, self.recovery_action)


@dataclass(frozen=True)
class RecoveryAssessment:
    """Advisory assessment of whether recovery is still meaningful.

    ``confidence`` is recovery confidence: how likely another attempt is
    to be useful given the attempt history. It is separate from the
    Phase 9 decision confidence, which is never overridden. A low
    recovery confidence only ever leads to escalation, never to more
    permissions.
    """

    attempt: RecoveryAttempt
    attempts: int
    new_information: bool
    repeated_failure: bool
    repeated_decision: bool
    confidence: str
    reason: str

    @property
    def should_escalate(self) -> bool:
        return self.confidence == RECOVERY_CONFIDENCE_LOW


class RecoveryHistory:
    """Append-only record of failures across autonomous iterations."""

    def __init__(self) -> None:
        self._attempts: list[RecoveryAttempt] = []

    @property
    def attempts(self) -> tuple[RecoveryAttempt, ...]:
        return tuple(self._attempts)

    def __len__(self) -> int:
        return len(self._attempts)

    def record(
        self,
        iteration: int,
        result: ActionResult,
        decision: RecoveryDecision,
        state_signature: str,
    ) -> RecoveryAssessment:
        """Record a failure and assess whether recovery remains useful.

        Raises:
            ValueError: for a successful result or out-of-order iteration.
        """

        if result.success:
            raise ValueError("Only failed actions are recorded as recovery.")

        if self._attempts and iteration <= self._attempts[-1].iteration:
            raise ValueError(
                "Recovery attempts must be recorded in increasing "
                "iteration order."
            )

        attempt = RecoveryAttempt(
            iteration=iteration,
            action=result.action,
            failure_type=decision.failure_type,
            category=decision.category,
            recovery_action=decision.recovery_action,
            state_signature=state_signature,
            message=result.message,
        )

        earlier_same = [
            previous
            for previous in self._attempts
            if previous.failure_key == attempt.failure_key
        ]

        self._attempts.append(attempt)

        attempts = len(earlier_same) + 1

        # A repeat is informative only if the state changed since the
        # last failure of the same kind.
        new_information = (
            not earlier_same
            or earlier_same[-1].state_signature != state_signature
        )

        repeated_failure = attempts > 1 and not new_information

        recent = self._attempts[-REPEATED_DECISION_THRESHOLD:]
        repeated_decision = (
            len(recent) == REPEATED_DECISION_THRESHOLD
            and len({a.decision_key for a in recent}) == 1
        )

        if repeated_failure:
            confidence = RECOVERY_CONFIDENCE_LOW
            reason = (
                f"'{attempt.action}' failed again ({attempt.failure_type}) "
                "without any new information; retrying would repeat the "
                "same failure."
            )
        elif repeated_decision:
            confidence = RECOVERY_CONFIDENCE_LOW
            reason = (
                f"The same recovery decision for '{attempt.action}' was "
                f"made {REPEATED_DECISION_THRESHOLD} times in a row; "
                "recovery is looping."
            )
        elif attempts >= MAX_RECOVERY_ATTEMPTS:
            confidence = RECOVERY_CONFIDENCE_LOW
            reason = (
                f"'{attempt.action}' has failed {attempts} times "
                f"({attempt.failure_type}); the recovery limit is reached."
            )
        elif attempts > 1:
            confidence = RECOVERY_CONFIDENCE_MEDIUM
            reason = (
                f"'{attempt.action}' failed again, but the state changed "
                "since the last failure; one more attempt is reasonable."
            )
        else:
            confidence = RECOVERY_CONFIDENCE_HIGH
            reason = "First failure of this kind; recovery may proceed."

        return RecoveryAssessment(
            attempt=attempt,
            attempts=attempts,
            new_information=new_information,
            repeated_failure=repeated_failure,
            repeated_decision=repeated_decision,
            confidence=confidence,
            reason=reason,
        )
