"""Structured human intervention for FLY-CODER Phase 10.7.

When the agent stops at the human boundary, an ``InterventionRequest``
describes why, what the human may decide, and (for repair approval) the
exact proposal: file, original and proposed content, unified diff,
fingerprint, and code version.

A ``HumanDecision`` answers one specific request. ``apply_human_decision``
validates it against the request *and* the current state before changing
anything:

* APPROVE            records an approval bound to the reviewed fingerprint.
                     The repair is then applied by the normal
                     ``approve_repair`` action through ``agent.run_once``
                     and the guardrails, and verified afterwards.
* REJECT             discards an unapplied proposal (the file is never
                     touched) and remembers its fingerprint; for a failed
                     applied repair it authorizes a rollback instead.
* REQUEST_CHANGES    discards the proposal, keeps the feedback, and returns
                     to safe preparation (the file is re-read).
* PROVIDE_INFORMATION records the information (and an optional workspace
                     file hint) for the next diagnostic step.
* CANCEL_TASK        records the cancellation; the session ends the task.

No decision applies code, runs an action, or completes a task directly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import PurePosixPath

from flycoder.state import CodingState
from flycoder.tools.changes import ChangeProposal, build_change_diff
from flycoder.tools.completion_evidence import CompletionEvidence
from flycoder.tools.human_input import (
    HUMAN_INPUT_APPLIED_REPAIR_FAILED,
    HUMAN_INPUT_INSUFFICIENT_DIAGNOSIS,
    HUMAN_INPUT_NO_SAFE_REPAIR,
    HUMAN_INPUT_REASONS,
    HUMAN_INPUT_REPAIR_APPROVAL,
    HUMAN_INPUT_STALE_PROPOSAL,
    clear_human_input,
)
from flycoder.tools.proposals import (
    current_proposal_fingerprint,
    discard_proposal,
    record_human_approval,
)


DECISION_APPROVE = "APPROVE"
DECISION_REJECT = "REJECT"
DECISION_REQUEST_CHANGES = "REQUEST_CHANGES"
DECISION_PROVIDE_INFORMATION = "PROVIDE_INFORMATION"
DECISION_CANCEL_TASK = "CANCEL_TASK"

DECISIONS = {
    DECISION_APPROVE,
    DECISION_REJECT,
    DECISION_REQUEST_CHANGES,
    DECISION_PROVIDE_INFORMATION,
    DECISION_CANCEL_TASK,
}

# Human input requested without a recorded reason.
REASON_UNSPECIFIED = "unspecified"

ALLOWED_DECISIONS: dict[str, tuple[str, ...]] = {
    HUMAN_INPUT_REPAIR_APPROVAL: (
        DECISION_APPROVE,
        DECISION_REJECT,
        DECISION_REQUEST_CHANGES,
        DECISION_CANCEL_TASK,
    ),
    HUMAN_INPUT_INSUFFICIENT_DIAGNOSIS: (
        DECISION_PROVIDE_INFORMATION,
        DECISION_CANCEL_TASK,
    ),
    HUMAN_INPUT_NO_SAFE_REPAIR: (
        DECISION_PROVIDE_INFORMATION,
        DECISION_CANCEL_TASK,
    ),
    HUMAN_INPUT_STALE_PROPOSAL: (
        DECISION_REQUEST_CHANGES,
        DECISION_REJECT,
        DECISION_CANCEL_TASK,
    ),
    HUMAN_INPUT_APPLIED_REPAIR_FAILED: (
        DECISION_REJECT,
        DECISION_CANCEL_TASK,
    ),
    REASON_UNSPECIFIED: (
        DECISION_PROVIDE_INFORMATION,
        DECISION_CANCEL_TASK,
    ),
}

_ERROR_CONTEXT_LINES = 5


class HumanDecisionError(ValueError):
    """Raised when a human decision is invalid for the current request."""


# ==================================================================
# INTERVENTION REQUESTS
# ==================================================================


@dataclass(frozen=True)
class InterventionRequest:
    """Immutable description of one required human intervention."""

    intervention_id: str
    task: str
    task_intent: str | None
    reason: str
    reason_description: str
    task_status: str | None
    allowed_decisions: tuple[str, ...]
    file: str | None
    original_content: str | None
    proposed_content: str | None
    diff: str | None
    description: str | None
    proposal_fingerprint: str | None
    proposal_code_version: int | None
    code_version: int
    verification_status: str | None
    error_context: str | None
    human_notes: tuple[str, ...]
    repeated: bool = False
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        compare=False,
    )

    @property
    def is_repair_approval(self) -> bool:
        return self.reason == HUMAN_INPUT_REPAIR_APPROVAL

    @property
    def has_reviewed_proposal(self) -> bool:
        """Whether the human was shown exact proposal contents."""

        return self.original_content is not None and self.proposed_content is not None

    def stale_reasons(self, state: CodingState) -> list[str]:
        """Return why this request no longer matches the state.

        The request is compared field by field against the live state.
        For a reviewed proposal the actual contents are compared and the
        fingerprint is recomputed, so a proposal whose contents changed
        while its stored fingerprint did not is still detected as stale.
        """

        reasons: list[str] = []

        if not state.user_input_needed:
            reasons.append("The task is no longer waiting for human input.")

        current_reason = state.human_input_reason or REASON_UNSPECIFIED

        if current_reason != self.reason:
            reasons.append(
                f"The intervention reason changed to '{current_reason}'."
            )

        if state.task != self.task:
            reasons.append("The task changed after the request.")

        if state.task_intent != self.task_intent:
            reasons.append("The task intent changed after the request.")

        if state.current_file != self.file:
            reasons.append("The selected file changed after the request.")

        if state.code_version != self.code_version:
            reasons.append("The code version changed after the request.")

        if self.proposal_fingerprint != state.proposal_fingerprint:
            reasons.append("The proposal changed after the request.")

        if self.has_reviewed_proposal:
            if state.proposal_original_content != self.original_content:
                reasons.append(
                    "The proposal's original content changed after the "
                    "request."
                )

            if state.proposed_content != self.proposed_content:
                reasons.append(
                    "The proposed content changed after the request."
                )

            if state.proposal_code_version != self.proposal_code_version:
                reasons.append(
                    "The proposal code version changed after the request."
                )

            if current_proposal_fingerprint(state) != self.proposal_fingerprint:
                reasons.append(
                    "The current proposal no longer matches the reviewed "
                    "fingerprint."
                )

        return reasons

    def is_valid(self, state: CodingState) -> bool:
        return not self.stale_reasons(state)

    def render(self) -> str:
        """Return a human-readable description of the request."""

        lines = [
            f"Human intervention required [{self.intervention_id}]",
            f"Task: {self.task}",
            f"Reason: {self.reason} — {self.reason_description}",
        ]

        if self.task_status:
            lines.append(f"Task status: {self.task_status}")

        if self.file:
            lines.append(f"File: {self.file}")

        if self.description:
            lines.append(f"Proposal: {self.description}")

        if self.proposal_fingerprint:
            lines.append(f"Proposal fingerprint: {self.proposal_fingerprint}")

        lines.append(f"Code version: {self.code_version}")
        lines.append(
            "Verification status: "
            f"{self.verification_status or 'not evaluated'}"
        )

        if self.error_context:
            lines.append("Error context:")
            lines.extend(f"  {line}" for line in self.error_context.splitlines())

        if self.diff is not None:
            lines.append("Proposed diff:")
            lines.append(self.diff if self.diff else "  (no textual changes)")

        if self.repeated:
            lines.append(
                "This intervention has repeated without progress; only "
                "cancellation is accepted."
            )

        lines.append("Allowed decisions: " + ", ".join(self.allowed_decisions))

        return "\n".join(lines)


def _error_context(state: CodingState) -> str | None:
    if not state.last_error:
        return None

    lines = [line for line in state.last_error.splitlines() if line.strip()]

    return "\n".join(lines[-_ERROR_CONTEXT_LINES:])


def _intervention_id(state: CodingState, reason: str) -> str:
    """Deterministic id: the same situation always has the same id."""

    payload = repr(
        (
            state.task,
            reason,
            state.current_file,
            state.proposal_fingerprint,
            state.code_version,
            state.last_error,
            tuple(state.rejected_proposals),
        )
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_intervention_request(
    state: CodingState,
    *,
    task_status: str | None = None,
    evidence: CompletionEvidence | None = None,
) -> InterventionRequest:
    """Describe the human intervention the current state requires.

    Raises:
        ValueError: if the state is not waiting for human input.
    """

    if not state.user_input_needed:
        raise ValueError("The task is not waiting for human input.")

    reason = state.human_input_reason or REASON_UNSPECIFIED
    description = HUMAN_INPUT_REASONS.get(
        reason,
        "Explicit human input is required.",
    )

    has_proposal = (
        state.repair_proposed
        and state.current_file is not None
        and state.proposed_content is not None
        and state.proposal_original_content is not None
    )

    diff = None

    if has_proposal:
        diff = build_change_diff(
            ChangeProposal(
                file=state.current_file,
                original_content=state.proposal_original_content,
                proposed_content=state.proposed_content,
            )
        )

    return InterventionRequest(
        intervention_id=_intervention_id(state, reason),
        task=state.task,
        task_intent=state.task_intent,
        reason=reason,
        reason_description=description,
        task_status=task_status,
        allowed_decisions=ALLOWED_DECISIONS.get(
            reason,
            ALLOWED_DECISIONS[REASON_UNSPECIFIED],
        ),
        file=state.current_file,
        original_content=state.proposal_original_content if has_proposal else None,
        proposed_content=state.proposed_content if has_proposal else None,
        diff=diff,
        description=state.repair_description if has_proposal else None,
        proposal_fingerprint=state.proposal_fingerprint,
        proposal_code_version=state.proposal_code_version,
        code_version=state.code_version,
        verification_status=evidence.status if evidence is not None else None,
        error_context=_error_context(state),
        human_notes=tuple(state.human_notes),
    )


def restrict_to_cancellation(request: InterventionRequest) -> InterventionRequest:
    """Return a copy that only accepts cancellation (repeated requests)."""

    return replace(
        request,
        allowed_decisions=(DECISION_CANCEL_TASK,),
        repeated=True,
    )


# ==================================================================
# HUMAN DECISIONS
# ==================================================================


@dataclass(frozen=True)
class HumanDecision:
    """An explicit human answer to one intervention request."""

    kind: str
    intervention_id: str
    proposal_fingerprint: str | None = None
    confirmed: bool = False
    information: str = ""
    file: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in DECISIONS:
            raise HumanDecisionError(f"Unknown human decision: {self.kind!r}")

        if not isinstance(self.intervention_id, str) or not self.intervention_id:
            raise HumanDecisionError("intervention_id must be a non-empty string")


@dataclass(frozen=True)
class DecisionRecord:
    """Immutable record of a validated, applied human decision."""

    intervention_id: str
    kind: str
    intervention_reason: str
    proposal_fingerprint: str | None
    information: str
    outcome: str
    decided_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        compare=False,
    )

    def describe(self) -> str:
        line = f"{self.kind} ({self.intervention_reason}): {self.outcome}"

        if self.information:
            line += f" Note: {self.information}"

        return line


def _validated_hint(file: str | None) -> str | None:
    """Accept only a relative path that stays inside the workspace."""

    if file is None:
        return None

    path = PurePosixPath(file.replace("\\", "/"))

    if not file.strip() or path.is_absolute() or ".." in path.parts:
        raise HumanDecisionError(
            "A file hint must be a relative path inside the workspace."
        )

    return str(path)


def apply_human_decision(
    state: CodingState,
    request: InterventionRequest,
    decision: HumanDecision,
) -> DecisionRecord:
    """Validate a human decision and apply its effect on the task state.

    Nothing here writes files, executes actions, or completes the task.
    An approval only records an ``ApprovalRecord``; applying it is left to
    the ``approve_repair`` action, which re-validates everything.

    Raises:
        HumanDecisionError: if the decision does not match the request,
            the request is stale, or the decision is not allowed.
    """

    if not isinstance(request, InterventionRequest):
        raise HumanDecisionError("request must be an InterventionRequest")

    if not isinstance(decision, HumanDecision):
        raise HumanDecisionError("decision must be a HumanDecision")

    if decision.intervention_id != request.intervention_id:
        raise HumanDecisionError(
            "The decision answers a different intervention request."
        )

    stale = request.stale_reasons(state)

    if stale:
        raise HumanDecisionError(
            "The intervention request is stale: " + " ".join(stale)
        )

    if decision.kind not in request.allowed_decisions:
        raise HumanDecisionError(
            f"{decision.kind} is not allowed for '{request.reason}'. "
            f"Allowed: {', '.join(request.allowed_decisions)}."
        )

    information = decision.information.strip()

    def record(outcome: str) -> DecisionRecord:
        return DecisionRecord(
            intervention_id=request.intervention_id,
            kind=decision.kind,
            intervention_reason=request.reason,
            proposal_fingerprint=request.proposal_fingerprint,
            information=information,
            outcome=outcome,
        )

    if decision.kind == DECISION_APPROVE:
        if not decision.confirmed:
            raise HumanDecisionError(
                "Approval requires explicit confirmation after reviewing "
                "the diff."
            )

        if (
            not decision.proposal_fingerprint
            or decision.proposal_fingerprint != request.proposal_fingerprint
        ):
            raise HumanDecisionError(
                "The approved fingerprint does not match the reviewed "
                "proposal."
            )

        try:
            record_human_approval(
                state,
                decision.proposal_fingerprint,
                intervention_id=request.intervention_id,
            )
        except ValueError as error:
            raise HumanDecisionError(str(error)) from error

        clear_human_input(state)

        return record(
            "Approval recorded for this exact proposal; it will be applied "
            "and verified through the normal workflow."
        )

    if decision.kind == DECISION_REJECT:
        if request.reason == HUMAN_INPUT_APPLIED_REPAIR_FAILED:
            if state.proposal_fingerprint and (
                state.proposal_fingerprint not in state.rejected_proposals
            ):
                state.rejected_proposals.append(state.proposal_fingerprint)

            state.rollback_authorized = True
            clear_human_input(state)

            return record(
                "The applied repair was rejected; rollback is authorized and "
                "will run through the normal workflow."
            )

        discard_proposal(state, rejected=True)
        clear_human_input(state)

        if information:
            state.human_notes.append(f"Rejected proposal: {information}")

        return record("Proposal rejected and discarded; no file was changed.")

    if decision.kind == DECISION_REQUEST_CHANGES:
        if not information:
            raise HumanDecisionError(
                "Requesting changes requires feedback describing them."
            )

        # The reviewed proposal will not be offered again; a stale
        # proposal is merely discarded so a fresh one can be prepared.
        discard_proposal(
            state,
            rejected=request.reason == HUMAN_INPUT_REPAIR_APPROVAL,
        )
        state.human_notes.append(f"Requested changes: {information}")

        # Return to safe preparation: re-read the file before any new
        # proposal is computed.
        state.current_file_content = None
        clear_human_input(state)

        return record(
            "Proposal discarded; the task returns to preparation with the "
            "feedback recorded."
        )

    if decision.kind == DECISION_PROVIDE_INFORMATION:
        if not information:
            raise HumanDecisionError("Providing information requires text.")

        hint = _validated_hint(decision.file)
        state.human_notes.append(information)

        if hint is not None:
            state.diagnosis_hint_file = hint

            if request.reason == HUMAN_INPUT_INSUFFICIENT_DIAGNOSIS:
                # Diagnose again with the human-identified file.
                state.error_inspected = False
            elif request.reason == HUMAN_INPUT_NO_SAFE_REPAIR:
                # Investigate the human-identified file instead.
                state.current_file = hint
                state.current_file_content = None

        clear_human_input(state)

        return record("Information recorded for the next diagnostic step.")

    # CANCEL_TASK: the session ends the task; the state is left as it is.
    if information:
        state.human_notes.append(f"Cancelled: {information}")

    return record("Task cancelled by a human.")


# ==================================================================
# INTERACTIVE REVIEW (shared by the command-line interfaces)
# ==================================================================


def prompt_for_approval(
    request: InterventionRequest,
    *,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None] = print,
    interactive: bool,
) -> HumanDecision | None:
    """Show the exact proposal and ask for an explicit decision.

    Fails closed: without an interactive terminal, on end of input, or
    on any answer other than ``yes``/``no``, no decision is returned and
    nothing is approved.
    """

    if not request.is_repair_approval:
        raise ValueError("Only repair approval requests can be reviewed here.")

    output_fn(request.render())
    output_fn(
        f"Approving writes exactly this content to {request.file}. The "
        "approval is bound to the fingerprint above; any change to the "
        "proposal or the file requires a new review. The change is "
        "verified after it is applied."
    )

    if not interactive:
        output_fn(
            "Approval requires interactive confirmation. No changes were "
            "applied."
        )
        return None

    try:
        answer = input_fn(
            "Apply this exact change? Type 'yes' to approve or 'no' to "
            "reject: "
        )
    except (EOFError, KeyboardInterrupt):
        output_fn("No confirmation received. No changes were applied.")
        return None

    answer = (answer or "").strip().lower()

    if answer == "yes":
        return HumanDecision(
            kind=DECISION_APPROVE,
            intervention_id=request.intervention_id,
            proposal_fingerprint=request.proposal_fingerprint,
            confirmed=True,
        )

    if answer == "no":
        return HumanDecision(
            kind=DECISION_REJECT,
            intervention_id=request.intervention_id,
            information="Rejected during interactive review.",
        )

    output_fn("No decision recorded. No changes were applied.")
    return None
