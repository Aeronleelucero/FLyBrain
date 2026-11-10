"""Repair proposal binding for FLY-CODER Phase 10.7.

A proposal is recorded together with everything an approval must be
bound to:

* the affected file,
* the exact original content the proposal was computed from,
* the exact proposed content,
* the code version at proposal time,
* a deterministic fingerprint of file, original, and proposed content.

Human approval is an ``ApprovalRecord`` bound to one fingerprint and one
code version. A changed proposal, a changed original file, or a newer
code version invalidates it; a new approval is then required. The
boolean ``repair_approved`` keeps its old meaning (an approved repair
was applied) and is never used as authorization on its own.

Only the human decision workflow (``human_intervention``) may create an
``ApprovalRecord``. No action, recovery recommendation, or confidence
score can.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from flycoder.state import CodingState


def compute_proposal_fingerprint(
    file: str,
    original_content: str,
    proposed_content: str,
) -> str:
    """Return a deterministic SHA-256 fingerprint of a proposal."""

    payload = json.dumps(
        {
            "file": file,
            "original_content": original_content,
            "proposed_content": proposed_content,
        },
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ApprovalRecord:
    """Explicit human approval of one exact proposal."""

    fingerprint: str
    file: str
    code_version: int
    intervention_id: str | None = None

    def __post_init__(self) -> None:
        if not self.fingerprint or not isinstance(self.fingerprint, str):
            raise ValueError("fingerprint must be a non-empty string")

        if not self.file or not isinstance(self.file, str):
            raise ValueError("file must be a non-empty string")

        if (
            not isinstance(self.code_version, int)
            or isinstance(self.code_version, bool)
            or self.code_version < 0
        ):
            raise ValueError("code_version must be a non-negative integer")


def record_proposal(
    state: CodingState,
    *,
    file: str,
    original_content: str,
    proposed_content: str,
    description: str,
) -> str:
    """Record a new proposal and return its fingerprint.

    Any earlier approval is discarded: a new proposal always needs a new
    human approval. Nothing is written to the workspace.
    """

    fingerprint = compute_proposal_fingerprint(
        file,
        original_content,
        proposed_content,
    )

    state.repair_description = description
    state.proposed_content = proposed_content
    state.proposal_original_content = original_content
    state.proposal_code_version = state.code_version
    state.proposal_fingerprint = fingerprint
    state.repair_proposed = True
    state.repair_approved = False
    state.repair_applied = False
    state.repair_approval = None

    return fingerprint


def discard_proposal(state: CodingState, *, rejected: bool) -> None:
    """Discard an unapplied proposal, optionally remembering a rejection.

    A rejected fingerprint is never offered again, so the agent cannot
    re-present a proposal the human already turned down.
    """

    if state.repair_applied:
        raise ValueError("An applied repair cannot be discarded; roll it back.")

    if rejected and state.proposal_fingerprint:
        if state.proposal_fingerprint not in state.rejected_proposals:
            state.rejected_proposals.append(state.proposal_fingerprint)

    state.repair_proposed = False
    state.repair_approved = False
    state.repair_description = None
    state.proposed_content = None
    state.proposal_original_content = None
    state.proposal_code_version = None
    state.proposal_fingerprint = None
    state.repair_approval = None


def current_proposal_fingerprint(state: CodingState) -> str | None:
    """Recompute the fingerprint of the proposal currently in state."""

    if (
        not state.current_file
        or state.proposal_original_content is None
        or state.proposed_content is None
    ):
        return None

    return compute_proposal_fingerprint(
        state.current_file,
        state.proposal_original_content,
        state.proposed_content,
    )


def proposal_problems(state: CodingState) -> list[str]:
    """Return reasons the current proposal cannot be approved as-is."""

    problems: list[str] = []

    if not state.repair_proposed:
        problems.append("No repair proposal exists.")
        return problems

    if state.repair_applied:
        problems.append("The repair has already been applied.")

    recomputed = current_proposal_fingerprint(state)

    if recomputed is None or not state.proposal_fingerprint:
        problems.append(
            "The proposal has no recorded fingerprint and cannot be bound "
            "to an approval."
        )
    elif recomputed != state.proposal_fingerprint:
        problems.append("The proposal changed after it was recorded.")

    if state.proposal_code_version != state.code_version:
        problems.append(
            "The code changed after the proposal was created "
            f"(proposal version {state.proposal_code_version}, current "
            f"version {state.code_version})."
        )

    return problems


def approval_problems(state: CodingState) -> list[str]:
    """Return reasons the recorded approval does not authorize applying.

    An empty list means a human approved exactly the current proposal for
    the current code version. The on-disk original is checked separately
    by the approval action, which can read the workspace.
    """

    problems = proposal_problems(state)
    approval = state.repair_approval

    if not isinstance(approval, ApprovalRecord):
        problems.append("No explicit human approval is recorded.")
        return problems

    if approval.fingerprint != state.proposal_fingerprint:
        problems.append(
            "The recorded approval is for a different proposal."
        )

    if approval.file != state.current_file:
        problems.append("The recorded approval is for a different file.")

    if approval.code_version != state.code_version:
        problems.append(
            "The recorded approval is for an older code version."
        )

    return problems


def record_human_approval(
    state: CodingState,
    fingerprint: str,
    *,
    intervention_id: str | None = None,
) -> ApprovalRecord:
    """Record explicit human approval of the reviewed proposal.

    This is the only place an ``ApprovalRecord`` is created. It must only
    be called from a human decision path, after the human saw the exact
    proposal identified by ``fingerprint``.

    Raises:
        ValueError: if the proposal is not approvable or the fingerprint
            does not match the current proposal.
    """

    problems = proposal_problems(state)

    if problems:
        raise ValueError("Proposal cannot be approved: " + " ".join(problems))

    if fingerprint != state.proposal_fingerprint:
        raise ValueError(
            "The reviewed proposal fingerprint does not match the current "
            "proposal; a new review is required."
        )

    record = ApprovalRecord(
        fingerprint=fingerprint,
        file=state.current_file,
        code_version=state.code_version,
        intervention_id=intervention_id,
    )

    state.repair_approval = record

    return record
