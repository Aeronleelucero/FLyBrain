"""Explicit human-input reasons for FLY-CODER Phase 10.6.

``CodingState.user_input_needed`` is the guardrail boundary: while it is
set, every action is blocked (``guardrails.check_approval_boundary``).
Phase 10.6 keeps that boundary unchanged and records *why* it was set,
so three situations are no longer ambiguous:

1. Safe diagnosis continues
       ``error_inspected`` is True and ``user_input_needed`` is False.
       Read-only investigation (``read_file``) and proposal preparation
       proceed through the normal guardrails.
2. A repair proposal awaits approval
       ``user_input_needed`` with reason ``REPAIR_APPROVAL``.
3. Other explicit human input is required
       ``user_input_needed`` with any other reason.

Setting or clearing the boundary must go through ``request_human_input``
and ``clear_human_input`` so the flag and its reason never disagree.
Nothing here grants approval.
"""

from __future__ import annotations

from flycoder.state import CodingState


HUMAN_INPUT_REPAIR_APPROVAL = "repair_approval"
HUMAN_INPUT_INSUFFICIENT_DIAGNOSIS = "insufficient_diagnosis"
HUMAN_INPUT_NO_SAFE_REPAIR = "no_safe_repair"
HUMAN_INPUT_STALE_PROPOSAL = "stale_proposal"
HUMAN_INPUT_APPLIED_REPAIR_FAILED = "applied_repair_failed"

HUMAN_INPUT_REASONS = {
    HUMAN_INPUT_REPAIR_APPROVAL: (
        "A repair proposal is waiting for explicit human approval."
    ),
    HUMAN_INPUT_INSUFFICIENT_DIAGNOSIS: (
        "The error was inspected, but no failing file could be identified "
        "for further investigation."
    ),
    HUMAN_INPUT_NO_SAFE_REPAIR: (
        "No safe automatic repair could be prepared from the available "
        "evidence."
    ),
    HUMAN_INPUT_STALE_PROPOSAL: (
        "The repair proposal is stale because the file changed after it "
        "was created."
    ),
    HUMAN_INPUT_APPLIED_REPAIR_FAILED: (
        "An approved repair was applied but the failure persists; a human "
        "must decide between rollback and a new proposal."
    ),
}


def request_human_input(state: CodingState, reason: str) -> None:
    """Stop at the human boundary for an explicit, known reason."""

    if reason not in HUMAN_INPUT_REASONS:
        raise ValueError(f"Unknown human input reason: {reason!r}")

    state.user_input_needed = True
    state.human_input_reason = reason


def clear_human_input(state: CodingState) -> None:
    """Clear the human boundary after the required input was given."""

    state.user_input_needed = False
    state.human_input_reason = None


def awaiting_repair_approval(state: CodingState) -> bool:
    """Return whether execution is stopped for repair approval."""

    return (
        state.user_input_needed
        and state.human_input_reason == HUMAN_INPUT_REPAIR_APPROVAL
    )


def describe_human_input(state: CodingState) -> str | None:
    """Return a human-readable explanation of the current boundary."""

    if not state.user_input_needed:
        return None

    if state.human_input_reason in HUMAN_INPUT_REASONS:
        return HUMAN_INPUT_REASONS[state.human_input_reason]

    return "Human input is required (no reason was recorded)."


def human_input_violations(state: CodingState) -> list[str]:
    """Return contradictions between the boundary and its reason."""

    violations: list[str] = []

    reason = state.human_input_reason

    if reason is not None and reason not in HUMAN_INPUT_REASONS:
        violations.append(f"Unknown human_input_reason {reason!r}.")

    if reason is not None and not state.user_input_needed:
        violations.append(
            "human_input_reason cannot be set when user_input_needed is "
            "false."
        )

    if reason == HUMAN_INPUT_REPAIR_APPROVAL and not state.repair_proposed:
        violations.append(
            "Repair approval cannot be pending without a repair proposal."
        )

    return violations
