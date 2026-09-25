"""Decision confidence tools for FLY-CODER Phase 9.5."""

from __future__ import annotations

from dataclasses import dataclass, field

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState


@dataclass
class ConfidenceAssessment:
    """Describe confidence in a proposed agent decision."""

    action: str
    score: float
    level: str
    evidence: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    requires_human_review: bool = False
    reason: str = ""


def _level_for_score(score: float) -> str:
    """Convert a normalized confidence score into a confidence level."""

    if score >= 0.80:
        return "high"

    if score >= 0.50:
        return "medium"

    return "low"


def _clamp_score(score: float) -> float:
    """Keep confidence scores inside the normalized 0..1 range."""

    return max(0.0, min(1.0, score))


def assess_confidence(
    result: ActionResult,
    state: CodingState,
) -> ConfidenceAssessment:
    """Assess confidence in the result and current decision context.

    Confidence describes evidence quality and decision certainty.
    It does not grant execution permission or repair approval.
    """

    evidence: list[str] = []
    uncertainties: list[str] = []
    risks: list[str] = []

    score = 0.50

    if result.success:
        score += 0.20
        evidence.append("The selected action reported success.")
    else:
        score -= 0.15
        uncertainties.append("The selected action reported failure.")

    if state.last_error:
        uncertainties.append("The workspace contains a recorded error.")
        score -= 0.05

    if state.error_inspected:
        evidence.append("The latest error has been inspected.")
        score += 0.10

    if state.current_file:
        evidence.append("A relevant current file has been identified.")
        score += 0.05
    else:
        uncertainties.append("No current file has been identified.")
        score -= 0.05

    if state.current_file_content is not None:
        evidence.append("The current file content is available.")
        score += 0.10
    elif state.current_file:
        uncertainties.append("The current file has not been read yet.")
        score -= 0.05

    if state.tests_run:
        evidence.append("Tests have been executed.")
        score += 0.05

        if state.tests_passed:
            evidence.append("The latest test run passed.")
            score += 0.05
        else:
            uncertainties.append("The latest test run did not pass.")
            score -= 0.10

    if state.user_input_needed:
        uncertainties.append("The workflow is waiting for human input or approval.")
        risks.append("Execution must stop at the human approval boundary.")
        score -= 0.20

    if result.action in {
        "write_code",
        "propose_repair",
        "approve_repair",
        "rollback_repair",
    }:
        risks.append(
            "This action can affect the repair workflow and must respect "
            "existing safety controls."
        )

    if result.action == "approve_repair":
        risks.append("Repair approval must always remain explicitly human-controlled.")
        score -= 0.30

    score = _clamp_score(score)

    requires_human_review = (
        state.user_input_needed
        or result.action == "approve_repair"
        or score < 0.50
    )

    level = _level_for_score(score)

    if requires_human_review:
        reason = (
            "Confidence assessment indicates that human review or the "
            "existing approval boundary should control further execution."
        )
    elif level == "high":
        reason = (
            "Available evidence provides strong support for the current "
            "decision, while normal safety controls still apply."
        )
    elif level == "medium":
        reason = (
            "Some evidence supports the current decision, but additional "
            "evidence may improve certainty."
        )
    else:
        reason = (
            "Available evidence is insufficient for a confident decision."
        )

    return ConfidenceAssessment(
        action=result.action,
        score=score,
        level=level,
        evidence=evidence,
        uncertainties=uncertainties,
        risks=risks,
        requires_human_review=requires_human_review,
        reason=reason,
    )