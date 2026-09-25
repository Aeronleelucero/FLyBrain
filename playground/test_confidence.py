from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.confidence import (
    ConfidenceAssessment,
    assess_confidence,
)


def test_confidence_returns_assessment():
    result = ActionResult(
        action="inspect_files",
        success=True,
        message="Files inspected.",
    )

    assessment = assess_confidence(
        result,
        CodingState(task="Inspect workspace"),
    )

    assert isinstance(assessment, ConfidenceAssessment)
    assert assessment.action == "inspect_files"
    assert 0.0 <= assessment.score <= 1.0
    assert assessment.level in {"low", "medium", "high"}


def test_successful_observation_provides_positive_evidence():
    result = ActionResult(
        action="inspect_files",
        success=True,
        message="Files inspected.",
    )

    assessment = assess_confidence(
        result,
        CodingState(task="Inspect workspace"),
    )

    assert assessment.score > 0.50
    assert "The selected action reported success." in assessment.evidence


def test_failed_action_reduces_confidence():
    result = ActionResult(
        action="run_tests",
        success=False,
        message="Tests failed.",
    )

    assessment = assess_confidence(
        result,
        CodingState(task="Run tests"),
    )

    assert assessment.score < 0.50
    assert assessment.level == "low"


def test_inspected_error_increases_evidence():
    result = ActionResult(
        action="fix_error",
        success=True,
        message="Error inspected.",
    )

    state = CodingState(task="Repair code")
    state.error_inspected = True

    assessment = assess_confidence(result, state)

    assert "The latest error has been inspected." in assessment.evidence


def test_current_file_and_content_increase_confidence():
    result = ActionResult(
        action="read_file",
        success=True,
        message="File read.",
    )

    state = CodingState(task="Read file")
    state.current_file = "example.py"
    state.current_file_content = "print('hello')"

    assessment = assess_confidence(result, state)

    assert "A relevant current file has been identified." in assessment.evidence
    assert "The current file content is available." in assessment.evidence
    assert assessment.score >= 0.80


def test_missing_file_information_creates_uncertainty():
    result = ActionResult(
        action="read_file",
        success=True,
        message="File read.",
    )

    state = CodingState(task="Read file")

    assessment = assess_confidence(result, state)

    assert any(
        "No current file has been identified." in item
        for item in assessment.uncertainties
    )


def test_failed_tests_create_uncertainty():
    result = ActionResult(
        action="run_tests",
        success=True,
        message="Tests completed.",
    )

    state = CodingState(task="Run tests")
    state.tests_run = True
    state.tests_passed = False

    assessment = assess_confidence(result, state)

    assert any(
        "The latest test run did not pass." in item
        for item in assessment.uncertainties
    )


def test_human_input_requires_human_review():
    result = ActionResult(
        action="propose_repair",
        success=True,
        message="Repair proposed.",
    )

    state = CodingState(task="Repair code")
    state.user_input_needed = True

    assessment = assess_confidence(result, state)

    assert assessment.requires_human_review is True
    assert assessment.level in {"low", "medium", "high"}
    assert any(
        "human input" in item.lower()
        for item in assessment.uncertainties
    )


def test_approve_repair_always_requires_human_review():
    result = ActionResult(
        action="approve_repair",
        success=True,
        message="Repair approved.",
    )

    assessment = assess_confidence(
        result,
        CodingState(task="Repair code"),
    )

    assert assessment.requires_human_review is True
    assert any(
        "explicitly human-controlled" in item
        for item in assessment.risks
    )


def test_confidence_does_not_grant_permission():
    result = ActionResult(
        action="propose_repair",
        success=True,
        message="Repair proposed.",
    )

    state = CodingState(task="Repair code")
    state.error_inspected = True
    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.tests_run = True
    state.tests_passed = True

    assessment = assess_confidence(result, state)

    assert assessment.score >= 0.80

    # High confidence is evidence about certainty.
    # It does not automatically approve the repair.
    assert assessment.requires_human_review is False
    assert assessment.action == "propose_repair"
