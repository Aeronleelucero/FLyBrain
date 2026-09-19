"""Tests for Phase 7.7 verification integration."""

from pathlib import Path

from flycoder.tools.changes import ChangeProposal
from flycoder.tools.filesystem import Workspace
from flycoder.tools.regression import (
    RegressionVerificationResult,
)
from flycoder.tools.testing import (
    FailureAnalysisResult,
    TestExecutionResult,
)
from flycoder.tools.verification import (
    ChangeVerificationResult,
)
from flycoder.tools.verification_pipeline import (
    VerificationPipelineResult,
    build_verification_pipeline_report,
)


def execution(
    passed: bool,
    output: str = "",
) -> TestExecutionResult:
    """Build a deterministic execution result."""
    return TestExecutionResult(
        command=["pytest", "-q"],
        return_code=0 if passed else 1,
        output=output,
        passed=passed,
        timed_out=False,
        duration_seconds=0.1,
    )


def change_result(
    verified: bool,
) -> ChangeVerificationResult:
    """Build a deterministic change-verification result."""
    return ChangeVerificationResult(
        verified=verified,
        file="example.py",
        reason=(
            "actual file content matches proposed content"
            if verified
            else "content mismatch"
        ),
        expected_content="updated\n",
        actual_content=(
            "updated\n"
            if verified
            else "different\n"
        ),
    )


def regression_result(
    verified: bool,
) -> RegressionVerificationResult:
    """Build a deterministic regression result."""
    targeted = execution(
        passed=True,
        output="1 passed",
    )

    regression = execution(
        passed=verified,
        output=(
            "469 passed"
            if verified
            else "FAILED tests/test_other.py::test_other"
        ),
    )

    return RegressionVerificationResult(
        targeted=targeted,
        regression=regression,
        regressions_detected=not verified,
        reason=(
            "targeted tests and the full regression suite passed"
            if verified
            else "the full regression suite failed"
        ),
    )


def test_pipeline_is_verified_when_every_stage_passes():
    result = VerificationPipelineResult(
        change=change_result(True),
        targeted=execution(
            True,
            "5 passed",
        ),
        targeted_analysis=FailureAnalysisResult(
            passed=True,
        ),
        regression=regression_result(True),
        reason=(
            "change verification, targeted tests, "
            "and regression verification all passed"
        ),
    )

    assert result.change_verified is True
    assert result.targeted_passed is True
    assert result.regression_verified is True
    assert result.verified is True
    assert result.failed is False


def test_pipeline_fails_when_change_verification_fails():
    result = VerificationPipelineResult(
        change=change_result(False),
        targeted=None,
        targeted_analysis=None,
        regression=None,
        reason=(
            "change verification failed; "
            "tests were not executed"
        ),
    )

    assert result.change_verified is False
    assert result.targeted_passed is False
    assert result.regression_verified is False
    assert result.verified is False
    assert result.failed is True


def test_pipeline_fails_when_targeted_tests_fail():
    result = VerificationPipelineResult(
        change=change_result(True),
        targeted=execution(
            False,
            "FAILED tests/test_example.py::test_example",
        ),
        targeted_analysis=FailureAnalysisResult(
            failures=[],
            passed=False,
        ),
        regression=None,
        reason=(
            "change verification passed, but "
            "targeted tests failed"
        ),
    )

    assert result.change_verified is True
    assert result.targeted_passed is False
    assert result.regression_verified is False
    assert result.verified is False


def test_pipeline_fails_when_regression_fails():
    result = VerificationPipelineResult(
        change=change_result(True),
        targeted=execution(
            True,
            "5 passed",
        ),
        targeted_analysis=FailureAnalysisResult(
            passed=True,
        ),
        regression=regression_result(False),
        reason=(
            "change and targeted tests passed, "
            "but regression verification failed"
        ),
    )

    assert result.change_verified is True
    assert result.targeted_passed is True
    assert result.regression_verified is False
    assert result.verified is False


def test_report_contains_all_verification_stages():
    result = VerificationPipelineResult(
        change=change_result(True),
        targeted=execution(
            True,
            "5 passed",
        ),
        targeted_analysis=FailureAnalysisResult(
            passed=True,
        ),
        regression=regression_result(True),
        reason="all verification stages passed",
    )

    report = build_verification_pipeline_report(
        result
    )

    assert report["verified"] is True
    assert report["failed"] is False
    assert report["change_verified"] is True
    assert report["targeted_passed"] is True
    assert report["regression_verified"] is True

    assert report["change"]["verified"] is True
    assert report["targeted"]["passed"] is True
    assert report["targeted_analysis"]["passed"] is True
    assert report["regression"]["verified"] is True


def test_report_handles_skipped_stages():
    result = VerificationPipelineResult(
        change=change_result(False),
        targeted=None,
        targeted_analysis=None,
        regression=None,
        reason="change verification failed",
    )

    report = build_verification_pipeline_report(
        result
    )

    assert report["verified"] is False
    assert report["targeted"] is None
    assert report["targeted_analysis"] is None
    assert report["regression"] is None


def test_pipeline_result_is_safe_to_construct_without_execution():
    result = VerificationPipelineResult(
        change=change_result(True),
        targeted=None,
        targeted_analysis=None,
        regression=None,
        reason="verification not yet executed",
    )

    assert result.change_verified is True
    assert result.targeted_passed is False
    assert result.regression_verified is False
    assert result.verified is False


def test_pipeline_report_does_not_expose_raw_test_output():
    result = VerificationPipelineResult(
        change=change_result(True),
        targeted=execution(
            True,
            "secret-looking output",
        ),
        targeted_analysis=FailureAnalysisResult(
            passed=True,
        ),
        regression=regression_result(True),
        reason="all verification stages passed",
    )

    report = build_verification_pipeline_report(
        result
    )

    assert "secret-looking output" not in str(report)
