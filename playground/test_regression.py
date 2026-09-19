"""Tests for Phase 7.6 regression verification."""

from pathlib import Path

from flycoder.tools.regression import (
    RegressionVerificationResult,
    build_regression_verification_report,
)
from flycoder.tools.testing import TestExecutionResult


def execution(
    *,
    passed: bool,
    output: str = "",
    return_code: int | None = None,
    timed_out: bool = False,
) -> TestExecutionResult:
    """Build a deterministic test execution result."""
    if return_code is None:
        return_code = 0 if passed else 1

    return TestExecutionResult(
        command=["pytest", "-q"],
        return_code=return_code,
        output=output,
        passed=passed,
        timed_out=timed_out,
        duration_seconds=0.1,
    )


def test_verified_when_targeted_and_regression_pass():
    targeted = execution(
        passed=True,
        output="5 passed",
    )

    regression = execution(
        passed=True,
        output="463 passed",
    )

    result = RegressionVerificationResult(
        targeted=targeted,
        regression=regression,
        regressions_detected=False,
        reason=(
            "targeted tests and the full regression suite passed"
        ),
    )

    assert result.verified is True
    assert result.regressions_detected is False


def test_targeted_failure_is_not_reported_as_regression():
    targeted = execution(
        passed=False,
        output=(
            "FAILED test_example.py::test_example "
            "- AssertionError: failed"
        ),
    )

    regression = execution(
        passed=True,
        output="463 passed",
    )

    result = RegressionVerificationResult(
        targeted=targeted,
        regression=regression,
        regressions_detected=False,
        reason=(
            "targeted tests failed; regression verification "
            "cannot establish a successful change"
        ),
    )

    assert result.verified is False
    assert result.regressions_detected is False


def test_regression_failure_is_detected():
    targeted = execution(
        passed=True,
        output="5 passed",
    )

    regression = execution(
        passed=False,
        output=(
            "FAILED tests/test_other.py::test_other "
            "- AssertionError: regression"
        ),
    )

    result = RegressionVerificationResult(
        targeted=targeted,
        regression=regression,
        regressions_detected=True,
        reason=(
            "the full regression suite failed with "
            "1 analyzed failure(s)"
        ),
    )

    assert result.verified is False
    assert result.regressions_detected is True


def test_regression_timeout_is_detected():
    targeted = execution(
        passed=True,
        output="5 passed",
    )

    regression = execution(
        passed=False,
        timed_out=True,
        output="pytest timed out after 60 seconds",
        return_code=-1,
    )

    result = RegressionVerificationResult(
        targeted=targeted,
        regression=regression,
        regressions_detected=True,
        reason=(
            "the full regression suite timed out"
        ),
    )

    assert result.verified is False
    assert result.regressions_detected is True


def test_report_is_serializable():
    targeted = execution(
        passed=True,
        output="5 passed",
    )

    regression = execution(
        passed=True,
        output="463 passed",
    )

    result = RegressionVerificationResult(
        targeted=targeted,
        regression=regression,
        regressions_detected=False,
        reason=(
            "targeted tests and the full regression suite passed"
        ),
    )

    report = build_regression_verification_report(
        result
    )

    assert report["verified"] is True
    assert report["regressions_detected"] is False
    assert report["targeted"]["passed"] is True
    assert report["regression"]["passed"] is True


def test_report_preserves_failure_state():
    targeted = execution(
        passed=True,
        output="5 passed",
    )

    regression = execution(
        passed=False,
        output="FAILED test.py::test_x - AssertionError: x",
    )

    result = RegressionVerificationResult(
        targeted=targeted,
        regression=regression,
        regressions_detected=True,
        reason="the full regression suite failed",
    )

    report = build_regression_verification_report(
        result
    )

    assert report["verified"] is False
    assert report["regressions_detected"] is True
    assert report["regression"]["passed"] is False
