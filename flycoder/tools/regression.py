"""Regression verification tools for FLY-CODER."""

from dataclasses import dataclass

from flycoder.tools.testing import (
    TestExecutionResult,
    analyze_test_failures,
    build_test_execution_report,
    execute_tests,
)


@dataclass
class RegressionVerificationResult:
    """Represent the result of regression verification."""

    __test__ = False

    targeted: TestExecutionResult
    regression: TestExecutionResult
    regressions_detected: bool
    reason: str

    @property
    def verified(self) -> bool:
        """Return whether regression verification passed."""
        return (
            self.targeted.passed
            and self.regression.passed
            and not self.regressions_detected
        )


def verify_regression(
    workspace: str,
    targeted_tests: list[str] | None = None,
    timeout: int = 60,
) -> RegressionVerificationResult:
    """
    Run targeted tests followed by the complete regression suite.

    This function executes tests only. It does not modify source files.
    """
    targeted = execute_tests(
        workspace,
        test_files=targeted_tests,
        timeout=timeout,
    )

    regression = execute_tests(
        workspace,
        test_files=None,
        timeout=timeout,
    )

    targeted_analysis = analyze_test_failures(
        targeted
    )

    regression_analysis = analyze_test_failures(
        regression
    )

    if not targeted.passed:
        reason = (
            "targeted tests failed; regression verification "
            "cannot establish a successful change"
        )

        return RegressionVerificationResult(
            targeted=targeted,
            regression=regression,
            regressions_detected=False,
            reason=reason,
        )

    if regression.passed:
        reason = (
            "targeted tests and the full regression suite passed"
        )

        return RegressionVerificationResult(
            targeted=targeted,
            regression=regression,
            regressions_detected=False,
            reason=reason,
        )

    regression_count = regression_analysis.failure_count

    if regression_analysis.timed_out:
        reason = (
            "the full regression suite timed out"
        )
    elif regression_analysis.collection_error:
        reason = (
            "the full regression suite encountered "
            "a collection error"
        )
    elif regression_count:
        reason = (
            f"the full regression suite failed with "
            f"{regression_count} analyzed failure(s)"
        )
    else:
        reason = (
            "the full regression suite failed"
        )

    return RegressionVerificationResult(
        targeted=targeted,
        regression=regression,
        regressions_detected=True,
        reason=reason,
    )


def build_regression_verification_report(
    result: RegressionVerificationResult,
) -> dict:
    """Build a serializable regression-verification report."""
    return {
        "verified": result.verified,
        "regressions_detected": result.regressions_detected,
        "reason": result.reason,
        "targeted": build_test_execution_report(
            result.targeted
        ),
        "regression": build_test_execution_report(
            result.regression
        ),
    }
