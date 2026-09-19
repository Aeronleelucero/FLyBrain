"""Integrated verification pipeline for FLY-CODER."""

from dataclasses import dataclass

from flycoder.tools.changes import ChangeProposal
from flycoder.tools.filesystem import Workspace
from flycoder.tools.regression import (
    RegressionVerificationResult,
    verify_regression,
)
from flycoder.tools.testing import (
    FailureAnalysisResult,
    TestExecutionResult,
    analyze_test_failures,
    execute_tests,
)
from flycoder.tools.verification import (
    ChangeVerificationResult,
    verify_change,
)


@dataclass
class VerificationPipelineResult:
    """Represent the complete verification state of a change."""

    __test__ = False

    change: ChangeVerificationResult
    targeted: TestExecutionResult | None
    targeted_analysis: FailureAnalysisResult | None
    regression: RegressionVerificationResult | None
    reason: str

    @property
    def change_verified(self) -> bool:
        """Return whether the applied change matches the proposal."""
        return self.change.verified

    @property
    def targeted_passed(self) -> bool:
        """Return whether targeted tests passed."""
        return (
            self.targeted is not None
            and self.targeted.passed
        )

    @property
    def regression_verified(self) -> bool:
        """Return whether regression verification passed."""
        return (
            self.regression is not None
            and self.regression.verified
        )

    @property
    def verified(self) -> bool:
        """Return whether the entire verification pipeline passed."""
        return (
            self.change_verified
            and self.targeted_passed
            and self.regression_verified
        )

    @property
    def failed(self) -> bool:
        """Return whether verification failed."""
        return not self.verified


def verify_change_pipeline(
    workspace: Workspace,
    proposal: ChangeProposal,
    targeted_tests: list[str] | None = None,
    timeout: int = 60,
) -> VerificationPipelineResult:
    """
    Verify an applied change through the complete Phase 7 pipeline.

    The pipeline:

        change verification
            ↓
        targeted tests
            ↓
        failure analysis
            ↓
        full regression verification

    The pipeline never modifies source files.
    """
    change_result = verify_change(
        workspace,
        proposal,
    )

    if not change_result.verified:
        return VerificationPipelineResult(
            change=change_result,
            targeted=None,
            targeted_analysis=None,
            regression=None,
            reason=(
                "change verification failed; "
                "tests were not executed"
            ),
        )

    targeted = execute_tests(
        workspace.root,
        test_files=targeted_tests,
        timeout=timeout,
    )

    targeted_analysis = analyze_test_failures(
        targeted
    )

    if not targeted.passed:
        return VerificationPipelineResult(
            change=change_result,
            targeted=targeted,
            targeted_analysis=targeted_analysis,
            regression=None,
            reason=(
                "change verification passed, but "
                "targeted tests failed"
            ),
        )

    regression = verify_regression(
        workspace.root,
        targeted_tests=targeted_tests,
        timeout=timeout,
    )

    if not regression.verified:
        return VerificationPipelineResult(
            change=change_result,
            targeted=targeted,
            targeted_analysis=targeted_analysis,
            regression=regression,
            reason=(
                "change and targeted tests passed, "
                "but regression verification failed"
            ),
        )

    return VerificationPipelineResult(
        change=change_result,
        targeted=targeted,
        targeted_analysis=targeted_analysis,
        regression=regression,
        reason=(
            "change verification, targeted tests, "
            "and regression verification all passed"
        ),
    )


def build_verification_pipeline_report(
    result: VerificationPipelineResult,
) -> dict:
    """Build a serializable verification pipeline report."""
    report = {
        "verified": result.verified,
        "failed": result.failed,
        "change_verified": result.change_verified,
        "targeted_passed": result.targeted_passed,
        "regression_verified": result.regression_verified,
        "reason": result.reason,
    }

    report["change"] = {
        "verified": result.change.verified,
        "file": result.change.file,
        "reason": result.change.reason,
        "changed": result.change.changed,
    }

    if result.targeted is None:
        report["targeted"] = None
    else:
        report["targeted"] = {
            "passed": result.targeted.passed,
            "failed": result.targeted.failed,
            "return_code": result.targeted.return_code,
            "timed_out": result.targeted.timed_out,
            "duration_seconds": (
                result.targeted.duration_seconds
            ),
        }

    if result.targeted_analysis is None:
        report["targeted_analysis"] = None
    else:
        report["targeted_analysis"] = {
            "passed": result.targeted_analysis.passed,
            "failed": result.targeted_analysis.failed,
            "failure_count": (
                result.targeted_analysis.failure_count
            ),
            "collection_error": (
                result.targeted_analysis.collection_error
            ),
            "timed_out": (
                result.targeted_analysis.timed_out
            ),
        }

    if result.regression is None:
        report["regression"] = None
    else:
        report["regression"] = {
            "verified": result.regression.verified,
            "regressions_detected": (
                result.regression.regressions_detected
            ),
            "reason": result.regression.reason,
            "targeted_passed": (
                result.regression.targeted.passed
            ),
            "regression_passed": (
                result.regression.regression.passed
            ),
        }

    return report
