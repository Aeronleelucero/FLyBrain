"""Verification actions for FLY-CODER Phase 10.5.

``verify_changes`` runs the existing Phase 7 verification pipeline for
the code change the agent applied (an explicitly approved repair):

    change check (disk content matches the applied proposal)
        -> targeted tests (Phase 7 test selection)
        -> full regression suite

It reuses Phase 7 test discovery, selection, and execution; it is not a
second test runner. It never modifies workspace files, never approves
or applies a repair, and a failed verification does not trigger one.
"""

from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.changes import ChangeProposal
from flycoder.tools.completion_evidence import record_verification
from flycoder.tools.filesystem import Workspace
from flycoder.tools.freshness import record_test_run
from flycoder.tools.testing import discover_tests, select_tests
from flycoder.tools.verification_pipeline import (
    VerificationPipelineResult,
    build_verification_pipeline_report,
    verify_change_pipeline,
)


VERIFICATION_TIMEOUT_SECONDS = 60


def _failure_output(result: VerificationPipelineResult) -> str:
    """Return the most specific failure output the pipeline produced."""

    parts = [f"Verification failed: {result.reason}"]

    if result.targeted is not None and not result.targeted.passed:
        parts.append(result.targeted.output)
    elif result.regression is not None and not result.regression.verified:
        parts.append(result.regression.regression.output)

    return "\n".join(part for part in parts if part)


def verify_changes_action(
    workspace: Workspace,
    state: CodingState,
) -> ActionResult:
    """Verify the applied repair with the Phase 7 verification pipeline."""

    if not state.repair_applied:
        return ActionResult(
            action="verify_changes",
            success=False,
            message="No applied code change is available to verify.",
        )

    if (
        not state.current_file
        or state.repair_original_content is None
        or state.proposed_content is None
    ):
        return ActionResult(
            action="verify_changes",
            success=False,
            message=(
                "The applied change is missing its file, original content, "
                "or proposed content and cannot be verified."
            ),
        )

    proposal = ChangeProposal(
        file=state.current_file,
        original_content=state.repair_original_content,
        proposed_content=state.proposed_content,
        description=state.repair_description or "",
    )

    selection = select_tests(
        discover_tests(workspace.root),
        changed_files=[state.current_file],
        fallback_to_all=False,
    )

    # No related tests: targeted verification falls back to the full
    # suite inside the pipeline.
    targeted_tests = selection.selected or None

    result = verify_change_pipeline(
        workspace,
        proposal,
        targeted_tests=targeted_tests,
        timeout=VERIFICATION_TIMEOUT_SECONDS,
    )

    record = record_verification(result, state)
    state.verification_record = record

    if result.targeted is not None:
        # The pipeline ran tests on the current code. The full
        # regression result (or the failed targeted run) is the test
        # outcome for this code version.
        record_test_run(
            state,
            passed=result.verified,
            output=None if result.verified else _failure_output(result),
        )
    else:
        # The change check failed before any test ran: no test result is
        # recorded, but the failure must be visible to the agent.
        state.last_error = _failure_output(result)
        state.error_inspected = False

    return ActionResult(
        action="verify_changes",
        success=result.verified,
        message=(
            "Verification passed: change, targeted tests, and regression "
            "suite."
            if result.verified
            else f"Verification failed: {result.reason}."
        ),
        data={
            "file": state.current_file,
            "targeted_tests": selection.selected,
            "test_selection_fallback": targeted_tests is None,
            "verification": build_verification_pipeline_report(result),
            "verification_signature": record.code_signature,
        },
    )
