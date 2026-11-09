"""Tests for Phase 10.4.4 completion evidence."""

import copy
import dataclasses

import pytest

from flycoder.actions import create_action_registry
from flycoder.state import CodingState
from flycoder.tools.changes import ChangeProposal
from flycoder.tools.completion_evidence import (
    EVIDENCE_FAILED,
    EVIDENCE_INCOMPLETE,
    EVIDENCE_INCONSISTENT,
    EVIDENCE_MISSING,
    EVIDENCE_NEEDS_APPROVAL,
    EVIDENCE_STALE,
    EVIDENCE_VERIFIED,
    METHOD_STATE_TESTS,
    METHOD_TASK_OBSERVATION,
    METHOD_VERIFICATION_PIPELINE,
    READ_ONLY_INTENTS,
    TEST_VERIFIED_INTENTS,
    CompletionEvidence,
    VerificationRecord,
    build_code_signature,
    evaluate_completion_evidence,
    record_verification,
)
from flycoder.tools.filesystem import Workspace
from flycoder.tools.regression import RegressionVerificationResult
from flycoder.tools.task_progress import INTENT_MILESTONES, measure_task_progress
from flycoder.tools.testing import (
    FailureAnalysisResult,
    TestExecutionResult,
    TestFailure,
)
from flycoder.tools.verification import ChangeVerificationResult
from flycoder.tools.verification_pipeline import (
    VerificationPipelineResult,
    verify_change_pipeline,
)


ORIGINAL = (
    "def test_sample():\n"
    '    assert False, "Intentional failure for testing"\n'
)
REPAIRED = ORIGINAL.replace("assert False", "assert True")
FAILURE_OUTPUT = (
    "F\n"
    "test_sample.py:2: AssertionError\n"
    "FAILED test_sample.py::test_sample - AssertionError: Intentional\n"
    "1 failed in 0.01s"
)


# ------------------------------------------------------------------
# Builders using the real Phase 7 result structures
# ------------------------------------------------------------------


def execution(passed: bool) -> TestExecutionResult:
    return TestExecutionResult(
        command=["pytest", "-q"],
        return_code=0 if passed else 1,
        output="" if passed else FAILURE_OUTPUT,
        passed=passed,
    )


def pipeline(
    *,
    targeted_passed=True,
    regression_passed=True,
    change_verified=True,
    file="test_sample.py",
    expected=REPAIRED,
) -> VerificationPipelineResult:
    change = ChangeVerificationResult(
        verified=change_verified,
        file=file,
        reason="change check",
        expected_content=expected,
        actual_content=expected if change_verified else "other",
    )

    if not change_verified:
        return VerificationPipelineResult(
            change=change,
            targeted=None,
            targeted_analysis=None,
            regression=None,
            reason="change verification failed; tests were not executed",
        )

    failures = (
        []
        if targeted_passed
        else [
            TestFailure(
                test_file="test_sample.py",
                test_name="test_sample",
                error_type="AssertionError",
                message="Intentional",
            )
        ]
    )

    regression = None

    if targeted_passed:
        regression = RegressionVerificationResult(
            targeted=execution(True),
            regression=execution(regression_passed),
            regressions_detected=not regression_passed,
            reason="regression check",
        )

    return VerificationPipelineResult(
        change=change,
        targeted=execution(targeted_passed),
        targeted_analysis=FailureAnalysisResult(
            failures=failures,
            passed=targeted_passed,
        ),
        regression=regression,
        reason="pipeline reason",
    )


def applied_state(**overrides) -> CodingState:
    """State after an explicitly approved repair was applied."""

    values = {
        "task": "Fix the failing test",
        "task_intent": "repair",
        "files": ["test_sample.py"],
        "current_file": "test_sample.py",
        "current_file_content": REPAIRED,
        "proposed_content": REPAIRED,
        "repair_original_content": ORIGINAL,
        "repair_description": "Replace the failing assertion.",
        "repair_proposed": True,
        "repair_approved": True,
        "repair_applied": True,
    }
    values.update(overrides)
    return CodingState(**values)


def pending_state(**overrides) -> CodingState:
    """State with a repair proposal waiting for human approval."""

    values = {
        "task": "Fix the failing test",
        "task_intent": "repair",
        "files": ["test_sample.py"],
        "current_file": "test_sample.py",
        "current_file_content": ORIGINAL,
        "proposed_content": REPAIRED,
        "repair_description": "Replace the failing assertion.",
        "tests_run": True,
        "tests_passed": False,
        "last_error": FAILURE_OUTPUT,
        "error_inspected": True,
        "repair_proposed": True,
        "user_input_needed": True,
    }
    values.update(overrides)
    return CodingState(**values)


def make_test_state(**overrides) -> CodingState:
    values = {
        "task": "Run the tests",
        "task_intent": "test",
        "files": ["test_sample.py"],
    }
    values.update(overrides)
    return CodingState(**values)


def evaluate(state, verification=None, intent=None):
    return evaluate_completion_evidence(state, verification, intent)


# ------------------------------------------------------------------
# Intent coverage
# ------------------------------------------------------------------


def test_every_intent_has_explicit_criteria():
    assert READ_ONLY_INTENTS | TEST_VERIFIED_INTENTS == set(INTENT_MILESTONES)
    assert not READ_ONLY_INTENTS & TEST_VERIFIED_INTENTS


# ------------------------------------------------------------------
# Successful verification
# ------------------------------------------------------------------


def test_passing_tests_verify_test_task():
    evidence = evaluate(make_test_state(tests_run=True, tests_passed=True))

    assert evidence.status == EVIDENCE_VERIFIED
    assert evidence.verified
    assert evidence.verification_method == METHOD_STATE_TESTS
    assert evidence.tests_run and evidence.tests_passed
    assert evidence.unresolved_errors == ()
    assert evidence.criteria_satisfied
    assert evidence.human_approval_required is False


def test_repair_not_needed_when_tests_already_pass():
    state = make_test_state(
        task="Fix the failing test",
        task_intent="repair",
        tests_run=True,
        tests_passed=True,
    )

    evidence = evaluate(state)

    assert evidence.verified
    assert evidence.human_approval_required is False


def test_approved_repair_with_rerun_tests_is_verified():
    evidence = evaluate(applied_state(tests_run=True, tests_passed=True))

    assert evidence.verified
    assert evidence.human_approval_required
    assert evidence.human_approval_granted
    assert "explicitly approved and applied" in " ".join(evidence.reasons)


def test_fresh_pipeline_record_verifies():
    state = applied_state()
    record = record_verification(pipeline(), state)

    evidence = evaluate(state, record)

    assert evidence.verified
    assert evidence.verification_method == METHOD_VERIFICATION_PIPELINE
    assert evidence.tests_run and evidence.tests_passed
    assert evidence.verification is record
    assert evidence.verification_signature == evidence.code_signature


# ------------------------------------------------------------------
# Failed verification
# ------------------------------------------------------------------


def test_failed_tests_are_failed_not_missing():
    state = make_test_state(
        tests_run=True,
        tests_passed=False,
        last_error=FAILURE_OUTPUT,
    )

    evidence = evaluate(state)

    assert evidence.status == EVIDENCE_FAILED
    assert evidence.failed and not evidence.missing
    assert evidence.tests_run is True
    assert evidence.tests_passed is False
    assert "Test run on the current code failed." in evidence.unresolved_errors
    assert "1 failed in 0.01s" in evidence.unresolved_errors


def test_fresh_failed_pipeline_is_failed_with_test_names():
    state = applied_state()
    record = record_verification(pipeline(targeted_passed=False), state)

    evidence = evaluate(state, record)

    assert evidence.status == EVIDENCE_FAILED
    assert "Failing test: test_sample.py::test_sample" in (
        evidence.unresolved_errors
    )


def test_failure_wins_over_passing_source():
    state = applied_state(tests_run=True, tests_passed=True)
    record = record_verification(pipeline(regression_passed=False), state)

    evidence = evaluate(state, record)

    assert evidence.status == EVIDENCE_FAILED
    assert not evidence.verified


def test_change_verification_failure_is_failed():
    state = make_test_state()
    record = record_verification(
        pipeline(change_verified=False, file=None, expected=None),
        state,
    )

    evidence = evaluate(state, record)

    assert evidence.status == EVIDENCE_FAILED
    assert evidence.tests_run is False


# ------------------------------------------------------------------
# Missing evidence
# ------------------------------------------------------------------


def test_tests_not_run_is_missing():
    evidence = evaluate(make_test_state())

    assert evidence.status == EVIDENCE_MISSING
    assert evidence.missing and not evidence.failed
    assert evidence.tests_run is False
    assert evidence.verification_method is None


def test_applied_repair_without_rerun_is_missing():
    evidence = evaluate(applied_state())

    assert evidence.status == EVIDENCE_MISSING
    assert "re-run after the applied repair" in " ".join(evidence.reasons)


@pytest.mark.parametrize("intent", ["test", "repair"])
def test_finished_flag_is_not_evidence(intent):
    state = make_test_state(task_intent=intent, finished=True)

    evidence = evaluate(state)

    assert evidence.status == EVIDENCE_MISSING
    assert not evidence.verified


def test_finished_flag_does_not_change_evaluation():
    state = applied_state()

    unfinished = evaluate(state)
    state.finished = True

    assert evaluate(state) == unfinished


def test_full_progress_is_not_evidence():
    # Every milestone is achieved, but an unresolved error remains.
    state = make_test_state(
        tests_run=True,
        tests_passed=True,
        last_error="RuntimeError: leftover",
        finished=True,
    )

    assert measure_task_progress(state).percent == 100
    assert evaluate(state).status == EVIDENCE_FAILED


def test_full_progress_with_stale_record_only_is_not_verified():
    state = applied_state(tests_run=True, tests_passed=True)
    record = record_verification(pipeline(), state)

    state.tests_run = False
    state.tests_passed = False
    state.current_file_content = ORIGINAL

    assert evaluate(state, record).status == EVIDENCE_STALE


# ------------------------------------------------------------------
# Stale evidence
# ------------------------------------------------------------------


def test_pipeline_record_becomes_stale_after_code_change():
    state = applied_state()
    record = record_verification(pipeline(), state)

    state.current_file_content = REPAIRED + "# edited\n"
    state.proposed_content = state.current_file_content

    evidence = evaluate(state, record)

    assert evidence.status == EVIDENCE_STALE
    assert evidence.missing
    assert evidence.verification_signature != evidence.code_signature
    assert "stale" in evidence.reasons[0]


def test_record_from_before_repair_is_stale_after_apply():
    before = pending_state()
    record = record_verification(pipeline(expected=ORIGINAL), before)

    after = applied_state()

    assert evaluate(after, record).status == EVIDENCE_STALE


def test_stale_failure_is_not_upgraded_to_success():
    state = make_test_state(current_file="a.py", current_file_content="x = 1\n")
    record = record_verification(pipeline(targeted_passed=False), state)

    state.current_file_content = "x = 2\n"

    evidence = evaluate(state, record)

    assert evidence.status == EVIDENCE_STALE
    assert not evidence.verified


def test_stale_record_ignored_when_fresh_state_tests_pass():
    state = make_test_state(current_file="a.py", current_file_content="x = 1\n")
    record = record_verification(pipeline(targeted_passed=False), state)

    state.current_file_content = "x = 2\n"
    state.tests_run = True
    state.tests_passed = True

    evidence = evaluate(state, record)

    assert evidence.verified
    assert evidence.verification_method == METHOD_STATE_TESTS
    assert any("stale" in reason for reason in evidence.reasons)


def test_code_signature_tracks_content():
    state = applied_state()

    first = build_code_signature(state)

    assert build_code_signature(copy.deepcopy(state)) == first

    state.current_file_content = ORIGINAL

    assert build_code_signature(state) != first


# ------------------------------------------------------------------
# Unresolved errors
# ------------------------------------------------------------------


def test_unresolved_error_without_tests_is_failed():
    evidence = evaluate(make_test_state(last_error="ImportError: no module"))

    assert evidence.status == EVIDENCE_FAILED
    assert evidence.unresolved_errors == ("ImportError: no module",)


def test_unresolved_error_blocks_passing_tests():
    state = make_test_state(
        tests_run=True,
        tests_passed=True,
        last_error="RuntimeError: leftover",
    )

    assert evaluate(state).status == EVIDENCE_FAILED


def test_long_error_summary_is_truncated():
    evidence = evaluate(make_test_state(last_error="E" * 1000))

    assert len(evidence.unresolved_errors[0]) == 300
    assert evidence.unresolved_errors[0].endswith("...")


# ------------------------------------------------------------------
# Human approval
# ------------------------------------------------------------------


def test_pending_repair_needs_approval():
    evidence = evaluate(pending_state())

    assert evidence.status == EVIDENCE_NEEDS_APPROVAL
    assert evidence.needs_approval
    assert evidence.human_approval_required
    assert not evidence.human_approval_granted
    assert not evidence.verified


def test_passing_evidence_cannot_bypass_pending_approval():
    state = pending_state(tests_passed=True, last_error=None)
    record = record_verification(pipeline(expected=REPAIRED), state)

    evidence = evaluate(state, record)

    assert evidence.status == EVIDENCE_NEEDS_APPROVAL


def test_rejected_stale_proposal_still_needs_approval():
    # State left by approve_repair_action after a stale-proposal rejection.
    state = pending_state(repair_approved=False, repair_applied=False)

    assert evaluate(state).status == EVIDENCE_NEEDS_APPROVAL


def test_applied_without_approval_is_inconsistent():
    state = applied_state(
        repair_approved=False,
        tests_run=True,
        tests_passed=True,
    )

    evidence = evaluate(state)

    assert evidence.status == EVIDENCE_INCONSISTENT
    assert not evidence.human_approval_granted


# ------------------------------------------------------------------
# Read-only tasks
# ------------------------------------------------------------------


def test_read_only_inspection_is_verified_without_approval():
    state = CodingState(
        task="Inspect the project",
        task_intent="inspect",
        files=["a.py"],
    )

    evidence = evaluate(state)

    assert evidence.verified
    assert evidence.verification_method == METHOD_TASK_OBSERVATION
    assert evidence.human_approval_required is False
    assert evidence.tests_run is False


def test_read_only_task_not_observed_is_incomplete():
    state = CodingState(task="Inspect the project", task_intent="inspect")

    evidence = evaluate(state)

    assert evidence.status == EVIDENCE_INCOMPLETE
    assert not evidence.missing


def test_partial_review_is_incomplete():
    state = CodingState(
        task="Review the code",
        task_intent="review",
        files=["a.py", "b.py"],
        relevant_files=["a.py", "b.py"],
        files_analyzed=["a.py"],
    )

    evidence = evaluate(state)

    assert evidence.status == EVIDENCE_INCOMPLETE
    assert "1 of 2 file(s) analyzed." in " ".join(evidence.reasons)


def test_explain_task_is_not_blocked_by_the_error_it_explains():
    state = CodingState(
        task="Explain the error",
        task_intent="explain",
        files=["a.py"],
        relevant_files=["a.py"],
        files_analyzed=["a.py"],
        last_error="ValueError: bad",
    )

    evidence = evaluate(state)

    assert evidence.verified
    assert evidence.unresolved_errors == ()
    assert any("does not block" in reason for reason in evidence.reasons)


def test_read_only_task_ignores_verification_record():
    state = CodingState(
        task="Inspect the project",
        task_intent="inspect",
        files=["a.py"],
    )
    record = record_verification(pipeline(targeted_passed=False), state)

    evidence = evaluate(state, record)

    assert evidence.verified
    assert any("not required" in reason for reason in evidence.reasons)


# ------------------------------------------------------------------
# Inconsistent inputs
# ------------------------------------------------------------------


def test_tests_passed_without_run_is_inconsistent():
    evidence = evaluate(make_test_state(tests_passed=True))

    assert evidence.status == EVIDENCE_INCONSISTENT
    assert any("tests_passed" in reason for reason in evidence.reasons)


def test_record_for_other_file_is_inconsistent():
    state = applied_state()
    record = record_verification(pipeline(file="other.py"), state)

    assert evaluate(state, record).status == EVIDENCE_INCONSISTENT


def test_record_for_other_content_is_inconsistent():
    state = applied_state()
    record = record_verification(pipeline(expected=ORIGINAL), state)

    assert evaluate(state, record).status == EVIDENCE_INCONSISTENT


def test_record_with_regression_but_no_targeted_is_inconsistent():
    result = pipeline()
    result.targeted = None
    state = make_test_state()

    evidence = evaluate(state, record_verification(result, state))

    assert evidence.status == EVIDENCE_INCONSISTENT


@pytest.mark.parametrize(
    "value",
    [pipeline(), {"verified": True}, "record"],
)
def test_non_record_verification_rejected(value):
    with pytest.raises(TypeError):
        evaluate(make_test_state(), value)


def test_missing_or_unknown_intent_rejected():
    with pytest.raises(ValueError):
        evaluate(CodingState(task="x"))

    with pytest.raises(ValueError):
        evaluate(make_test_state(), intent="deploy")


def test_verification_record_validation():
    with pytest.raises(TypeError):
        VerificationRecord(result={"verified": True}, code_signature="s")

    with pytest.raises(ValueError):
        VerificationRecord(result=pipeline(), code_signature="")

    with pytest.raises(TypeError):
        record_verification({"verified": True}, make_test_state())


# ------------------------------------------------------------------
# Immutability and construction invariants
# ------------------------------------------------------------------


def valid_fields(**overrides):
    values = {
        "intent": "test",
        "status": EVIDENCE_VERIFIED,
        "verification_method": METHOD_STATE_TESTS,
        "code_signature": "sig",
        "verification_signature": None,
        "tests_run": True,
        "tests_passed": True,
        "unresolved_errors": (),
        "human_approval_required": False,
        "human_approval_granted": False,
        "criteria_satisfied": True,
        "reasons": ("ok",),
    }
    values.update(overrides)
    return values


def test_evidence_is_frozen():
    evidence = evaluate(make_test_state(tests_run=True, tests_passed=True))

    with pytest.raises(dataclasses.FrozenInstanceError):
        evidence.status = EVIDENCE_FAILED

    with pytest.raises(AttributeError):
        evidence.verified = False


def test_evidence_lists_are_copied_to_tuples():
    reasons = ["ok"]
    evidence = CompletionEvidence(**valid_fields(reasons=reasons))

    reasons.append("tampered")

    assert evidence.reasons == ("ok",)


def test_record_copies_pipeline_result():
    result = pipeline()
    state = applied_state()
    record = record_verification(result, state)

    result.reason = "tampered"
    result.change.verified = False

    assert record.result.reason == "pipeline reason"
    assert evaluate(state, record).verified


@pytest.mark.parametrize(
    "overrides",
    [
        {"criteria_satisfied": False},
        {"verification_method": None},
        {"unresolved_errors": ("boom",)},
        {"human_approval_required": True},
        {"tests_passed": False},
    ],
)
def test_contradictory_verified_evidence_rejected(overrides):
    with pytest.raises(ValueError, match="VERIFIED evidence"):
        CompletionEvidence(**valid_fields(**overrides))


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"intent": "deploy"}, ValueError),
        ({"status": "DONE"}, ValueError),
        ({"verification_method": "guess"}, ValueError),
        ({"status": EVIDENCE_MISSING, "tests_run": False}, ValueError),
        (
            {
                "status": EVIDENCE_MISSING,
                "human_approval_granted": True,
            },
            ValueError,
        ),
        ({"reasons": "ok"}, TypeError),
    ],
)
def test_invalid_evidence_rejected(overrides, error):
    with pytest.raises(error):
        CompletionEvidence(**valid_fields(**overrides))


# ------------------------------------------------------------------
# Read-only evaluation and re-evaluation
# ------------------------------------------------------------------


def test_evaluation_does_not_mutate_state():
    state = pending_state(relevant_files=["test_sample.py"])
    record = record_verification(pipeline(), state)
    before = copy.deepcopy(state)

    for intent in INTENT_MILESTONES:
        evaluate(state, record, intent)

    assert state == before


def test_reevaluation_follows_state_changes():
    state = pending_state()

    assert evaluate(state).status == EVIDENCE_NEEDS_APPROVAL

    # Mirror approve_repair_action's state changes.
    state.repair_approved = True
    state.repair_applied = True
    state.repair_original_content = ORIGINAL
    state.current_file_content = REPAIRED
    state.user_input_needed = False
    state.tests_run = False
    state.tests_passed = False
    state.last_error = None
    state.error_inspected = False

    assert evaluate(state).status == EVIDENCE_MISSING

    state.tests_run = True
    state.tests_passed = False
    state.last_error = FAILURE_OUTPUT

    assert evaluate(state).status == EVIDENCE_FAILED

    state.tests_passed = True
    state.last_error = None

    assert evaluate(state).verified


# ------------------------------------------------------------------
# Integration: real actions and the real Phase 7 pipeline
# ------------------------------------------------------------------


def test_real_repair_cycle_with_real_pipeline(tmp_path):
    (tmp_path / "test_sample.py").write_text(ORIGINAL)

    workspace = Workspace(tmp_path)
    registry = create_action_registry()
    state = CodingState(task="Fix the failing test", task_intent="repair")

    def run(action):
        result = registry.execute(action, workspace=workspace, state=state)
        assert result.action == action
        return result

    run("inspect_files")

    assert evaluate(state).status == EVIDENCE_MISSING

    run("run_tests")

    assert evaluate(state).status == EVIDENCE_FAILED

    run("fix_error")
    run("read_file")
    assert run("propose_repair").success

    assert evaluate(state).status == EVIDENCE_NEEDS_APPROVAL

    # Explicit human approval, as the CLI approval boundary would do.
    assert run("approve_repair").success

    assert evaluate(state).status == EVIDENCE_MISSING

    proposal = ChangeProposal(
        file=state.current_file,
        original_content=state.repair_original_content,
        proposed_content=state.proposed_content,
    )
    result = verify_change_pipeline(
        workspace,
        proposal,
        targeted_tests=["test_sample.py"],
    )
    record = record_verification(result, state)

    evidence = evaluate(state, record)

    assert result.verified
    assert evidence.verified
    assert evidence.verification_method == METHOD_VERIFICATION_PIPELINE
    assert evidence.human_approval_granted

    assert run("rollback_repair").success
    assert (tmp_path / "test_sample.py").read_text() == ORIGINAL

    assert evaluate(state, record).status == EVIDENCE_STALE


def test_real_pipeline_failure_is_failed(tmp_path):
    (tmp_path / "test_sample.py").write_text(ORIGINAL)

    workspace = Workspace(tmp_path)
    state = make_test_state()
    proposal = ChangeProposal(
        file="test_sample.py",
        original_content=ORIGINAL,
        proposed_content=ORIGINAL,
    )

    result = verify_change_pipeline(
        workspace,
        proposal,
        targeted_tests=["test_sample.py"],
    )
    evidence = evaluate(state, record_verification(result, state))

    assert result.targeted is not None and not result.targeted.passed
    assert evidence.status == EVIDENCE_FAILED
    assert evidence.tests_run is True
    assert evidence.tests_passed is False
