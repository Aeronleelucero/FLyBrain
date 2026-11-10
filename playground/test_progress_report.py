"""Tests for Phase 10.4.5 progress reporting."""

import contextlib
import copy
import dataclasses
import io

import pytest

from flycoder.agent import FlyCoderAgent
from flycoder.state import CodingState
from flycoder.tools import progress_report as progress_report_module
from flycoder.tools.completion_evidence import (
    EVIDENCE_FAILED,
    EVIDENCE_INCONSISTENT,
    EVIDENCE_MISSING,
    EVIDENCE_NEEDS_APPROVAL,
    EVIDENCE_STALE,
    EVIDENCE_VERIFIED,
    evaluate_completion_evidence,
    record_verification,
)
from flycoder.tools.progress import ProgressTracker
from flycoder.tools.progress_history import (
    ProgressChange,
    ProgressHistory,
    ProgressSnapshot,
)
from flycoder.tools.progress_report import (
    HistorySummary,
    LifecycleSummary,
    build_progress_report,
)
from flycoder.tools.task_lifecycle import (
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_INITIALIZING,
    TASK_NEEDS_HUMAN,
    TASK_RUNNING,
    TASK_VERIFYING,
    TaskLifecycle,
)
from flycoder.tools.task_progress import TaskProgressMetrics
from flycoder.tools.regression import RegressionVerificationResult
from flycoder.tools.testing import FailureAnalysisResult, TestExecutionResult
from flycoder.tools.verification import ChangeVerificationResult
from flycoder.tools.verification_pipeline import VerificationPipelineResult


ORIGINAL = (
    "def test_sample():\n"
    '    assert False, "Intentional failure for testing"\n'
)
REPAIRED = ORIGINAL.replace("assert False", "assert True")


def repair_state(**overrides) -> CodingState:
    values = {
        "task": "Fix the failing test",
        "task_intent": "repair",
        "files": ["test_sample.py"],
    }
    values.update(overrides)
    return CodingState(**values)


def pending_state() -> CodingState:
    return repair_state(
        current_file="test_sample.py",
        current_file_content=ORIGINAL,
        proposed_content=REPAIRED,
        repair_description="Replace the failing assertion.",
        tests_run=True,
        last_error="1 failed in 0.01s",
        error_inspected=True,
        repair_proposed=True,
        user_input_needed=True,
    )


def apply_repair(state: CodingState) -> None:
    """Mirror the state changes of approve_repair_action."""

    state.repair_original_content = state.current_file_content
    state.current_file_content = state.proposed_content
    state.repair_approved = True
    state.repair_applied = True
    state.user_input_needed = False
    state.tests_run = False
    state.tests_passed = False
    state.last_error = None
    state.error_inspected = False


def evidence_for(state, verification=None):
    return evaluate_completion_evidence(state, verification)


def history_of(*states) -> ProgressHistory:
    history = ProgressHistory()

    for iteration, state in enumerate(states, start=1):
        history.capture(iteration, state)

    return history


def started_lifecycle(task: str) -> TaskLifecycle:
    lifecycle = TaskLifecycle(task=task)
    lifecycle.transition(TASK_INITIALIZING, "init")
    lifecycle.transition(TASK_RUNNING, "run")
    return lifecycle


# ------------------------------------------------------------------
# 1–3. New task, partial progress, 100% without verification
# ------------------------------------------------------------------


def test_new_task_report():
    state = repair_state(files=[])

    report = build_progress_report(state)

    assert report.task == "Fix the failing test"
    assert report.intent == "repair"
    assert report.percent == 0
    assert report.achieved_count == 0
    assert report.applicable_count == 6
    assert report.next_milestone == "workspace_inspected"
    assert report.summary[0] == "Accomplished: no milestones achieved yet."
    assert report.verified is False
    assert report.verification_status is None


def test_partial_progress_report():
    state = CodingState(
        task="Review the code",
        task_intent="review",
        files=["a.py", "b.py"],
        relevant_files=["a.py", "b.py"],
        files_analyzed=["a.py"],
    )

    report = build_progress_report(state)

    assert report.completion_fraction == pytest.approx(2.5 / 3)
    assert report.percent == 83
    assert report.pending_milestones == ("files_analyzed",)
    assert report.summary[1] == "Remaining: files_analyzed (next: files_analyzed)."


def test_full_progress_without_verification_is_not_claimed():
    state = repair_state(
        task="Run the tests",
        task_intent="test",
        tests_run=True,
        tests_passed=True,
        last_error="RuntimeError: leftover",
    )

    report = build_progress_report(
        state,
        completion_evidence=evidence_for(state),
    )

    assert report.percent == 100
    assert report.verified is False
    assert report.verification_status == EVIDENCE_FAILED
    assert report.summary[4].startswith("Verified: no (FAILED).")
    assert any("progress is not proof" in i for i in report.inconsistencies)
    assert "Status: NOT VERIFIED (FAILED)" in report.render()


def test_full_progress_without_evidence_is_unavailable_not_failed():
    state = repair_state(tests_run=True, tests_passed=True)

    report = build_progress_report(state)

    assert report.percent == 100
    assert report.verified is False
    assert report.evidence_available is False
    assert report.verification_status is None
    assert "no completion evidence was supplied" in report.summary[4]
    assert "VERIFICATION UNAVAILABLE" in report.render()


# ------------------------------------------------------------------
# 4–7. Verified, failed, missing, stale evidence
# ------------------------------------------------------------------


def test_verified_completion_report():
    state = repair_state(tests_run=True, tests_passed=True)

    report = build_progress_report(
        state,
        completion_evidence=evidence_for(state),
    )

    assert report.verified is True
    assert report.verification_status == EVIDENCE_VERIFIED
    assert report.summary[4] == "Verified: yes, by state_tests."
    assert report.consistent
    assert "Status: VERIFIED" in report.render()


def test_failed_verification_report():
    state = repair_state(
        tests_run=True,
        tests_passed=False,
        last_error="FAILED test_sample.py::test_sample\n1 failed in 0.01s",
    )

    report = build_progress_report(
        state,
        completion_evidence=evidence_for(state),
    )

    rendered = report.render()

    assert report.verification_status == EVIDENCE_FAILED
    assert "! unresolved: 1 failed in 0.01s" in rendered
    assert "stays failed until new evidence" in report.summary[4]


def test_missing_evidence_report():
    state = repair_state()

    report = build_progress_report(
        state,
        completion_evidence=evidence_for(state),
    )

    assert report.evidence_available is True
    assert report.verification_status == EVIDENCE_MISSING
    assert report.summary[4] == (
        "Verified: no (MISSING). Tests have not been run on the current code."
    )


def passing_pipeline() -> VerificationPipelineResult:
    passing = TestExecutionResult(
        command=["pytest"],
        return_code=0,
        output="",
        passed=True,
    )

    return VerificationPipelineResult(
        change=ChangeVerificationResult(
            verified=True,
            file="test_sample.py",
            reason="ok",
            expected_content=REPAIRED,
            actual_content=REPAIRED,
        ),
        targeted=passing,
        targeted_analysis=FailureAnalysisResult(passed=True),
        regression=RegressionVerificationResult(
            targeted=passing,
            regression=passing,
            regressions_detected=False,
            reason="ok",
        ),
        reason="all passed",
    )


def test_stale_evidence_report():
    state = pending_state()
    apply_repair(state)
    record = record_verification(passing_pipeline(), state)

    state.current_file_content = REPAIRED + "# edited\n"
    state.proposed_content = state.current_file_content

    report = build_progress_report(
        state,
        completion_evidence=evidence_for(state, record),
    )

    assert report.verification_status == EVIDENCE_STALE
    assert "Verified: no (STALE)." in report.summary[4]
    assert "stale" in report.render()


# ------------------------------------------------------------------
# 8–11. History: increases, regressions, below peak, empty
# ------------------------------------------------------------------


def test_progress_increase_reported():
    first = repair_state(files=[])
    second = repair_state()

    report = build_progress_report(second, history=history_of(first, second))

    assert report.history.increase_count == 1
    assert report.history.net_progress == pytest.approx(1 / 6)
    assert report.summary[2] == "Progressing: yes; +17 points (iteration 1 → 2)."
    assert report.summary[3] == "Regressions: none recorded; net progress +17 points."


def test_progress_regression_and_below_peak_reported():
    passed = repair_state(tests_run=True, tests_passed=True)
    failed = repair_state(tests_run=True, tests_passed=False, last_error="boom")

    history = history_of(repair_state(files=[]), passed, failed)
    report = build_progress_report(failed, history=history)

    assert report.history.regression_count == 1
    assert report.history.below_peak is True
    assert report.history.peak.iteration == 2
    assert report.summary[2].startswith("Progressing: no; progress fell -")
    assert "below the peak of 100% (iteration 2)" in report.summary[3]
    assert "DECREASED (100% →" in report.render()


def test_unchanged_progress_reported():
    state = repair_state()

    report = build_progress_report(state, history=history_of(state, state))

    assert report.summary[2] == "Progressing: no change (iteration 1 → 2)."


def test_single_snapshot_history():
    state = repair_state()

    report = build_progress_report(state, history=history_of(state))

    assert report.history.snapshot_count == 1
    assert "only one snapshot (iteration 1)" in report.summary[2]


def test_empty_history_differs_from_missing_history():
    state = repair_state()

    empty = build_progress_report(state, history=ProgressHistory())
    missing = build_progress_report(state)

    assert empty.history_available and empty.history.is_empty
    assert empty.history.latest is None
    assert empty.history.peak is None
    assert "history is empty" in empty.summary[2]
    assert "history is empty" in empty.summary[3]

    assert missing.history is None
    assert "no progress history was supplied" in missing.summary[2]
    assert "no progress history was supplied" in missing.summary[3]


def test_recent_changes_are_limited():
    states = [repair_state(files=[])] + [repair_state()] * 5

    report = build_progress_report(states[-1], history=history_of(*states))

    assert report.history.snapshot_count == 6
    assert len(report.history.recent_changes) == 3
    assert report.history.recent_changes[-1].after.iteration == 6


def test_history_behind_current_state_reported():
    state = repair_state(files=[])
    history = history_of(state)
    state.files = ["test_sample.py"]

    report = build_progress_report(state, history=history)

    assert any(
        "does not match current progress" in i for i in report.inconsistencies
    )


# ------------------------------------------------------------------
# 12–13. Lifecycle
# ------------------------------------------------------------------


def test_missing_lifecycle_is_not_invented():
    report = build_progress_report(repair_state())

    assert report.lifecycle is None
    assert report.lifecycle_available is False
    assert "Lifecycle:" not in report.render()


def test_supplied_lifecycle_reported():
    lifecycle = started_lifecycle("Fix the failing test")
    lifecycle.record_iteration(3)
    lifecycle.record_action("inspect_files")
    lifecycle.record_action("run_tests")
    lifecycle.record_error("Tests failed.")

    report = build_progress_report(repair_state(), task_lifecycle=lifecycle)

    assert report.lifecycle == LifecycleSummary(
        task="Fix the failing test",
        status=TASK_RUNNING,
        is_terminal=False,
        current_iteration=3,
        action_count=2,
        error_count=1,
        latest_error="Tests failed.",
        transition_count=2,
        latest_transition_reason="run",
    )
    assert (
        "Lifecycle: RUNNING | iteration 3 | 2 action(s) | 1 error(s)"
        in report.render()
    )


def test_lifecycle_completed_without_verification_is_inconsistent():
    state = CodingState(task="Inspect", task_intent="inspect", finished=True)
    lifecycle = started_lifecycle("Inspect")
    lifecycle.transition(TASK_VERIFYING, "finished")
    lifecycle.transition(TASK_COMPLETED, "finished")

    report = build_progress_report(state, task_lifecycle=lifecycle)

    assert any("Lifecycle reports COMPLETED" in i for i in report.inconsistencies)
    assert any("marked the task finished" in i for i in report.inconsistencies)
    assert report.verified is False


def test_lifecycle_task_mismatch_reported():
    report = build_progress_report(
        repair_state(),
        task_lifecycle=started_lifecycle("Other task"),
    )

    assert any("does not match state task" in i for i in report.inconsistencies)


def test_lifecycle_needs_human_requests_intervention():
    lifecycle = started_lifecycle("Fix the failing test")
    lifecycle.transition(TASK_NEEDS_HUMAN, "stuck")

    report = build_progress_report(repair_state(), task_lifecycle=lifecycle)

    assert report.human_intervention_required
    assert "the lifecycle status is NEEDS_HUMAN" in report.human_intervention_reasons


# ------------------------------------------------------------------
# 14–15. Human approval
# ------------------------------------------------------------------


def test_human_approval_required_reported():
    state = pending_state()

    report = build_progress_report(
        state,
        completion_evidence=evidence_for(state),
    )

    assert report.verification_status == EVIDENCE_NEEDS_APPROVAL
    assert report.human_intervention_required
    assert report.human_intervention_reasons == (
        "the task state is waiting for human input",
        "milestone 'repair_applied' requires explicit human approval",
        "completion evidence requires repair approval",
    )
    assert report.summary[5].startswith("Human intervention: required — ")


def test_human_approval_granted_reported():
    state = pending_state()
    apply_repair(state)
    state.tests_run = True
    state.tests_passed = True

    report = build_progress_report(
        state,
        completion_evidence=evidence_for(state),
    )

    assert report.verified
    assert report.evidence.human_approval_required
    assert report.evidence.human_approval_granted
    assert report.human_intervention_required is False
    assert report.summary[5] == (
        "Human intervention: not indicated by the available information."
    )


def test_inconsistent_evidence_requests_review():
    state = repair_state(tests_passed=True)

    report = build_progress_report(
        state,
        completion_evidence=evidence_for(state),
    )

    assert report.verification_status == EVIDENCE_INCONSISTENT
    assert (
        "completion evidence is inconsistent and needs review"
        in report.human_intervention_reasons
    )


# ------------------------------------------------------------------
# 16–17. Read-only task, no applicable milestones
# ------------------------------------------------------------------


def test_read_only_task_report():
    state = CodingState(
        task="Inspect the project",
        task_intent="inspect",
        files=["a.py"],
    )

    report = build_progress_report(
        state,
        completion_evidence=evidence_for(state),
    )

    assert report.verified
    assert report.evidence.verification_method == "task_observation"
    assert report.evidence.human_approval_required is False
    assert report.human_intervention_required is False
    assert report.summary[0] == (
        "Accomplished: 1/1 applicable milestone(s) — workspace_inspected."
    )


def test_no_applicable_milestones(monkeypatch):
    # No real intent has zero applicable milestones; force the edge case.
    monkeypatch.setattr(
        progress_report_module,
        "measure_task_progress",
        lambda state, intent=None: TaskProgressMetrics(intent="inspect", milestones=()),
    )

    report = build_progress_report(
        CodingState(task="x", task_intent="inspect")
    )

    assert report.applicable_count == 0
    assert report.percent == 0
    assert report.next_milestone is None
    assert "no applicable milestones" in report.summary[0]
    assert report.summary[1] == (
        "Remaining: unknown; no applicable milestones exist."
    )


# ------------------------------------------------------------------
# 18. Progress and evidence disagree
# ------------------------------------------------------------------


def test_verified_evidence_with_pending_progress_reported():
    # Pipeline-verified repair: state tests were never re-run, so the
    # tests_run milestone is pending while evidence is VERIFIED.
    state = pending_state()
    apply_repair(state)
    record = record_verification(passing_pipeline(), state)

    report = build_progress_report(
        state,
        completion_evidence=evidence_for(state, record),
    )

    assert report.verified
    assert report.percent < 100
    assert any(
        "VERIFIED, but 2 milestone(s) are still pending" in i
        for i in report.inconsistencies
    )


def test_evidence_for_different_code_reported():
    state = pending_state()
    apply_repair(state)
    evidence = evidence_for(state)

    state.current_file_content = ORIGINAL

    report = build_progress_report(state, completion_evidence=evidence)

    assert report.evidence is evidence
    assert any("different code" in i for i in report.inconsistencies)


def test_evidence_intent_mismatch_reported():
    state = repair_state(tests_run=True, tests_passed=True)
    evidence = evaluate_completion_evidence(state, intent="test")

    report = build_progress_report(state, completion_evidence=evidence)

    assert any("intent 'test'" in i for i in report.inconsistencies)


def test_supplied_evidence_is_never_recalculated():
    # Evidence produced when tests failed stays FAILED in the report even
    # though the state has since moved on.
    state = repair_state(tests_run=True, tests_passed=False, last_error="x")
    evidence = evidence_for(state)

    state.tests_passed = True
    state.last_error = None

    report = build_progress_report(state, completion_evidence=evidence)

    assert report.evidence is evidence
    assert report.verification_status == EVIDENCE_FAILED
    assert any("progress is not proof" in i for i in report.inconsistencies)


# ------------------------------------------------------------------
# 19–21. Determinism, immutability, read-only
# ------------------------------------------------------------------


def full_inputs():
    state = pending_state()
    history = history_of(repair_state(files=[]), state)
    lifecycle = started_lifecycle(state.task)
    lifecycle.record_iteration(2)
    lifecycle.record_action("propose_repair")
    return state, history, evidence_for(state), lifecycle


def test_identical_inputs_produce_identical_reports():
    first = build_progress_report(*full_inputs())
    second = build_progress_report(*full_inputs())

    assert first == second
    assert first.render() == second.render()
    assert first.summary == second.summary


def test_report_is_immutable():
    report = build_progress_report(*full_inputs())

    with pytest.raises(dataclasses.FrozenInstanceError):
        report.task = "other"

    with pytest.raises(dataclasses.FrozenInstanceError):
        report.history.regression_count = 5

    with pytest.raises(dataclasses.FrozenInstanceError):
        report.lifecycle.status = TASK_COMPLETED

    assert isinstance(report.summary, tuple)
    assert isinstance(report.inconsistencies, tuple)


def test_report_unaffected_by_later_history_and_lifecycle_changes():
    state, history, evidence, lifecycle = full_inputs()
    report = build_progress_report(state, history, evidence, lifecycle)
    rendered = report.render()

    history.capture(3, repair_state(files=[]))
    lifecycle.transition(TASK_NEEDS_HUMAN, "later")
    lifecycle.record_action("another")

    assert report.history.snapshot_count == 2
    assert report.lifecycle.status == TASK_RUNNING
    assert report.render() == rendered


def test_building_report_does_not_mutate_inputs():
    state, history, evidence, lifecycle = full_inputs()

    state_before = copy.deepcopy(state)
    snapshots_before = history.snapshots
    lifecycle_before = copy.deepcopy(lifecycle)

    report = build_progress_report(state, history, evidence, lifecycle)
    report.render()

    assert state == state_before
    assert history.snapshots == snapshots_before
    assert lifecycle == lifecycle_before


# ------------------------------------------------------------------
# 22. The two ProgressSnapshot types
# ------------------------------------------------------------------


def test_iteration_level_tracker_is_rejected_as_history():
    with pytest.raises(TypeError, match="iteration-level ProgressTracker"):
        build_progress_report(repair_state(), history=ProgressTracker())


def test_report_history_uses_task_level_snapshots():
    from flycoder.tools.progress import ProgressSnapshot as IterationSnapshot

    state = repair_state()
    report = build_progress_report(
        state,
        history=history_of(repair_state(files=[]), state),
    )

    assert ProgressSnapshot is not IterationSnapshot
    assert isinstance(report.history.latest, ProgressSnapshot)
    assert not isinstance(report.history.latest, IterationSnapshot)
    assert all(
        isinstance(change, ProgressChange)
        for change in report.history.recent_changes
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"completion_evidence": {"status": "VERIFIED"}},
        {"task_lifecycle": "RUNNING"},
        {"history": [1, 2]},
    ],
)
def test_wrong_input_types_rejected(kwargs):
    with pytest.raises(TypeError):
        build_progress_report(repair_state(), **kwargs)


def test_state_type_and_intent_validated():
    with pytest.raises(TypeError):
        build_progress_report({"task": "x"})

    with pytest.raises(ValueError):
        build_progress_report(CodingState(task="x"))


def test_history_summary_rejects_negative_limit():
    with pytest.raises(ValueError):
        HistorySummary.from_history(ProgressHistory(), recent_limit=-1)


# ------------------------------------------------------------------
# Integration with the real agent
# ------------------------------------------------------------------


def test_real_agent_review_report(tmp_path):
    for name in ("alpha.py", "beta.py"):
        (tmp_path / name).write_text(f"def {name[:-3]}():\n    return 1\n")

    agent = FlyCoderAgent(tmp_path)
    state = CodingState(task="Review the code")
    lifecycle = started_lifecycle(state.task)
    history = ProgressHistory()

    with contextlib.redirect_stdout(io.StringIO()):
        for iteration in range(1, 15):
            if state.finished:
                break
            result = agent.run_once(state)
            lifecycle.record_iteration(iteration)
            lifecycle.record_action(result.action)
            history.capture(iteration, state)

    report = build_progress_report(
        state,
        history,
        evaluate_completion_evidence(state),
        lifecycle,
    )

    assert report.verified
    assert report.percent == 100
    assert report.history.regression_count == 0
    assert report.history.net_progress > 0
    assert report.lifecycle.action_count == report.history.snapshot_count
    # finished is set, but verified evidence exists, so no contradiction.
    assert report.agent_marked_finished
    assert report.consistent


def test_real_agent_repair_report_stops_at_human_boundary(tmp_path):
    (tmp_path / "test_sample.py").write_text(ORIGINAL)

    agent = FlyCoderAgent(tmp_path)
    state = CodingState(task="Fix the failing test")
    history = ProgressHistory()

    with contextlib.redirect_stdout(io.StringIO()):
        for iteration in range(1, 10):
            if state.finished or state.user_input_needed:
                break
            agent.run_once(state)
            history.capture(iteration, state)

    report = build_progress_report(
        state,
        history,
        evaluate_completion_evidence(state),
    )

    # Phase 10.6: the run prepares a proposal and stops for approval.
    assert report.verification_status == EVIDENCE_NEEDS_APPROVAL
    assert report.verified is False
    assert report.human_intervention_required
    assert "the task state is waiting for human input" in (
        report.human_intervention_reasons
    )
    # inspected, tests run, diagnosed, proposed: 4 of 6 milestones.
    assert report.percent == 67
    assert report.next_milestone == "repair_applied"


def test_verified_evidence_with_failed_lifecycle_reported():
    state = CodingState(task="Inspect", task_intent="inspect", files=["a.py"])
    lifecycle = started_lifecycle("Inspect")
    lifecycle.transition(TASK_VERIFYING, "limit")
    lifecycle.transition(TASK_FAILED, "limit")

    report = build_progress_report(
        state,
        completion_evidence=evidence_for(state),
        task_lifecycle=lifecycle,
    )

    assert report.verified
    assert (
        "Completion evidence is VERIFIED, but the lifecycle ended as FAILED."
        in report.inconsistencies
    )
