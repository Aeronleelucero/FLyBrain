"""Tests for the Phase 10.4.2 task-level progress metrics."""

import contextlib
import copy
import io

import pytest

from flycoder.actions import create_action_registry
from flycoder.agent import FlyCoderAgent
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace
from flycoder.tools.iteration import observe_iteration
from flycoder.actions.registry import ActionResult
from flycoder.tools.proposals import record_human_approval
from flycoder.tools.task_progress import (
    FILES_ANALYZED,
    INTENT_MILESTONES,
    MILESTONE_ACHIEVED,
    MILESTONE_NOT_APPLICABLE,
    MILESTONE_PENDING,
    MilestoneStatus,
    REPAIR_APPLIED,
    TESTS_RUN,
    TaskProgress,
    TaskProgressMetrics,
    measure_task_progress,
)


def milestone(key, status, fraction=None, requires_human=False):
    if fraction is None:
        fraction = 1.0 if status == MILESTONE_ACHIEVED else 0.0

    return MilestoneStatus(
        key=key,
        description=key,
        status=status,
        fraction=fraction,
        detail=f"{key} detail",
        requires_human=requires_human,
    )


def repair_state(**overrides) -> CodingState:
    values = {
        "task": "Fix the failing test",
        "task_intent": "repair",
        "files": ["test_a.py"],
    }
    values.update(overrides)
    return CodingState(**values)


# ------------------------------------------------------------------
# Fraction arithmetic
# ------------------------------------------------------------------


def test_empty_metrics_are_zero():
    metrics = TaskProgressMetrics(intent="inspect", milestones=())

    assert metrics.completion_fraction == 0.0
    assert metrics.percent == 0
    assert metrics.all_milestones_achieved is False
    assert metrics.next_milestone is None
    assert metrics.awaiting_human is False


def test_not_applicable_excluded_from_denominator():
    metrics = TaskProgressMetrics(
        intent="repair",
        milestones=(
            milestone("a", MILESTONE_ACHIEVED),
            milestone("b", MILESTONE_NOT_APPLICABLE),
            milestone("c", MILESTONE_NOT_APPLICABLE),
            milestone("d", MILESTONE_PENDING),
        ),
    )

    assert metrics.applicable_count == 2
    assert metrics.not_applicable_count == 2
    assert metrics.achieved_count == 1
    assert metrics.pending_count == 1
    assert metrics.completion_fraction == 0.5


def test_partial_milestones_contribute_fraction():
    metrics = TaskProgressMetrics(
        intent="review",
        milestones=(
            milestone("a", MILESTONE_ACHIEVED),
            milestone("b", MILESTONE_ACHIEVED),
            milestone("c", MILESTONE_PENDING, fraction=0.25),
        ),
    )

    assert metrics.completion_fraction == pytest.approx(2.25 / 3)
    assert metrics.percent == 75


def test_percent_never_rounds_up_to_100_when_unfinished():
    metrics = TaskProgressMetrics(
        intent="review",
        milestones=tuple(
            milestone(str(i), MILESTONE_ACHIEVED) for i in range(199)
        )
        + (milestone("last", MILESTONE_PENDING, fraction=0.5),),
    )

    assert metrics.completion_fraction > 0.995
    assert metrics.percent == 99
    assert metrics.all_milestones_achieved is False


def test_next_milestone_is_first_pending_in_order():
    metrics = TaskProgressMetrics(
        intent="repair",
        milestones=(
            milestone("a", MILESTONE_ACHIEVED),
            milestone("b", MILESTONE_NOT_APPLICABLE),
            milestone("c", MILESTONE_PENDING),
            milestone("d", MILESTONE_PENDING),
        ),
    )

    assert metrics.next_milestone.key == "c"


def test_explain_lists_every_milestone():
    state = repair_state(tests_run=True, tests_passed=True)

    lines = measure_task_progress(state).explain()

    assert lines[0] == (
        "Task progress (repair): 100% — 3/3 applicable milestone(s) "
        "achieved."
    )
    assert len(lines) == 1 + len(INTENT_MILESTONES["repair"])
    assert any("[NOT_APPLICABLE] repair_applied" in line for line in lines)


# ------------------------------------------------------------------
# State-derived metrics
# ------------------------------------------------------------------


def test_repair_not_needed_reaches_full_progress():
    metrics = measure_task_progress(
        repair_state(tests_run=True, tests_passed=True)
    )

    assert metrics.applicable_count == 3
    assert metrics.completion_fraction == 1.0
    assert metrics.all_milestones_achieved is True


def test_review_partial_progress():
    state = CodingState(
        task="Review the code",
        task_intent="review",
        files=["a.py", "b.py", "c.py", "d.py"],
        relevant_files=["a.py", "b.py", "c.py", "d.py"],
        files_analyzed=["a.py", "b.py"],
    )

    metrics = measure_task_progress(state)

    assert metrics.completion_fraction == pytest.approx(2.5 / 3)
    assert metrics.next_milestone.key == FILES_ANALYZED


def test_progress_decreases_when_tests_regress():
    state = CodingState(
        task="Run the tests",
        task_intent="test",
        files=["test_a.py"],
        tests_run=True,
        tests_passed=True,
    )

    before = measure_task_progress(state).completion_fraction

    state.tests_passed = False

    after = measure_task_progress(state).completion_fraction

    assert before == 1.0
    assert after == pytest.approx(2 / 3)


def test_progress_decreases_after_approved_repair_invalidates_tests():
    state = repair_state(
        tests_run=True,
        tests_passed=True,
        error_inspected=True,
        repair_proposed=True,
        repair_approved=True,
        repair_applied=True,
    )

    before = measure_task_progress(state)

    state.tests_run = False
    state.tests_passed = False

    after = measure_task_progress(state)

    assert before.all_milestones_achieved
    assert after.completion_fraction == pytest.approx(4 / 6)
    assert after.next_milestone.key == TESTS_RUN


# ------------------------------------------------------------------
# Approval boundary
# ------------------------------------------------------------------


def test_awaiting_human_at_repair_approval():
    state = repair_state(
        tests_run=True,
        last_error="boom",
        error_inspected=True,
        repair_proposed=True,
        user_input_needed=True,
    )

    metrics = measure_task_progress(state)

    assert metrics.awaiting_human is True
    assert metrics.next_milestone.key == REPAIR_APPLIED
    assert metrics.completion_fraction == pytest.approx(4 / 6)


@pytest.mark.parametrize(
    "approved, applied",
    [(True, False), (False, True)],
)
def test_repair_milestone_requires_approval_and_application(
    approved,
    applied,
):
    state = repair_state(
        tests_run=True,
        tests_passed=True,
        error_inspected=True,
        repair_proposed=True,
        repair_approved=approved,
        repair_applied=applied,
    )

    metrics = measure_task_progress(state)

    assert metrics.all_milestones_achieved is False
    assert metrics.percent < 100
    assert metrics.next_milestone.key == REPAIR_APPLIED


# ------------------------------------------------------------------
# Completion is never inferred
# ------------------------------------------------------------------


def test_finished_flag_does_not_affect_metrics():
    state = repair_state(tests_run=True, tests_passed=False)

    unfinished = measure_task_progress(state)

    state.finished = True

    finished = measure_task_progress(state)

    assert finished == unfinished
    assert finished.all_milestones_achieved is False


def test_metrics_expose_no_completion_verdict():
    metrics = measure_task_progress(
        repair_state(tests_run=True, tests_passed=True)
    )

    for name in ("completed", "complete", "verified", "is_complete"):
        assert not hasattr(metrics, name)


def test_metrics_are_read_only():
    state = repair_state(
        tests_run=True,
        last_error="boom",
        error_inspected=True,
        repair_proposed=True,
        user_input_needed=True,
    )

    before = copy.deepcopy(state)

    for intent in INTENT_MILESTONES:
        metrics = measure_task_progress(state, intent)
        metrics.explain()

    assert state == before

    with pytest.raises(AttributeError):
        metrics.intent = "other"


def test_missing_intent_rejected():
    with pytest.raises(ValueError):
        measure_task_progress(CodingState(task="x"))


# ------------------------------------------------------------------
# Separation from iteration activity statistics
# ------------------------------------------------------------------


def test_successful_actions_are_not_task_progress():
    state = CodingState(task="Run the tests", task_intent="test")
    activity = TaskProgress()

    for iteration in range(1, 4):
        activity.record(
            observe_iteration(
                iteration,
                ActionResult(action="read_file", success=True, message=""),
                state,
                previous_signature="different",
            )
        )

    assert activity.successful_iterations == 3
    assert activity.progress_rate == 1.0
    assert measure_task_progress(state).completion_fraction == 0.0


# ------------------------------------------------------------------
# Integration with real actions and the real agent
# ------------------------------------------------------------------


FAILING_TEST = (
    "def test_sample():\n"
    '    assert False, "Intentional failure for testing"\n'
)


def test_real_repair_cycle_progress_and_rollback_regression(tmp_path):
    (tmp_path / "test_sample.py").write_text(FAILING_TEST)

    workspace = Workspace(tmp_path)
    registry = create_action_registry()
    state = repair_state(files=[])

    def run(action):
        result = registry.execute(action, workspace=workspace, state=state)
        return result, measure_task_progress(state)

    fractions = []

    for action in ("inspect_files", "run_tests", "fix_error", "read_file"):
        _, metrics = run(action)
        fractions.append(metrics.completion_fraction)

    assert state.current_file == "test_sample.py"
    assert fractions == sorted(fractions)

    result, metrics = run("propose_repair")

    assert result.success
    assert metrics.awaiting_human is True
    assert metrics.completion_fraction == pytest.approx(4 / 6)

    # Explicit human approval of the reviewed proposal, then apply.
    record_human_approval(state, state.proposal_fingerprint)
    result, metrics = run("approve_repair")

    assert result.success
    assert metrics.milestones[4].status == MILESTONE_ACHIEVED
    assert metrics.next_milestone.key == TESTS_RUN

    result, metrics = run("run_tests")

    assert result.success
    assert metrics.all_milestones_achieved

    result, metrics = run("rollback_repair")

    assert result.success
    assert (tmp_path / "test_sample.py").read_text() == FAILING_TEST
    assert metrics.all_milestones_achieved is False
    assert metrics.completion_fraction < 0.5


def test_real_agent_review_progress_increases_per_file(tmp_path):
    for name in ("alpha.py", "beta.py", "gamma.py"):
        (tmp_path / name).write_text(f"def {name[:-3]}():\n    return 1\n")

    agent = FlyCoderAgent(tmp_path)
    state = CodingState(task="Review the code")

    fractions = []

    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(20):
            if state.finished:
                break
            agent.run_once(state)
            fractions.append(measure_task_progress(state).completion_fraction)

    assert state.finished
    assert fractions == sorted(fractions)
    assert len(set(fractions)) >= 4
    assert measure_task_progress(state).all_milestones_achieved
