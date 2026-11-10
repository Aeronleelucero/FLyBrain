"""Tests for Phase 10.4.3 task progress history."""

import copy
import dataclasses

import pytest

from flycoder.state import CodingState
from flycoder.tools.iteration import build_state_signature
from flycoder.tools.progress_history import (
    PROGRESS_DECREASED,
    PROGRESS_INCREASED,
    PROGRESS_UNCHANGED,
    ProgressChange,
    ProgressHistory,
    ProgressSnapshot,
    capture_snapshot,
)
from flycoder.tools.task_progress import (
    REPAIR_APPLIED,
    TESTS_PASSED,
    TESTS_RUN,
    WORKSPACE_INSPECTED,
    measure_task_progress,
)


APPLICABLE = ("a", "b", "c", "d")


def snap(iteration, achieved=0, **overrides):
    """Build a consistent snapshot with ``achieved`` of four milestones."""

    keys = APPLICABLE[:achieved]
    values = {
        "iteration": iteration,
        "completion_fraction": achieved / 4,
        "percent": round(achieved / 4 * 100) if achieved < 4 else 100,
        "achieved_milestones": keys,
        "applicable_milestones": APPLICABLE,
        "next_milestone": APPLICABLE[achieved] if achieved < 4 else None,
        "state_signature": f"sig-{iteration}",
    }
    values.update(overrides)
    return ProgressSnapshot(**values)


def history_of(*achieved_values):
    history = ProgressHistory()

    for iteration, achieved in enumerate(achieved_values, start=1):
        history.record(snap(iteration, achieved))

    return history


# ------------------------------------------------------------------
# Empty history
# ------------------------------------------------------------------


def test_empty_history_is_safe():
    history = ProgressHistory()

    assert history.is_empty
    assert len(history) == 0
    assert list(history) == []
    assert history.snapshots == ()
    assert history.first is None
    assert history.latest is None
    assert history.changes == ()
    assert history.regression_count == 0
    assert history.increase_count == 0
    assert history.net_progress == 0.0
    assert history.peak is None
    assert history.below_peak is False


def test_single_snapshot_history():
    history = history_of(2)

    assert history.first is history.latest
    assert history.changes == ()
    assert history.net_progress == 0.0
    assert history.peak is history.latest
    assert history.below_peak is False


# ------------------------------------------------------------------
# Chronological order
# ------------------------------------------------------------------


def test_snapshots_preserved_in_chronological_order():
    history = ProgressHistory()

    for iteration in (0, 1, 3, 10):
        history.record(snap(iteration, 1))

    assert [s.iteration for s in history] == [0, 1, 3, 10]
    assert history.first.iteration == 0
    assert history.latest.iteration == 10


@pytest.mark.parametrize("iteration", [1, 2])
def test_out_of_order_or_duplicate_iteration_rejected(iteration):
    history = ProgressHistory()
    history.record(snap(2, 1))

    with pytest.raises(ValueError, match="increasing iteration order"):
        history.record(snap(iteration, 2))

    assert len(history) == 1


@pytest.mark.parametrize("value", [None, "snapshot", {"iteration": 1}])
def test_non_snapshot_rejected(value):
    with pytest.raises(TypeError):
        ProgressHistory().record(value)


def test_iteration_progress_snapshot_is_not_accepted():
    # The iteration-level ProgressSnapshot is a different model.
    from flycoder.tools.progress import ProgressSnapshot as IterationSnapshot

    other = IterationSnapshot(
        iteration=1,
        progress=True,
        progress_count=1,
        no_progress_count=0,
        state_signature="s",
        action="read_file",
    )

    with pytest.raises(TypeError):
        ProgressHistory().record(other)


# ------------------------------------------------------------------
# Changes, increases, decreases
# ------------------------------------------------------------------


def test_progress_increases_detected():
    history = history_of(0, 1, 3)

    assert [c.direction for c in history.changes] == [
        PROGRESS_INCREASED,
        PROGRESS_INCREASED,
    ]
    assert history.increase_count == 2
    assert history.regression_count == 0
    assert history.changes[1].delta == pytest.approx(0.5)
    assert history.changes[1].gained_milestones == ("b", "c")
    assert history.changes[1].lost_milestones == ()


def test_progress_decreases_detected():
    history = history_of(3, 1)

    change = history.changes[0]

    assert change.direction == PROGRESS_DECREASED
    assert change.decreased
    assert change.delta == pytest.approx(-0.5)
    assert change.lost_milestones == ("b", "c")
    assert change.gained_milestones == ()


def test_repeated_values_are_unchanged():
    history = history_of(2, 2, 2)

    assert [c.direction for c in history.changes] == [
        PROGRESS_UNCHANGED,
        PROGRESS_UNCHANGED,
    ]
    assert history.increase_count == 0
    assert history.regression_count == 0


def test_float_noise_is_unchanged():
    before = snap(1, 1, completion_fraction=1 / 3, percent=33)
    after = snap(2, 1, completion_fraction=(0.1 + 0.2) / 0.9, percent=33)

    assert before.completion_fraction != after.completion_fraction
    assert ProgressChange(before, after).direction == PROGRESS_UNCHANGED


def test_regression_counting():
    history = history_of(1, 3, 2, 2, 4, 0, 1)

    assert history.regression_count == 2
    assert history.increase_count == 3
    assert [
        (c.before.iteration, c.after.iteration) for c in history.regressions
    ] == [(2, 3), (5, 6)]


def test_net_progress_between_first_and_latest():
    assert history_of(1, 4, 2).net_progress == pytest.approx(0.25)
    assert history_of(3, 4, 1).net_progress == pytest.approx(-0.5)
    assert history_of(2, 0, 2).net_progress == 0.0


def test_regression_preserves_earlier_snapshots():
    history = history_of(1, 4)
    best = history.latest

    history.record(snap(3, 1))

    assert history.snapshots[1] is best
    assert history.snapshots[1].all_milestones_achieved
    assert history.peak is best
    assert history.below_peak is True


def test_peak_is_earliest_maximum():
    history = history_of(1, 3, 3, 2)

    assert history.peak.iteration == 2


# ------------------------------------------------------------------
# Immutability
# ------------------------------------------------------------------


def test_snapshot_is_frozen():
    snapshot = snap(1, 1)

    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.completion_fraction = 1.0


def test_snapshot_lists_are_copied_to_tuples():
    achieved = ["a"]
    applicable = ["a", "b"]

    snapshot = ProgressSnapshot(
        iteration=1,
        completion_fraction=0.5,
        percent=50,
        achieved_milestones=achieved,
        applicable_milestones=applicable,
        next_milestone="b",
        state_signature="s",
    )

    achieved.append("b")
    applicable.clear()

    assert snapshot.achieved_milestones == ("a",)
    assert snapshot.applicable_milestones == ("a", "b")


def test_history_snapshots_cannot_be_modified_externally():
    history = history_of(1, 2)

    snapshots = history.snapshots

    assert isinstance(snapshots, tuple)

    with pytest.raises(AttributeError):
        history.snapshots = ()

    assert len(history) == 2


def test_change_is_frozen():
    change = history_of(1, 2).changes[0]

    with pytest.raises(dataclasses.FrozenInstanceError):
        change.after = change.before


# ------------------------------------------------------------------
# Snapshot validation
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"iteration": -1}, ValueError),
        ({"iteration": 1.0}, ValueError),
        ({"iteration": True}, ValueError),
        ({"completion_fraction": -0.1}, ValueError),
        ({"completion_fraction": 1.5}, ValueError),
        ({"completion_fraction": float("nan")}, ValueError),
        ({"completion_fraction": "0.5"}, ValueError),
        ({"percent": 101}, ValueError),
        ({"percent": 25.0}, ValueError),
        ({"achieved_milestones": "a"}, TypeError),
        ({"achieved_milestones": ("a", "a")}, ValueError),
        ({"achieved_milestones": ("a", "")}, ValueError),
        ({"achieved_milestones": ("z",)}, ValueError),
        ({"next_milestone": "z"}, ValueError),
        ({"next_milestone": "a"}, ValueError),
        ({"percent": 100}, ValueError),
        ({"state_signature": None}, TypeError),
    ],
)
def test_invalid_snapshot_rejected(overrides, error):
    values = dataclasses.asdict(snap(1, 1))
    values.update(overrides)

    with pytest.raises(error):
        ProgressSnapshot(**values)


def test_full_progress_requires_percent_100():
    with pytest.raises(ValueError, match="percent is 100"):
        snap(1, 4, percent=99)


def test_from_metrics_rejects_wrong_type():
    with pytest.raises(TypeError):
        ProgressSnapshot.from_metrics(1, {"completion_fraction": 1.0}, "s")


# ------------------------------------------------------------------
# Capture from real task progress
# ------------------------------------------------------------------


def repair_state(**overrides):
    values = {
        "task": "Fix the failing test",
        "task_intent": "repair",
        "files": ["test_a.py"],
    }
    values.update(overrides)
    return CodingState(**values)


def test_capture_matches_measured_metrics():
    state = repair_state(
        tests_run=True,
        last_error="boom",
        error_inspected=True,
        repair_proposed=True,
        user_input_needed=True,
    )

    metrics = measure_task_progress(state)
    snapshot = capture_snapshot(3, state)

    assert snapshot.iteration == 3
    assert snapshot.completion_fraction == metrics.completion_fraction
    assert snapshot.percent == metrics.percent
    assert snapshot.applicable_count == metrics.applicable_count
    assert snapshot.achieved_count == metrics.achieved_count
    assert snapshot.next_milestone == REPAIR_APPLIED
    assert snapshot.state_signature == build_state_signature(state)


def test_capture_not_applicable_milestones_excluded():
    snapshot = capture_snapshot(1, repair_state(tests_run=True, tests_passed=True))

    assert snapshot.applicable_milestones == (
        WORKSPACE_INSPECTED,
        TESTS_RUN,
        TESTS_PASSED,
    )
    assert snapshot.all_milestones_achieved
    assert snapshot.next_milestone is None


def test_history_records_real_regression_after_rollback_state():
    history = ProgressHistory()
    state = repair_state(
        tests_run=True,
        tests_passed=True,
        error_inspected=True,
        repair_proposed=True,
        repair_approved=True,
        repair_applied=True,
    )

    history.capture(1, state)

    # State after rollback_repair_action: repair cleared, tests stale.
    state.repair_proposed = False
    state.repair_approved = False
    state.repair_applied = False
    state.error_inspected = False
    state.tests_run = False
    state.tests_passed = False

    history.capture(2, state)

    assert history.regression_count == 1
    assert history.snapshots[0].all_milestones_achieved
    assert history.latest.percent < 100
    assert history.net_progress < 0
    assert TESTS_PASSED in history.regressions[0].lost_milestones


def test_capture_is_read_only():
    state = repair_state(
        tests_run=True,
        last_error="boom",
        error_inspected=True,
        repair_proposed=True,
        user_input_needed=True,
    )
    before = copy.deepcopy(state)

    history = ProgressHistory()
    history.capture(1, state)
    history.capture(2, state, "test")
    _ = (history.changes, history.peak, history.net_progress)

    assert state == before


def test_capture_requires_classified_intent():
    history = ProgressHistory()

    with pytest.raises(ValueError):
        history.capture(1, CodingState(task="x"))

    assert history.is_empty
