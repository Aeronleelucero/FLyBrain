import pytest

from flycoder.tools.iteration import IterationObservation
from flycoder.tools.progress import ProgressTracker
from flycoder.tools.stall import (
    STALL_CRITICAL,
    STALL_OK,
    STALL_STALLED,
    StallAssessment,
    assess_stall,
)


def make_observation(
    iteration: int,
    *,
    progress: bool,
    signature: str,
):
    return IterationObservation(
        iteration=iteration,
        action="inspect_files",
        success=True,
        message="Action completed.",
        state_signature=signature,
        progress=progress,
    )


def test_empty_tracker_is_not_stalled():
    tracker = ProgressTracker()

    assessment = assess_stall(tracker)

    assert isinstance(assessment, StallAssessment)
    assert assessment.status == STALL_OK
    assert assessment.no_progress_count == 0
    assert assessment.stalled is False
    assert assessment.critical is False


def test_one_no_progress_iteration_is_ok():
    tracker = ProgressTracker()

    tracker.record(
        make_observation(
            1,
            progress=False,
            signature="state-1",
        )
    )

    assessment = assess_stall(tracker)

    assert assessment.status == STALL_OK
    assert assessment.no_progress_count == 1
    assert assessment.stalled is False


def test_two_no_progress_iterations_are_stalled():
    tracker = ProgressTracker()

    tracker.record(
        make_observation(
            1,
            progress=False,
            signature="state-1",
        )
    )

    tracker.record(
        make_observation(
            2,
            progress=False,
            signature="state-1",
        )
    )

    assessment = assess_stall(tracker)

    assert assessment.status == STALL_STALLED
    assert assessment.no_progress_count == 2
    assert assessment.stalled is True
    assert assessment.critical is False


def test_three_no_progress_iterations_are_critical():
    tracker = ProgressTracker()

    for iteration in range(1, 4):
        tracker.record(
            make_observation(
                iteration,
                progress=False,
                signature="state-1",
            )
        )

    assessment = assess_stall(tracker)

    assert assessment.status == STALL_CRITICAL
    assert assessment.no_progress_count == 3
    assert assessment.stalled is True
    assert assessment.critical is True


def test_progress_resets_stall_count():
    tracker = ProgressTracker()

    tracker.record(
        make_observation(
            1,
            progress=False,
            signature="state-1",
        )
    )

    tracker.record(
        make_observation(
            2,
            progress=False,
            signature="state-1",
        )
    )

    tracker.record(
        make_observation(
            3,
            progress=True,
            signature="state-2",
        )
    )

    assessment = assess_stall(tracker)

    assert assessment.status == STALL_OK
    assert assessment.no_progress_count == 0
    assert assessment.stalled is False


def test_custom_stalled_threshold():
    tracker = ProgressTracker()

    for iteration in range(1, 4):
        tracker.record(
            make_observation(
                iteration,
                progress=False,
                signature="state-1",
            )
        )

    assessment = assess_stall(
        tracker,
        stalled_threshold=3,
        critical_threshold=5,
    )

    assert assessment.status == STALL_STALLED
    assert assessment.no_progress_count == 3
    assert assessment.threshold == 3


def test_custom_critical_threshold():
    tracker = ProgressTracker()

    for iteration in range(1, 5):
        tracker.record(
            make_observation(
                iteration,
                progress=False,
                signature="state-1",
            )
        )

    assessment = assess_stall(
        tracker,
        stalled_threshold=2,
        critical_threshold=4,
    )

    assert assessment.status == STALL_CRITICAL
    assert assessment.no_progress_count == 4
    assert assessment.threshold == 4


def test_invalid_stalled_threshold_is_rejected():
    tracker = ProgressTracker()

    with pytest.raises(
        ValueError,
        match="stalled_threshold must be at least 1",
    ):
        assess_stall(
            tracker,
            stalled_threshold=0,
        )


def test_invalid_critical_threshold_is_rejected():
    tracker = ProgressTracker()

    with pytest.raises(
        ValueError,
        match="critical_threshold must be greater than stalled_threshold",
    ):
        assess_stall(
            tracker,
            stalled_threshold=3,
            critical_threshold=3,
        )


def test_stall_message_describes_condition():
    tracker = ProgressTracker()

    tracker.record(
        make_observation(
            1,
            progress=False,
            signature="state-1",
        )
    )

    tracker.record(
        make_observation(
            2,
            progress=False,
            signature="state-1",
        )
    )

    assessment = assess_stall(tracker)

    assert "stalled" in assessment.message.lower()


def test_critical_message_describes_condition():
    tracker = ProgressTracker()

    for iteration in range(1, 4):
        tracker.record(
            make_observation(
                iteration,
                progress=False,
                signature="state-1",
            )
        )

    assessment = assess_stall(tracker)

    assert "critical" in assessment.message.lower()


def test_stall_assessment_is_advisory_only():
    tracker = ProgressTracker()

    tracker.record(
        make_observation(
            1,
            progress=False,
            signature="state-1",
        )
    )

    tracker.record(
        make_observation(
            2,
            progress=False,
            signature="state-1",
        )
    )

    assessment = assess_stall(tracker)

    assert assessment.stalled is True

    # The assessment itself does not mutate tracker state.
    assert tracker.no_progress_count == 2
    assert tracker.iterations == 2
