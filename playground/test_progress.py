from flycoder.tools.iteration import IterationObservation
from flycoder.tools.progress import (
    ProgressSnapshot,
    ProgressTracker,
)


def make_observation(
    iteration: int,
    *,
    progress: bool,
    signature: str,
    action: str = "inspect_files",
):
    return IterationObservation(
        iteration=iteration,
        action=action,
        success=True,
        message="Action completed.",
        state_signature=signature,
        progress=progress,
    )


def test_progress_tracker_starts_empty():
    tracker = ProgressTracker()

    assert tracker.iterations == 0
    assert tracker.progress_count == 0
    assert tracker.no_progress_count == 0
    assert tracker.has_progress is False
    assert tracker.stalled is False
    assert tracker.last_signature is None


def test_record_progress():
    tracker = ProgressTracker()

    observation = make_observation(
        1,
        progress=True,
        signature="state-1",
    )

    snapshot = tracker.record(observation)

    assert isinstance(snapshot, ProgressSnapshot)
    assert snapshot.iteration == 1
    assert snapshot.progress is True
    assert snapshot.progress_count == 1
    assert snapshot.no_progress_count == 0
    assert tracker.has_progress is True
    assert tracker.stalled is False


def test_record_no_progress():
    tracker = ProgressTracker()

    observation = make_observation(
        1,
        progress=False,
        signature="state-1",
    )

    snapshot = tracker.record(observation)

    assert snapshot.progress is False
    assert snapshot.progress_count == 0
    assert snapshot.no_progress_count == 1
    assert tracker.has_progress is False
    assert tracker.stalled is True


def test_progress_resets_no_progress_streak():
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

    assert tracker.no_progress_count == 2

    tracker.record(
        make_observation(
            3,
            progress=True,
            signature="state-2",
        )
    )

    assert tracker.progress_count == 1
    assert tracker.no_progress_count == 0
    assert tracker.stalled is False


def test_multiple_progress_iterations_are_counted():
    tracker = ProgressTracker()

    tracker.record(
        make_observation(
            1,
            progress=True,
            signature="state-1",
        )
    )

    tracker.record(
        make_observation(
            2,
            progress=True,
            signature="state-2",
        )
    )

    tracker.record(
        make_observation(
            3,
            progress=True,
            signature="state-3",
        )
    )

    assert tracker.iterations == 3
    assert tracker.progress_count == 3
    assert tracker.no_progress_count == 0
    assert tracker.last_signature == "state-3"


def test_progress_snapshot_preserves_action():
    tracker = ProgressTracker()

    snapshot = tracker.record(
        make_observation(
            1,
            progress=True,
            signature="state-1",
            action="read_file",
        )
    )

    assert snapshot.action == "read_file"


def test_progress_snapshot_preserves_signature():
    tracker = ProgressTracker()

    snapshot = tracker.record(
        make_observation(
            1,
            progress=True,
            signature="unique-state",
        )
    )

    assert snapshot.state_signature == "unique-state"
    assert tracker.last_signature == "unique-state"


def test_reset_clears_progress_state():
    tracker = ProgressTracker()

    tracker.record(
        make_observation(
            1,
            progress=True,
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

    tracker.reset()

    assert tracker.iterations == 0
    assert tracker.progress_count == 0
    assert tracker.no_progress_count == 0
    assert tracker.last_signature is None
    assert tracker.has_progress is False
    assert tracker.stalled is False


def test_progress_followed_by_stall_preserves_total_progress():
    tracker = ProgressTracker()

    tracker.record(
        make_observation(
            1,
            progress=True,
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
            progress=False,
            signature="state-1",
        )
    )

    assert tracker.progress_count == 1
    assert tracker.no_progress_count == 2
    assert tracker.has_progress is True
    assert tracker.stalled is True
