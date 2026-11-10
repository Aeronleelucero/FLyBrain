from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.iteration import observe_iteration
from flycoder.tools.task_progress import TaskProgress


def make_observation(
    *,
    iteration: int = 1,
    success: bool = True,
    previous_signature: str | None = None,
):
    state = CodingState(task="Test task")

    result = ActionResult(
        action="inspect_files",
        success=success,
        message="ok" if success else "failed",
    )

    return observe_iteration(
        iteration,
        result,
        state,
        previous_signature=previous_signature,
    )


def test_empty_progress():
    progress = TaskProgress()

    assert progress.total_iterations == 0
    assert progress.progress_iterations == 0
    assert progress.no_progress_iterations == 0
    assert progress.progress_rate == 0.0
    assert progress.has_progress is False
    assert progress.latest_progress is None


def test_record_progress_iteration():
    progress = TaskProgress()

    observation = make_observation(
        iteration=1,
        success=True,
        previous_signature="old",
    )

    progress.record(observation)

    assert progress.total_iterations == 1
    assert progress.progress_iterations == 1
    assert progress.no_progress_iterations == 0
    assert progress.successful_iterations == 1
    assert progress.failed_iterations == 0
    assert progress.progress_rate == 1.0
    assert progress.has_progress is True
    assert progress.latest_progress is True


def test_record_no_progress_iteration():
    progress = TaskProgress()

    observation = make_observation(
        iteration=1,
        success=True,
    )

    progress.record(observation)

    assert progress.total_iterations == 1
    assert progress.progress_iterations == 0
    assert progress.no_progress_iterations == 1
    assert progress.successful_iterations == 1
    assert progress.progress_rate == 0.0
    assert progress.has_progress is False
    assert progress.latest_progress is False


def test_record_failed_iteration():
    progress = TaskProgress()

    observation = make_observation(
        iteration=1,
        success=False,
    )

    progress.record(observation)

    assert progress.total_iterations == 1
    assert progress.successful_iterations == 0
    assert progress.failed_iterations == 1


def test_progress_rate_across_iterations():
    progress = TaskProgress()

    progress.record(
        make_observation(
            iteration=1,
            success=True,
            previous_signature="old",
        )
    )

    progress.record(
        make_observation(
            iteration=2,
            success=True,
        )
    )

    progress.record(
        make_observation(
            iteration=3,
            success=True,
            previous_signature="old",
        )
    )

    assert progress.total_iterations == 3
    assert progress.progress_iterations == 2
    assert progress.no_progress_iterations == 1
    assert progress.progress_rate == 2 / 3


def test_history_preserves_iteration_progress():
    progress = TaskProgress()

    progress.record(
        make_observation(
            iteration=1,
            success=True,
            previous_signature="old",
        )
    )

    progress.record(
        make_observation(
            iteration=2,
            success=True,
        )
    )

    assert progress.history == [True, False]


def test_latest_progress_updates():
    progress = TaskProgress()

    progress.record(
        make_observation(
            iteration=1,
            success=True,
        )
    )

    assert progress.latest_progress is False

    progress.record(
        make_observation(
            iteration=2,
            success=True,
            previous_signature="old",
        )
    )

    assert progress.latest_progress is True


def test_all_iterations_progressed():
    progress = TaskProgress()

    progress.record(
        make_observation(
            iteration=1,
            success=True,
            previous_signature="old",
        )
    )

    progress.record(
        make_observation(
            iteration=2,
            success=True,
            previous_signature="old",
        )
    )

    assert progress.all_iterations_progressed is True


def test_all_iterations_progressed_false_when_one_stalls():
    progress = TaskProgress()

    progress.record(
        make_observation(
            iteration=1,
            success=True,
            previous_signature="old",
        )
    )

    progress.record(
        make_observation(
            iteration=2,
            success=True,
        )
    )

    assert progress.all_iterations_progressed is False


def test_reset_clears_progress():
    progress = TaskProgress()

    progress.record(
        make_observation(
            iteration=1,
            success=True,
            previous_signature="old",
        )
    )

    progress.reset()

    assert progress.total_iterations == 0
    assert progress.successful_iterations == 0
    assert progress.failed_iterations == 0
    assert progress.progress_iterations == 0
    assert progress.no_progress_iterations == 0
    assert progress.history == []
    assert progress.latest_progress is None
