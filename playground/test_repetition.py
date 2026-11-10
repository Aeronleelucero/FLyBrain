import pytest

from flycoder.tools.repetition import (
    ActionRepetitionTracker,
    RepetitionAssessment,
    REPETITION_LOOP_RISK,
    REPETITION_NORMAL,
    REPETITION_REPEATED,
)


def test_new_tracker_is_normal():
    tracker = ActionRepetitionTracker()

    assessment = tracker.assess()

    assert isinstance(assessment, RepetitionAssessment)
    assert assessment.status == REPETITION_NORMAL
    assert assessment.action is None
    assert assessment.consecutive_count == 0
    assert assessment.repeated is False
    assert assessment.loop_risk is False


def test_first_action_is_normal():
    tracker = ActionRepetitionTracker()

    assessment = tracker.record("read_file")

    assert assessment.status == REPETITION_NORMAL
    assert assessment.action == "read_file"
    assert assessment.consecutive_count == 1
    assert assessment.repeated is False
    assert assessment.loop_risk is False


def test_second_consecutive_action_is_repeated():
    tracker = ActionRepetitionTracker()

    tracker.record("read_file")
    assessment = tracker.record("read_file")

    assert assessment.status == REPETITION_REPEATED
    assert assessment.action == "read_file"
    assert assessment.consecutive_count == 2
    assert assessment.repeated is True
    assert assessment.loop_risk is False


def test_third_consecutive_action_is_loop_risk():
    tracker = ActionRepetitionTracker()

    tracker.record("read_file")
    tracker.record("read_file")
    assessment = tracker.record("read_file")

    assert assessment.status == REPETITION_LOOP_RISK
    assert assessment.action == "read_file"
    assert assessment.consecutive_count == 3
    assert assessment.repeated is True
    assert assessment.loop_risk is True


def test_different_action_resets_consecutive_count():
    tracker = ActionRepetitionTracker()

    tracker.record("read_file")
    tracker.record("read_file")

    assessment = tracker.record("inspect_files")

    assert assessment.status == REPETITION_NORMAL
    assert assessment.action == "inspect_files"
    assert assessment.consecutive_count == 1


def test_repetition_can_resume_after_different_action():
    tracker = ActionRepetitionTracker()

    tracker.record("read_file")
    tracker.record("read_file")
    tracker.record("inspect_files")

    assessment = tracker.record("inspect_files")

    assert assessment.status == REPETITION_REPEATED
    assert assessment.action == "inspect_files"
    assert assessment.consecutive_count == 2


def test_empty_action_does_not_create_repetition():
    tracker = ActionRepetitionTracker()

    assessment = tracker.record("")

    assert assessment.status == REPETITION_NORMAL
    assert assessment.action is None
    assert assessment.consecutive_count == 0
    assert tracker.current_action is None


def test_actions_are_recorded():
    tracker = ActionRepetitionTracker()

    tracker.record("read_file")
    tracker.record("read_file")
    tracker.record("run_tests")

    assert tracker.actions == [
        "read_file",
        "read_file",
        "run_tests",
    ]


def test_custom_repeated_threshold():
    tracker = ActionRepetitionTracker()

    tracker.record("read_file")
    tracker.record("read_file")
    tracker.record("read_file")

    assessment = tracker.assess(
        repeated_threshold=3,
        loop_risk_threshold=5,
    )

    assert assessment.status == REPETITION_REPEATED
    assert assessment.consecutive_count == 3
    assert assessment.threshold == 3


def test_custom_loop_risk_threshold():
    tracker = ActionRepetitionTracker()

    for _ in range(4):
        tracker.record("read_file")

    assessment = tracker.assess(
        repeated_threshold=2,
        loop_risk_threshold=4,
    )

    assert assessment.status == REPETITION_LOOP_RISK
    assert assessment.consecutive_count == 4
    assert assessment.threshold == 4


def test_invalid_repeated_threshold_is_rejected():
    tracker = ActionRepetitionTracker()

    with pytest.raises(
        ValueError,
        match="repeated_threshold must be at least 1",
    ):
        tracker.assess(repeated_threshold=0)


def test_invalid_loop_risk_threshold_is_rejected():
    tracker = ActionRepetitionTracker()

    with pytest.raises(
        ValueError,
        match="loop_risk_threshold must be greater than",
    ):
        tracker.assess(
            repeated_threshold=3,
            loop_risk_threshold=3,
        )


def test_reset_clears_repetition_state():
    tracker = ActionRepetitionTracker()

    tracker.record("read_file")
    tracker.record("read_file")
    tracker.record("read_file")

    tracker.reset()

    assert tracker.actions == []
    assert tracker.current_action is None
    assert tracker.consecutive_count == 0

    assessment = tracker.assess()

    assert assessment.status == REPETITION_NORMAL
    assert assessment.action is None
    assert assessment.consecutive_count == 0


def test_message_describes_repetition():
    tracker = ActionRepetitionTracker()

    tracker.record("read_file")
    assessment = tracker.record("read_file")

    assert "repeated" in assessment.message.lower()


def test_message_describes_loop_risk():
    tracker = ActionRepetitionTracker()

    tracker.record("read_file")
    tracker.record("read_file")
    assessment = tracker.record("read_file")

    assert "loop risk" in assessment.message.lower()
