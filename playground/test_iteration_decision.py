import pytest

from flycoder.tools.iteration_decision import (
    ITERATION_CONTINUE,
    ITERATION_ESCALATE,
    ITERATION_STOP,
    ITERATION_VERIFY,
    IterationDecision,
    decide_iteration,
)
from flycoder.tools.repetition import (
    RepetitionAssessment,
    REPETITION_LOOP_RISK,
    REPETITION_NORMAL,
    REPETITION_REPEATED,
)
from flycoder.tools.stall import (
    StallAssessment,
    STALL_CRITICAL,
    STALL_OK,
    STALL_STALLED,
)


def make_stall(
    status: str,
    count: int,
) -> StallAssessment:
    return StallAssessment(
        status=status,
        no_progress_count=count,
        threshold=2,
        message="stall assessment",
    )


def make_repetition(
    status: str,
    count: int,
) -> RepetitionAssessment:
    return RepetitionAssessment(
        status=status,
        action="read_file",
        consecutive_count=count,
        threshold=2,
        message="repetition assessment",
    )


def test_normal_execution_continues():
    decision = decide_iteration()

    assert isinstance(decision, IterationDecision)
    assert decision.decision == ITERATION_CONTINUE
    assert decision.should_continue is True
    assert decision.should_verify is False
    assert decision.should_stop is False
    assert decision.should_escalate is False


def test_finished_task_requires_verification():
    decision = decide_iteration(
        task_finished=True,
    )

    assert decision.decision == ITERATION_VERIFY
    assert decision.should_verify is True


def test_human_input_requires_escalation():
    decision = decide_iteration(
        human_input_required=True,
    )

    assert decision.decision == ITERATION_ESCALATE
    assert decision.should_escalate is True
    assert decision.risks


def test_critical_stall_requires_escalation():
    decision = decide_iteration(
        stall=make_stall(
            STALL_CRITICAL,
            3,
        ),
    )

    assert decision.decision == ITERATION_ESCALATE
    assert decision.should_escalate is True
    assert decision.risks


def test_stalled_execution_can_continue_with_caution():
    decision = decide_iteration(
        stall=make_stall(
            STALL_STALLED,
            2,
        ),
    )

    assert decision.decision == ITERATION_CONTINUE
    assert decision.should_continue is True
    assert decision.risks


def test_loop_risk_requires_escalation():
    decision = decide_iteration(
        repetition=make_repetition(
            REPETITION_LOOP_RISK,
            3,
        ),
    )

    assert decision.decision == ITERATION_ESCALATE
    assert decision.should_escalate is True
    assert decision.risks


def test_repeated_action_adds_risk_but_can_continue():
    decision = decide_iteration(
        repetition=make_repetition(
            REPETITION_REPEATED,
            2,
        ),
    )

    assert decision.decision == ITERATION_CONTINUE
    assert decision.should_continue is True
    assert decision.risks


def test_failed_action_with_recovery_continues():
    decision = decide_iteration(
        action_success=False,
        recovery_available=True,
    )

    assert decision.decision == ITERATION_CONTINUE
    assert decision.should_continue is True


def test_failed_action_without_recovery_stops():
    decision = decide_iteration(
        action_success=False,
        recovery_available=False,
    )

    assert decision.decision == ITERATION_STOP
    assert decision.should_stop is True


def test_stall_and_repetition_risks_are_combined():
    decision = decide_iteration(
        stall=make_stall(
            STALL_STALLED,
            2,
        ),
        repetition=make_repetition(
            REPETITION_REPEATED,
            2,
        ),
    )

    assert decision.decision == ITERATION_CONTINUE
    assert len(decision.risks) == 2


def test_human_input_has_priority_over_finished_state():
    decision = decide_iteration(
        task_finished=True,
        human_input_required=True,
    )

    assert decision.decision == ITERATION_ESCALATE


def test_critical_stall_has_priority_over_normal_failure_recovery():
    decision = decide_iteration(
        action_success=False,
        recovery_available=True,
        stall=make_stall(
            STALL_CRITICAL,
            3,
        ),
    )

    assert decision.decision == ITERATION_ESCALATE


def test_loop_risk_has_priority_over_recovery():
    decision = decide_iteration(
        action_success=False,
        recovery_available=True,
        repetition=make_repetition(
            REPETITION_LOOP_RISK,
            3,
        ),
    )

    assert decision.decision == ITERATION_ESCALATE


def test_ok_assessments_do_not_add_risks():
    decision = decide_iteration(
        stall=make_stall(
            STALL_OK,
            0,
        ),
        repetition=make_repetition(
            REPETITION_NORMAL,
            1,
        ),
    )

    assert decision.decision == ITERATION_CONTINUE
    assert decision.risks == []


def test_decision_does_not_mutate_assessments():
    stall = make_stall(
        STALL_STALLED,
        2,
    )

    repetition = make_repetition(
        REPETITION_REPEATED,
        2,
    )

    decide_iteration(
        stall=stall,
        repetition=repetition,
    )

    assert stall.status == STALL_STALLED
    assert stall.no_progress_count == 2
    assert repetition.status == REPETITION_REPEATED
    assert repetition.consecutive_count == 2


def test_decision_properties_are_exclusive():
    decisions = [
        decide_iteration(),
        decide_iteration(task_finished=True),
        decide_iteration(action_success=False),
        decide_iteration(human_input_required=True),
    ]

    for decision in decisions:
        flags = [
            decision.should_continue,
            decision.should_verify,
            decision.should_stop,
            decision.should_escalate,
        ]

        assert sum(flags) == 1
