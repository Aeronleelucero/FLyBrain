"""Tests for the Phase 10.4.1 task-level progress model."""

import contextlib
import copy
import io

import pytest

from flycoder.agent import FlyCoderAgent
from flycoder.state import CodingState
from flycoder.tools.task_progress import (
    FAILURE_DIAGNOSED,
    FILES_ANALYZED,
    FILES_SELECTED,
    INTENT_MILESTONES,
    MILESTONE_ACHIEVED,
    MILESTONE_NOT_APPLICABLE,
    MILESTONE_PENDING,
    REPAIR_APPLIED,
    REPAIR_PROPOSED,
    TESTS_PASSED,
    TESTS_RUN,
    WORKSPACE_INSPECTED,
    build_progress_model,
    evaluate_milestones,
)


def statuses(state: CodingState) -> dict[str, str]:
    return {
        milestone.key: milestone.status
        for milestone in evaluate_milestones(state)
    }


def by_key(state: CodingState, key: str):
    return next(
        milestone
        for milestone in evaluate_milestones(state)
        if milestone.key == key
    )


# ------------------------------------------------------------------
# Model construction
# ------------------------------------------------------------------


@pytest.mark.parametrize("intent", sorted(INTENT_MILESTONES))
def test_every_agent_intent_has_a_model(intent):
    model = build_progress_model(intent)

    assert model.intent == intent
    assert model.keys == INTENT_MILESTONES[intent]
    assert model.keys[0] == WORKSPACE_INSPECTED


def test_model_covers_every_agent_classification():
    tasks = [
        "Explain the error",
        "Fix the failing test",
        "Improve the parser",
        "Review the code",
        "Run the tests",
        "Inspect the project",
    ]

    for task in tasks:
        build_progress_model(FlyCoderAgent.classify_task(task))


@pytest.mark.parametrize("intent", [None, ""])
def test_missing_intent_is_rejected(intent):
    with pytest.raises(ValueError, match="classified"):
        build_progress_model(intent)


def test_unknown_intent_is_rejected():
    with pytest.raises(ValueError, match="Unknown task intent"):
        build_progress_model("deploy")


def test_evaluate_uses_state_intent():
    state = CodingState(task="x", task_intent="inspect")

    assert [m.key for m in evaluate_milestones(state)] == [
        WORKSPACE_INSPECTED
    ]


def test_explicit_intent_overrides_state_intent():
    state = CodingState(task="x", task_intent="inspect")

    keys = [m.key for m in evaluate_milestones(state, "test")]

    assert keys == [WORKSPACE_INSPECTED, TESTS_RUN, TESTS_PASSED]


def test_repair_applied_is_the_only_human_milestone():
    for intent in INTENT_MILESTONES:
        for milestone in build_progress_model(intent).milestones:
            assert milestone.requires_human is (
                milestone.key == REPAIR_APPLIED
            )


# ------------------------------------------------------------------
# Inspect / analysis intents
# ------------------------------------------------------------------


def test_inspect_progress():
    state = CodingState(task="x", task_intent="inspect")

    assert statuses(state) == {WORKSPACE_INSPECTED: MILESTONE_PENDING}

    state.files = ["a.py"]

    milestone = by_key(state, WORKSPACE_INSPECTED)

    assert milestone.achieved
    assert milestone.fraction == 1.0
    assert "1 file(s)" in milestone.detail


def test_review_tracks_partial_file_analysis():
    state = CodingState(
        task="x",
        task_intent="review",
        files=["a.py", "b.py", "c.py", "d.py"],
        relevant_files=["a.py", "b.py", "c.py", "d.py"],
        files_analyzed=["a.py"],
    )

    milestone = by_key(state, FILES_ANALYZED)

    assert milestone.pending
    assert milestone.fraction == 0.25
    assert milestone.detail == "1 of 4 file(s) analyzed."

    state.files_analyzed = ["a.py", "b.py", "c.py", "d.py"]

    assert by_key(state, FILES_ANALYZED).achieved


def test_review_ignores_analyzed_files_outside_selection():
    state = CodingState(
        task="x",
        task_intent="review",
        relevant_files=["a.py", "b.py"],
        files_analyzed=["other.py", "a.py"],
    )

    assert by_key(state, FILES_ANALYZED).fraction == 0.5


@pytest.mark.parametrize("intent", ["explain", "improve"])
def test_explain_and_improve_complete_after_one_file(intent):
    state = CodingState(
        task="x",
        task_intent=intent,
        files=["a.py", "b.py"],
        relevant_files=["a.py", "b.py"],
    )

    assert by_key(state, FILES_SELECTED).achieved
    assert by_key(state, FILES_ANALYZED).pending

    state.files_analyzed = ["a.py"]

    assert by_key(state, FILES_ANALYZED).achieved


def test_files_analyzed_pending_without_selection():
    state = CodingState(
        task="x",
        task_intent="review",
        files_analyzed=["a.py"],
    )

    assert by_key(state, FILES_ANALYZED).pending
    assert by_key(state, FILES_ANALYZED).fraction == 0.0


# ------------------------------------------------------------------
# Test intent
# ------------------------------------------------------------------


def test_test_intent_failed_run_is_not_passed():
    state = CodingState(
        task="x",
        task_intent="test",
        files=["test_a.py"],
        tests_run=True,
        tests_passed=False,
    )

    assert statuses(state) == {
        WORKSPACE_INSPECTED: MILESTONE_ACHIEVED,
        TESTS_RUN: MILESTONE_ACHIEVED,
        TESTS_PASSED: MILESTONE_PENDING,
    }
    assert by_key(state, TESTS_PASSED).detail == "Tests ran and failed."


def test_tests_passed_requires_tests_run():
    # Contradictory state is never counted as verified progress.
    state = CodingState(
        task="x",
        task_intent="test",
        tests_run=False,
        tests_passed=True,
    )

    assert by_key(state, TESTS_PASSED).pending


# ------------------------------------------------------------------
# Repair intent
# ------------------------------------------------------------------


def repair_state(**overrides) -> CodingState:
    values = {
        "task": "Fix the failing test",
        "task_intent": "repair",
        "files": ["test_a.py"],
    }
    values.update(overrides)
    return CodingState(**values)


def test_repair_before_tests_is_all_pending():
    state = repair_state()

    result = statuses(state)

    assert result[WORKSPACE_INSPECTED] == MILESTONE_ACHIEVED
    assert all(
        result[key] == MILESTONE_PENDING
        for key in INTENT_MILESTONES["repair"][1:]
    )


def test_repair_not_needed_when_tests_already_pass():
    state = repair_state(tests_run=True, tests_passed=True)

    assert statuses(state) == {
        WORKSPACE_INSPECTED: MILESTONE_ACHIEVED,
        TESTS_RUN: MILESTONE_ACHIEVED,
        FAILURE_DIAGNOSED: MILESTONE_NOT_APPLICABLE,
        REPAIR_PROPOSED: MILESTONE_NOT_APPLICABLE,
        REPAIR_APPLIED: MILESTONE_NOT_APPLICABLE,
        TESTS_PASSED: MILESTONE_ACHIEVED,
    }


def test_repair_failure_observed_then_diagnosed():
    state = repair_state(
        tests_run=True,
        tests_passed=False,
        last_error="boom",
    )

    assert by_key(state, FAILURE_DIAGNOSED).pending
    assert "not diagnosed" in by_key(state, FAILURE_DIAGNOSED).detail

    state.error_inspected = True

    assert by_key(state, FAILURE_DIAGNOSED).achieved
    assert by_key(state, REPAIR_PROPOSED).pending


def test_repair_proposal_waits_for_human_approval():
    state = repair_state(
        tests_run=True,
        last_error="boom",
        error_inspected=True,
        repair_proposed=True,
        user_input_needed=True,
    )

    applied = by_key(state, REPAIR_APPLIED)

    assert by_key(state, REPAIR_PROPOSED).achieved
    assert applied.pending
    assert applied.requires_human
    assert "human approval" in applied.detail


def test_applied_but_unapproved_repair_is_never_counted():
    state = repair_state(
        repair_proposed=True,
        repair_applied=True,
        repair_approved=False,
    )

    assert by_key(state, REPAIR_APPLIED).pending


def test_tests_regress_to_pending_after_approved_repair():
    # Mirrors the state approve_repair_action leaves behind: tests and
    # error information are reset and must be re-established.
    state = repair_state(
        repair_proposed=True,
        repair_approved=True,
        repair_applied=True,
        tests_run=False,
        tests_passed=False,
        last_error=None,
        error_inspected=False,
    )

    result = statuses(state)

    assert result[FAILURE_DIAGNOSED] == MILESTONE_ACHIEVED
    assert result[REPAIR_PROPOSED] == MILESTONE_ACHIEVED
    assert result[REPAIR_APPLIED] == MILESTONE_ACHIEVED
    assert result[TESTS_RUN] == MILESTONE_PENDING
    assert result[TESTS_PASSED] == MILESTONE_PENDING
    assert "re-run" in by_key(state, TESTS_RUN).detail

    state.tests_run = True
    state.tests_passed = True

    assert all(
        status == MILESTONE_ACHIEVED
        for status in statuses(state).values()
    )


# ------------------------------------------------------------------
# Safety: evaluation is observational
# ------------------------------------------------------------------


def test_evaluation_does_not_mutate_state():
    state = repair_state(
        tests_run=True,
        last_error="boom",
        error_inspected=True,
        repair_proposed=True,
        user_input_needed=True,
        relevant_files=["test_a.py"],
    )

    before = copy.deepcopy(state)

    for intent in INTENT_MILESTONES:
        evaluate_milestones(state, intent)

    assert state == before


def test_milestone_status_is_immutable():
    milestone = evaluate_milestones(
        CodingState(task="x", task_intent="inspect")
    )[0]

    with pytest.raises(AttributeError):
        milestone.status = MILESTONE_ACHIEVED


# ------------------------------------------------------------------
# Integration with the real agent
# ------------------------------------------------------------------


def run_quietly(agent: FlyCoderAgent, state: CodingState, steps: int):
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(steps):
            if state.finished or state.user_input_needed:
                break
            agent.run_once(state)


def test_agent_inspect_task_reaches_all_milestones(tmp_path):
    (tmp_path / "module.py").write_text("x = 1\n")

    agent = FlyCoderAgent(tmp_path)
    state = CodingState(task="Inspect the project")

    run_quietly(agent, state, 5)

    assert state.finished
    assert all(m.achieved for m in evaluate_milestones(state))


def test_agent_repair_task_progress_reflects_real_state(tmp_path):
    (tmp_path / "test_sample.py").write_text(
        "def test_sample():\n"
        '    assert False, "Intentional failure for testing"\n'
    )

    agent = FlyCoderAgent(tmp_path)
    state = CodingState(task="Fix the failing test")

    run_quietly(agent, state, 10)

    result = statuses(state)

    assert result[WORKSPACE_INSPECTED] == MILESTONE_ACHIEVED
    assert result[TESTS_RUN] == MILESTONE_ACHIEVED
    assert result[FAILURE_DIAGNOSED] == MILESTONE_ACHIEVED
    assert result[REPAIR_APPLIED] == MILESTONE_PENDING
    assert result[TESTS_PASSED] == MILESTONE_PENDING

    # No file was modified by progress evaluation or the agent.
    assert "assert False" in (tmp_path / "test_sample.py").read_text()
