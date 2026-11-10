"""Tests for Phase 10.8: fine-grained autonomous completion outcomes."""

import contextlib
import copy
import io
from pathlib import Path

import pytest

from flycoder import cli as cli_module
from flycoder.actions.registry import ActionResult
from flycoder.agent import FlyCoderAgent, cli as agent_cli
from flycoder.state import CodingState
from flycoder.tools.autonomous import (
    AUTONOMOUS_BLOCKED,
    AUTONOMOUS_CANCELLED,
    AUTONOMOUS_COMPLETED,
    AUTONOMOUS_FAILED,
    AUTONOMOUS_MAX_ITERATIONS,
    AUTONOMOUS_NEEDS_HUMAN,
    AUTONOMOUS_UNVERIFIED,
    VERIFICATION_ACTION,
    _complete_finished_task,
    run_autonomous_task,
)
from flycoder.tools.completion_evidence import (
    EVIDENCE_FAILED,
    EVIDENCE_INCOMPLETE,
    EVIDENCE_INCONSISTENT,
    EVIDENCE_MISSING,
    EVIDENCE_NEEDS_APPROVAL,
    EVIDENCE_VERIFIED,
    evaluate_completion_evidence,
)
from flycoder.tools.completion_outcome import (
    OUTCOME_AWAITING_HUMAN,
    OUTCOME_BLOCKED,
    OUTCOME_BUDGET_EXHAUSTED,
    OUTCOME_CANCELLED,
    OUTCOME_DEFINITIONS,
    OUTCOME_EXECUTION_FAILED,
    OUTCOME_IN_PROGRESS,
    OUTCOME_PARTIALLY_COMPLETED,
    OUTCOME_VERIFICATION_FAILED,
    OUTCOME_VERIFIED_COMPLETE,
    OUTCOMES,
    CompletionOutcome,
    VerificationAttempt,
    classify_completion_outcome,
    infer_stop_status,
    meaningful_progress,
    outcome_consistency_problems,
    success_blockers,
    verification_attempt,
)
from flycoder.tools.human_input import HUMAN_INPUT_REPAIR_APPROVAL
from flycoder.tools.human_intervention import (
    DECISION_APPROVE,
    DECISION_CANCEL_TASK,
    DECISION_PROVIDE_INFORMATION,
    DECISION_REJECT,
    HumanDecision,
)
from flycoder.tools.progress_report import build_progress_report
from flycoder.tools.task_lifecycle import (
    TASK_BLOCKED,
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_NEEDS_HUMAN,
    TASK_RUNNING,
    TASK_VERIFYING,
    TaskLifecycle,
)
from flycoder.tools.task_progress import measure_task_progress
from flycoder.tools.task_session import TaskSession


PLAYGROUND = Path(__file__).parent

ORIGINAL = (
    "def test_sample():\n"
    '    assert False, "Intentional failure for testing"\n'
)
REPAIRED = ORIGINAL.replace("assert False", "assert True")
# The automatic repair fixes test_sample but test_other keeps failing,
# so verification of the applied change fails.
PARTIALLY_FIXABLE = ORIGINAL + "\n\ndef test_other():\n    assert 1 == 2\n"

REPAIR_TASK = "Fix the failing test"


def quietly(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def write(tmp_path, content=ORIGINAL):
    (tmp_path / "test_sample.py").write_text(content)


def disk(tmp_path):
    return (tmp_path / "test_sample.py").read_text()


def start(tmp_path, content=ORIGINAL, *, max_iterations=20):
    write(tmp_path, content)
    return quietly(
        TaskSession.start,
        FlyCoderAgent(tmp_path),
        REPAIR_TASK,
        max_iterations=max_iterations,
    )


def decide(session, kind, **kwargs):
    request = session.intervention_request()
    assert request is not None
    return session.decide(
        HumanDecision(kind=kind, intervention_id=request.intervention_id, **kwargs)
    )


def approve(session):
    request = session.intervention_request()
    return decide(
        session,
        DECISION_APPROVE,
        proposal_fingerprint=request.proposal_fingerprint,
        confirmed=True,
    )


def run(agent, state, **kwargs):
    return quietly(run_autonomous_task, agent, state, **kwargs)


class ScriptedAgent:
    """Agent stand-in applying scripted (action, success, changes) steps."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = 0

    @staticmethod
    def classify_task(task):
        return FlyCoderAgent.classify_task(task)

    def run_once(self, state):
        action, success, changes = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1

        for name, value in changes.items():
            setattr(state, name, value)

        return ActionResult(action=action, success=success, message=action)


def make_test_state(**overrides):
    values = dict(task="Run the tests", task_intent="test", files=["test_a.py"])
    values.update(overrides)
    return CodingState(**values)


def classify(state, stop_status, lifecycle_status=None, **kwargs):
    return classify_completion_outcome(
        state,
        evidence_for(state),
        stop_status=stop_status,
        lifecycle_status=lifecycle_status,
        **kwargs,
    )


def evidence_for(state):
    return evaluate_completion_evidence(state)


# ------------------------------------------------------------------
# Taxonomy
# ------------------------------------------------------------------


def test_every_outcome_has_a_complete_definition():
    assert OUTCOMES == set(OUTCOME_DEFINITIONS)

    for identifier, definition in OUTCOME_DEFINITIONS.items():
        assert definition.identifier == identifier
        assert definition.description
        assert definition.required_evidence
        assert definition.next_step
        assert definition.expected_lifecycle


def test_only_verified_completion_is_successful():
    successful = {i for i, d in OUTCOME_DEFINITIONS.items() if d.successful}

    assert successful == {OUTCOME_VERIFIED_COMPLETE}
    assert OUTCOME_DEFINITIONS[OUTCOME_VERIFIED_COMPLETE].verification_required


def test_awaiting_human_is_resumable_and_distinct_from_blocked():
    awaiting = OUTCOME_DEFINITIONS[OUTCOME_AWAITING_HUMAN]
    blocked = OUTCOME_DEFINITIONS[OUTCOME_BLOCKED]

    assert awaiting.terminal is False and awaiting.resumable is True
    assert blocked.terminal is True and blocked.resumable is False
    assert awaiting.human_action_required and blocked.human_action_required

    resumable = {i for i, d in OUTCOME_DEFINITIONS.items() if d.resumable}
    assert resumable == {OUTCOME_AWAITING_HUMAN}


def test_outcomes_map_onto_existing_lifecycle_states():
    assert OUTCOME_DEFINITIONS[OUTCOME_VERIFIED_COMPLETE].expected_lifecycle == {
        TASK_COMPLETED
    }
    assert OUTCOME_DEFINITIONS[OUTCOME_CANCELLED].expected_lifecycle == {
        TASK_CANCELLED
    }
    assert OUTCOME_DEFINITIONS[OUTCOME_AWAITING_HUMAN].expected_lifecycle == {
        TASK_NEEDS_HUMAN
    }

    for outcome in (
        OUTCOME_PARTIALLY_COMPLETED,
        OUTCOME_EXECUTION_FAILED,
        OUTCOME_VERIFICATION_FAILED,
        OUTCOME_BUDGET_EXHAUSTED,
    ):
        assert TASK_COMPLETED not in OUTCOME_DEFINITIONS[outcome].expected_lifecycle


# ------------------------------------------------------------------
# Outcomes from real and scripted runs
# ------------------------------------------------------------------


def test_verified_successful_completion():
    result = run(FlyCoderAgent(PLAYGROUND), CodingState(task="Inspect playground files"))

    assert result.status == AUTONOMOUS_COMPLETED
    assert result.outcome.outcome == OUTCOME_VERIFIED_COMPLETE
    assert result.successful and result.outcome.successful
    assert result.outcome.verified
    assert result.outcome.terminal and not result.outcome.can_resume
    assert result.progress_report.successful
    assert result.progress_report.outcome is result.outcome
    assert result.progress_report.consistent


def test_partial_completion_is_not_reported_as_complete():
    # A repair that ran the tests and diagnosed the failure, then stopped
    # without a proposal: real progress, but not complete.
    state = CodingState(
        task=REPAIR_TASK,
        task_intent="repair",
        files=["test_a.py"],
        tests_run=True,
        tests_code_version=0,
        last_error="1 failed",
        error_inspected=True,
    )

    result = run(
        ScriptedAgent(("finish", True, {"finished": True})),
        state,
        max_iterations=3,
    )

    outcome = result.outcome

    assert result.status == AUTONOMOUS_UNVERIFIED
    assert outcome.outcome == OUTCOME_PARTIALLY_COMPLETED
    assert not outcome.successful and not result.successful
    assert {"workspace_inspected", "tests_run", "failure_diagnosed"} <= set(
        outcome.completed_work
    )
    assert "repair_applied" in outcome.remaining_work
    assert result.lifecycle.status == TASK_FAILED
    assert "Successful: no" in result.progress_report.render()


# ------------------------------------------------------------------
# Meaningful partial completion (Phase 10.8 review correction)
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "intent, extra",
    [
        ("test", {}),
        ("repair", {}),
        # Automatic file selection is incidental too.
        ("explain", {"relevant_files": ["a.py"]}),
        ("review", {"relevant_files": ["a.py", "b.py"]}),
        ("improve", {"relevant_files": ["a.py"]}),
    ],
)
def test_discovery_alone_is_not_partial_completion(intent, extra):
    state = CodingState(
        task="Task", task_intent=intent, files=["a.py", "b.py"], **extra
    )

    outcome = classify(state, AUTONOMOUS_UNVERIFIED, TASK_FAILED)

    assert outcome.outcome == OUTCOME_EXECUTION_FAILED
    assert outcome.outcome != OUTCOME_PARTIALLY_COMPLETED
    # Discovery stays visible as completed work.
    assert "workspace_inspected" in outcome.completed_work
    assert meaningful_progress(
        measure_task_progress(state, intent).milestones, state
    ) == ()
    assert any("discovery" in reason for reason in outcome.reasons)


def test_discovery_only_run_is_not_partial():
    result = run(
        ScriptedAgent(("finish", True, {"finished": True})),
        make_test_state(),
        max_iterations=3,
    )

    assert result.status == AUTONOMOUS_UNVERIFIED
    assert result.outcome.outcome == OUTCOME_EXECUTION_FAILED
    assert "workspace_inspected" in result.outcome.completed_work
    assert "tests_run" in result.outcome.remaining_work


def test_inspection_has_no_partial_outcome_from_discovery():
    state = CodingState(task="Inspect", task_intent="inspect", files=["a.py"])

    # Discovery is the inspect criterion: it verifies the inspection...
    assert evidence_for(state).verified
    assert classify(state, AUTONOMOUS_COMPLETED, TASK_COMPLETED).outcome == (
        OUTCOME_VERIFIED_COMPLETE
    )

    # ...but is never counted as partial progress when a run stops early.
    stopped = classify(state, AUTONOMOUS_UNVERIFIED, TASK_FAILED)

    assert stopped.outcome != OUTCOME_PARTIALLY_COMPLETED
    assert "workspace_inspected" in stopped.completed_work


def test_partially_reviewed_files_are_partial_completion():
    state = CodingState(
        task="Review the code",
        task_intent="review",
        files=["a.py", "b.py"],
        relevant_files=["a.py", "b.py"],
        files_analyzed=["a.py"],
    )

    result = run(
        ScriptedAgent(("finish", True, {"finished": True})),
        state,
        max_iterations=3,
    )

    outcome = result.outcome

    assert result.completion_evidence.status == EVIDENCE_INCOMPLETE
    assert outcome.outcome == OUTCOME_PARTIALLY_COMPLETED
    assert "files_analyzed" in outcome.remaining_work  # 1 of 2 analyzed
    assert any("files_analyzed" in reason for reason in outcome.reasons)


def improvement_state(**overrides):
    """An improve task that analyzed its file and holds an unapproved proposal."""

    values = dict(
        task="Improve the code",
        task_intent="improve",
        files=["a.py"],
        relevant_files=["a.py"],
        files_analyzed=["a.py"],
        current_file="a.py",
        current_file_content="print('x')\n",
        repair_proposed=True,
        repair_description="Use logging.",
        proposed_content="print('x')\n",
    )
    values.update(overrides)
    return CodingState(**values)


def test_improvement_analysis_with_unapproved_proposal_after_stop_is_partial():
    # The task stopped with no active human-intervention request.
    state = improvement_state()

    outcome = classify(state, AUTONOMOUS_UNVERIFIED, TASK_FAILED)

    assert state.user_input_needed is False
    assert evidence_for(state).status == EVIDENCE_NEEDS_APPROVAL
    assert outcome.outcome == OUTCOME_PARTIALLY_COMPLETED
    assert "files_analyzed" in outcome.completed_work


def test_improvement_analysis_with_pending_human_intervention_awaits_human():
    state = improvement_state(
        user_input_needed=True,
        human_input_reason=HUMAN_INPUT_REPAIR_APPROVAL,
    )
    evidence = evidence_for(state)
    stop_status = infer_stop_status(state, evidence, budget_exhausted=False)

    outcome = classify(state, stop_status, TASK_NEEDS_HUMAN)

    assert evidence.status == EVIDENCE_NEEDS_APPROVAL
    assert stop_status == AUTONOMOUS_NEEDS_HUMAN
    assert outcome.outcome == OUTCOME_AWAITING_HUMAN
    assert outcome.outcome != OUTCOME_PARTIALLY_COMPLETED
    assert outcome.human_action_required is True
    assert outcome.terminal is False
    assert outcome.definition.resumable is True
    assert outcome.can_resume is True

    # The autonomous loop stops at the same boundary with the same outcome.
    result = run(
        ScriptedAgent(
            (
                "improve_code",
                True,
                {
                    "user_input_needed": True,
                    "human_input_reason": HUMAN_INPUT_REPAIR_APPROVAL,
                },
            )
        ),
        improvement_state(),
    )

    assert result.status == AUTONOMOUS_NEEDS_HUMAN
    assert result.lifecycle.status == TASK_NEEDS_HUMAN
    assert result.outcome.outcome == OUTCOME_AWAITING_HUMAN
    assert result.outcome.human_action_required is True
    assert result.outcome.terminal is False
    assert result.outcome.can_resume is True


def test_test_task_with_applied_change_awaiting_retest_is_partial():
    state = make_test_state(
        current_file="test_a.py",
        current_file_content=REPAIRED,
        repair_proposed=True,
        repair_approved=True,
        repair_applied=True,
        repair_description="Fix.",
        proposed_content=REPAIRED,
        repair_original_content=ORIGINAL,
        code_version=1,
    )

    outcome = classify(state, AUTONOMOUS_UNVERIFIED, TASK_FAILED)

    assert evidence_for(state).status == EVIDENCE_MISSING
    assert outcome.outcome == OUTCOME_PARTIALLY_COMPLETED
    assert any("repair_applied" in reason for reason in outcome.reasons)


def test_meaningful_progress_never_overrides_failed_verification():
    applied_and_failing = CodingState(
        task=REPAIR_TASK,
        task_intent="repair",
        files=["test_a.py"],
        current_file="test_a.py",
        current_file_content=REPAIRED,
        repair_proposed=True,
        repair_approved=True,
        repair_applied=True,
        repair_description="Fix.",
        proposed_content=REPAIRED,
        repair_original_content=ORIGINAL,
        code_version=1,
        tests_run=True,
        tests_code_version=1,
        last_error="1 failed",
    )

    assert evidence_for(applied_and_failing).status == EVIDENCE_FAILED
    assert classify(
        applied_and_failing, AUTONOMOUS_UNVERIFIED, TASK_FAILED
    ).outcome == OUTCOME_VERIFICATION_FAILED

    reviewed = CodingState(
        task="Review the code",
        task_intent="review",
        files=["a.py", "b.py"],
        relevant_files=["a.py", "b.py"],
        files_analyzed=["a.py"],
    )

    assert classify(
        reviewed,
        AUTONOMOUS_UNVERIFIED,
        TASK_FAILED,
        verification_attempts=(VerificationAttempt(2, False),),
    ).outcome == OUTCOME_VERIFICATION_FAILED


def test_finish_without_any_work_is_an_execution_failure():
    state = CodingState(task="Inspect playground files", task_intent="inspect", finished=True)

    result = run(FlyCoderAgent(PLAYGROUND), state)

    assert result.status == AUTONOMOUS_UNVERIFIED
    assert result.outcome.outcome == OUTCOME_EXECUTION_FAILED
    assert result.outcome.completed_work == ()


def test_execution_failure_reports_the_unresolved_error():
    state = CodingState(task="Review the code", task_intent="review", files=["a.py"])

    result = run(
        ScriptedAgent(("review_code", False, {})),
        state,
        max_iterations=5,
    )

    assert result.status == AUTONOMOUS_FAILED
    assert result.outcome.outcome == OUTCOME_EXECUTION_FAILED
    assert "review_code" in result.outcome.unresolved_issues
    assert not result.successful


def test_failed_test_run_is_a_verification_failure():
    result = run(
        ScriptedAgent(
            (
                "finish",
                True,
                {
                    "finished": True,
                    "tests_run": True,
                    "tests_passed": False,
                    "last_error": "1 failed",
                },
            )
        ),
        make_test_state(),
    )

    assert result.completion_evidence.status == EVIDENCE_FAILED
    assert result.outcome.outcome == OUTCOME_VERIFICATION_FAILED
    assert "1 failed" in result.outcome.unresolved_issues
    assert not result.successful


def test_original_failure_before_a_change_is_not_a_verification_failure():
    state = CodingState(
        task=REPAIR_TASK,
        task_intent="repair",
        files=["test_a.py"],
        tests_run=True,
        last_error="1 failed",
    )

    outcome = classify(state, AUTONOMOUS_UNVERIFIED, TASK_FAILED)

    assert evidence_for(state).status == EVIDENCE_FAILED
    assert outcome.outcome == OUTCOME_PARTIALLY_COMPLETED
    assert "1 failed" in outcome.unresolved_issues


def test_guardrail_block_is_blocked_not_failed():
    # Contradictory state makes the real guardrail block the first action.
    state = CodingState(task="Run the tests", tests_passed=True)

    result = run(FlyCoderAgent(PLAYGROUND), state, max_iterations=5)

    outcome = result.outcome

    assert result.status == AUTONOMOUS_BLOCKED
    assert outcome.outcome == OUTCOME_BLOCKED
    assert outcome.terminal and not outcome.can_resume
    assert outcome.human_action_required
    assert outcome.unresolved_issues


def test_inconsistent_evidence_at_finish_is_blocked():
    result = run(
        ScriptedAgent(("finish", True, {"finished": True, "tests_passed": True})),
        make_test_state(),
    )

    assert result.completion_evidence.status == EVIDENCE_INCONSISTENT
    assert result.outcome.outcome == OUTCOME_BLOCKED
    assert any("Inconsistent" in issue for issue in result.outcome.unresolved_issues)
    assert not result.successful


def test_human_intervention_pending_is_awaiting_human(tmp_path):
    session = start(tmp_path)
    outcome = session.outcome

    assert session.status == AUTONOMOUS_NEEDS_HUMAN
    assert outcome.outcome == OUTCOME_AWAITING_HUMAN
    assert outcome.lifecycle_status == TASK_NEEDS_HUMAN
    assert outcome.terminal is False
    assert outcome.can_resume and outcome.human_action_required
    assert any("approval" in issue for issue in outcome.unresolved_issues)
    assert disk(tmp_path) == ORIGINAL


def test_user_cancellation_is_never_success(tmp_path):
    session = start(tmp_path)
    decide(session, DECISION_CANCEL_TASK, information="Not now.")

    outcome = session.outcome

    assert session.status == AUTONOMOUS_CANCELLED
    assert outcome.outcome == OUTCOME_CANCELLED
    assert not outcome.successful and not outcome.can_resume
    assert session.result.progress_report.outcome_status == OUTCOME_CANCELLED

    with pytest.raises(ValueError):
        session.resume()

    with pytest.raises(ValueError):
        session.lifecycle.transition(TASK_COMPLETED)

    assert session.outcome.outcome == OUTCOME_CANCELLED


def test_cancellation_wins_over_verified_evidence():
    state = CodingState(task="Inspect", task_intent="inspect", files=["a.py"])

    outcome = classify(state, AUTONOMOUS_CANCELLED, TASK_CANCELLED)

    assert outcome.verified
    assert outcome.outcome == OUTCOME_CANCELLED
    assert not outcome.successful


def test_iteration_budget_exhaustion_is_not_success():
    result = run(
        FlyCoderAgent(PLAYGROUND),
        CodingState(task="Inspect playground files"),
        max_iterations=1,
    )

    assert result.status == AUTONOMOUS_MAX_ITERATIONS
    # The evidence happens to verify inspection, but the agent never
    # concluded the task within its budget.
    assert result.completion_evidence.status == EVIDENCE_VERIFIED
    assert result.outcome.outcome == OUTCOME_BUDGET_EXHAUSTED
    assert not result.successful


def test_session_without_budget_cannot_resume(tmp_path):
    # inspect, run_tests, fix_error, read_file, propose_repair: the five
    # iterations reach the approval boundary with nothing left.
    session = start(tmp_path, max_iterations=5)

    assert session.outcome.outcome == OUTCOME_AWAITING_HUMAN
    assert session.remaining_iterations == 0
    assert session.outcome.can_resume is False
    assert "Can resume: no" in session.report().render()


# ------------------------------------------------------------------
# Resumption, recovery, and preserved history
# ------------------------------------------------------------------


def test_resumed_task_outcome_is_recalculated(tmp_path):
    session = start(tmp_path)
    first_steps = list(session.steps)
    errors_before = list(session.lifecycle.errors)

    approve(session)

    # The decision is recorded but nothing has run: still awaiting.
    assert session.outcome.outcome == OUTCOME_AWAITING_HUMAN

    quietly(session.resume)

    assert session.outcome.outcome == OUTCOME_VERIFIED_COMPLETE
    assert session.result.successful
    assert session.steps[: len(first_steps)] == first_steps
    assert session.lifecycle.errors[: len(errors_before)] == errors_before
    assert session.report().outcome_status == OUTCOME_VERIFIED_COMPLETE
    assert disk(tmp_path) == REPAIRED


def test_cancellation_only_intervention_cannot_resume(tmp_path):
    session = start(tmp_path, "def test_sample():\n    assert 1 == 2\n")

    for _ in range(2):
        assert session.outcome.can_resume
        decide(session, DECISION_PROVIDE_INFORMATION, information="Try again.")
        quietly(session.resume)

    assert session.intervention_request().allowed_decisions == (DECISION_CANCEL_TASK,)
    assert session.outcome.outcome == OUTCOME_AWAITING_HUMAN
    assert session.outcome.can_resume is False
    assert session.outcome.human_action_required


def test_recovery_followed_by_successful_verification(tmp_path):
    session = start(tmp_path)
    approve(session)
    quietly(session.resume)

    outcome = session.outcome

    # The initial failing test run was recorded as a recovery attempt.
    assert outcome.recovery_attempts >= 1
    assert outcome.recovery_attempts == len(session.result.recovery_history)
    assert [a.passed for a in outcome.verification_attempts] == [True]
    assert outcome.outcome == OUTCOME_VERIFIED_COMPLETE


def test_recovery_followed_by_failed_verification(tmp_path):
    session = start(tmp_path, PARTIALLY_FIXABLE)
    approve(session)
    quietly(session.resume)

    outcome = session.outcome

    assert outcome.outcome == OUTCOME_AWAITING_HUMAN
    assert outcome.evidence_status == EVIDENCE_FAILED
    assert [a.passed for a in outcome.verification_attempts] == [False]
    assert outcome.verification_failures == 1
    assert not outcome.successful
    assert "latest failed" in outcome.render()


def test_failed_verification_history_survives_resume_and_cancel(tmp_path):
    session = start(tmp_path, PARTIALLY_FIXABLE)
    approve(session)
    quietly(session.resume)

    failed_attempt = session.outcome.verification_attempts[-1]
    recovery_before = len(session.result.recovery_history)

    # Reject the failed repair (authorizes rollback), resume, then cancel.
    decide(session, DECISION_REJECT)
    quietly(session.resume)

    assert failed_attempt in session.outcome.verification_attempts
    assert len(session.result.recovery_history) >= recovery_before

    decide(session, DECISION_CANCEL_TASK)

    assert session.outcome.outcome == OUTCOME_CANCELLED
    assert failed_attempt in session.outcome.verification_attempts


def test_failed_verification_needs_a_newer_passing_verification():
    state = make_test_state(tests_run=True, tests_passed=True, tests_code_version=0)
    evidence = evidence_for(state)
    failed = VerificationAttempt(iteration=3, passed=False)
    passed = VerificationAttempt(iteration=5, passed=True)

    assert evidence.verified
    assert success_blockers(evidence) == ()
    assert success_blockers(evidence, (failed,))
    assert success_blockers(evidence, (failed, passed)) == ()

    lifecycle = TaskLifecycle(task=state.task, status=TASK_RUNNING)
    status, _ = _complete_finished_task(lifecycle, evidence, "done", (failed,))

    assert status == AUTONOMOUS_UNVERIFIED
    assert lifecycle.status == TASK_FAILED

    outcome = classify(
        state,
        AUTONOMOUS_UNVERIFIED,
        TASK_FAILED,
        verification_attempts=(failed,),
    )

    assert outcome.outcome == OUTCOME_VERIFICATION_FAILED


def test_failed_verification_without_recovery_is_a_verification_failure():
    result = run(ScriptedAgent((VERIFICATION_ACTION, False, {})), make_test_state())

    assert result.status == AUTONOMOUS_FAILED
    assert result.outcome.outcome == OUTCOME_VERIFICATION_FAILED
    assert result.outcome.verification_failures == 1


def test_loop_does_not_complete_after_unsuperseded_failed_verification():
    # The verification run fails and records its error (as the real
    # action does), so recovery continues; then the state is made to
    # look verified without any newer verification run.
    failed_run = {
        "tests_run": True,
        "tests_passed": False,
        "tests_code_version": 0,
        "last_error": "verification failed",
    }
    looks_verified = {
        "tests_passed": True,
        "last_error": None,
        "finished": True,
    }

    result = run(
        ScriptedAgent(
            (VERIFICATION_ACTION, False, failed_run),
            ("finish", True, looks_verified),
        ),
        make_test_state(),
    )

    assert result.iterations == 2
    assert result.completion_evidence.status == EVIDENCE_VERIFIED
    assert result.status == AUTONOMOUS_UNVERIFIED
    assert result.lifecycle.status == TASK_FAILED
    assert result.outcome.outcome == OUTCOME_VERIFICATION_FAILED
    assert not result.successful


def test_verification_attempts_ignore_guardrail_blocked_runs():
    blocked = ActionResult(
        action=VERIFICATION_ACTION,
        success=False,
        message="blocked",
        data={"guardrail": {"allowed": False}},
    )
    ran = ActionResult(action=VERIFICATION_ACTION, success=False, message="x")

    assert verification_attempt(1, blocked) is None
    assert verification_attempt(2, ran) == VerificationAttempt(2, False)
    assert verification_attempt(3, ActionResult("run_tests", True, "")) is None


# ------------------------------------------------------------------
# Classification invariants
# ------------------------------------------------------------------


def test_classification_is_deterministic_and_read_only():
    state = make_test_state(tests_run=True, last_error="1 failed")
    before = copy.deepcopy(state)
    evidence = evidence_for(state)

    first = classify_completion_outcome(
        state, evidence, stop_status=AUTONOMOUS_UNVERIFIED, lifecycle_status=TASK_FAILED
    )
    second = classify_completion_outcome(
        state, evidence, stop_status=AUTONOMOUS_UNVERIFIED, lifecycle_status=TASK_FAILED
    )

    assert first == second
    assert first.render() == second.render()
    assert state == before


def test_missing_evidence_is_rejected_not_assumed():
    with pytest.raises(TypeError):
        classify_completion_outcome(
            make_test_state(), None, stop_status=AUTONOMOUS_COMPLETED
        )

    with pytest.raises(ValueError):
        classify(make_test_state(), "DONE")


def test_contradictory_completion_record_is_blocked():
    state = make_test_state()  # MISSING evidence

    outcome = classify(state, AUTONOMOUS_COMPLETED, TASK_COMPLETED)

    assert outcome.outcome == OUTCOME_BLOCKED
    assert not outcome.successful
    assert any("MISSING" in reason for reason in outcome.reasons)
    assert outcome_consistency_problems(outcome)


def test_successful_outcome_cannot_be_hand_built_from_bad_inputs():
    fields = dict(
        outcome=OUTCOME_VERIFIED_COMPLETE,
        stop_status=AUTONOMOUS_COMPLETED,
        lifecycle_status=TASK_COMPLETED,
        evidence_status=EVIDENCE_VERIFIED,
        intent="test",
        verified=True,
        completed_work=(),
        remaining_work=(),
        unresolved_issues=(),
        verification_attempts=(),
        recovery_attempts=0,
        human_action_required=False,
        can_resume=False,
        reasons=(),
    )

    CompletionOutcome(**fields)  # valid

    bad = [
        dict(evidence_status=EVIDENCE_FAILED, verified=False),
        dict(lifecycle_status=TASK_FAILED),
        dict(stop_status=AUTONOMOUS_CANCELLED),
        dict(verification_attempts=(VerificationAttempt(1, False),)),
        dict(evidence_status=EVIDENCE_FAILED),  # verified mismatch
        dict(can_resume=True),
    ]

    for overrides in bad:
        with pytest.raises(ValueError):
            CompletionOutcome(**{**fields, **overrides})


def test_running_task_is_in_progress():
    outcome = classify(make_test_state(), None, TASK_VERIFYING)

    assert outcome.outcome == OUTCOME_IN_PROGRESS
    assert not outcome.terminal and not outcome.successful


@pytest.mark.parametrize(
    "state, budget_exhausted, expected",
    [
        (dict(user_input_needed=True), False, AUTONOMOUS_NEEDS_HUMAN),
        (dict(finished=True), False, AUTONOMOUS_UNVERIFIED),
        (
            dict(finished=True, tests_run=True, tests_passed=True),
            False,
            AUTONOMOUS_COMPLETED,
        ),
        ({}, True, AUTONOMOUS_MAX_ITERATIONS),
        ({}, False, AUTONOMOUS_UNVERIFIED),
    ],
)
def test_infer_stop_status_uses_the_same_success_gate(state, budget_exhausted, expected):
    state = make_test_state(**state)

    assert (
        infer_stop_status(state, evidence_for(state), budget_exhausted=budget_exhausted)
        == expected
    )


def test_finished_outcomes_match_their_lifecycle(tmp_path):
    results = [
        run(FlyCoderAgent(PLAYGROUND), CodingState(task="Inspect playground files")),
        run(FlyCoderAgent(PLAYGROUND), CodingState(task="Inspect playground files"), max_iterations=1),
        run(ScriptedAgent(("finish", True, {"finished": True})), make_test_state()),
        run(FlyCoderAgent(PLAYGROUND), CodingState(task="Run the tests", tests_passed=True)),
    ]

    for result in results:
        outcome = result.outcome
        assert outcome.lifecycle_status == result.lifecycle.status
        assert outcome.lifecycle_status in outcome.definition.expected_lifecycle
        assert outcome_consistency_problems(
            outcome, result.completion_evidence, result.lifecycle.status
        ) == ()
        assert outcome.successful == (result.lifecycle.status == TASK_COMPLETED)


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------


def test_report_renders_the_outcome_and_task_id(tmp_path):
    session = start(tmp_path)
    rendered = session.report().render()

    assert f"Task ID: {session.task_id}" in rendered
    assert "Outcome: AWAITING_HUMAN" in rendered
    assert "Successful: no" in rendered
    assert "Work verified: no" in rendered
    assert "Human action required: yes" in rendered
    assert "Can resume: yes" in rendered
    assert "Remaining:" in rendered


def test_report_flags_an_outcome_that_disagrees_with_the_lifecycle():
    state = make_test_state()
    evidence = evidence_for(state)
    outcome = classify_completion_outcome(
        state, evidence, stop_status=AUTONOMOUS_UNVERIFIED, lifecycle_status=TASK_FAILED
    )
    lifecycle = TaskLifecycle(task=state.task, status=TASK_BLOCKED)

    report = build_progress_report(state, None, evidence, lifecycle, outcome=outcome)

    assert any("lifecycle" in item for item in report.inconsistencies)

    with pytest.raises(TypeError):
        build_progress_report(state, outcome="PARTIALLY_COMPLETED")


def run_cli(tmp_path, argv, *, answer=None, entry=None):
    def input_fn(prompt):
        if answer is None:
            raise AssertionError("input must not be requested")
        return answer

    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        (entry or cli_module.main)(
            [str(tmp_path), *argv],
            input_fn=input_fn,
            is_interactive=True,
        )

    return buffer.getvalue()


def test_cli_reports_awaiting_human_without_success(tmp_path):
    write(tmp_path)

    output = run_cli(tmp_path, ["--task", REPAIR_TASK])

    assert "Outcome: AWAITING_HUMAN" in output
    assert "Successful: no" in output
    # The CLI keeps no paused session, so it cannot resume.
    assert "Can resume: no" in output


def test_cli_reports_verified_completion_after_approval(tmp_path):
    write(tmp_path)

    output = run_cli(tmp_path, ["--task", REPAIR_TASK, "--approve"], answer="yes")

    assert "Outcome: VERIFIED_COMPLETE" in output
    assert "Successful: yes" in output
    assert "Verification runs: 1 (0 failed)" in output


def test_cli_rejection_is_partial_not_success(tmp_path):
    write(tmp_path)

    output = run_cli(tmp_path, ["--task", REPAIR_TASK, "--approve"], answer="no")

    assert "Outcome: PARTIALLY_COMPLETED" in output
    assert "Successful: no" in output
    assert disk(tmp_path) == ORIGINAL


def test_agent_cli_no_longer_claims_success_for_unverified_work(tmp_path):
    write(tmp_path)

    output = run_cli(
        tmp_path,
        ["--task", REPAIR_TASK, "--max-steps", "2"],
        entry=agent_cli,
    )

    assert "Agent finished." not in output
    assert "Outcome: BUDGET_EXHAUSTED" in output
    assert "Successful: no" in output


def test_agent_cli_reports_verified_inspection(tmp_path):
    write(tmp_path)

    output = run_cli(tmp_path, ["--task", "Inspect the project"], entry=agent_cli)

    assert "Outcome: VERIFIED_COMPLETE" in output
    assert "Successful: yes" in output
