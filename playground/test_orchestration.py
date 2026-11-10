"""Tests for Phase 10.9: full agent orchestration."""

import contextlib
import copy
import io
from pathlib import Path

import pytest

import flycoder
import flycoder.actions
from flycoder.actions.registry import ActionRegistry, ActionResult
from flycoder.agent import FlyCoderAgent
from flycoder.state import CodingState
from flycoder.tools import autonomous
from flycoder.tools import task_session as task_session_module
from flycoder.tools.autonomous import (
    AUTONOMOUS_CANCELLED,
    AUTONOMOUS_COMPLETED,
    AUTONOMOUS_FAILED,
    COMPONENT_ERROR_ACTION,
    ResumeContext,
    attached_result,
    run_autonomous_task,
)
from flycoder.tools.completion_outcome import (
    OUTCOME_EXECUTION_FAILED,
    OUTCOME_VERIFICATION_FAILED,
    OUTCOME_VERIFIED_COMPLETE,
    outcome_consistency_problems,
)
from flycoder.tools.human_intervention import (
    DECISION_APPROVE,
    DECISION_CANCEL_TASK,
    DECISION_REJECT,
    HumanDecision,
    HumanDecisionError,
)
from flycoder.tools.progress_history import ProgressHistory
from flycoder.tools.recovery import RecoveryHistory
from flycoder.tools.task_lifecycle import (
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_INITIALIZING,
    TASK_NEEDS_HUMAN,
    TASK_RUNNING,
    TASK_VERIFYING,
    TaskLifecycle,
)
from flycoder.tools.task_session import TaskSession


PLAYGROUND = Path(__file__).parent

ORIGINAL = (
    "def test_sample():\n"
    '    assert False, "Intentional failure for testing"\n'
)
# The automatic repair fixes test_sample but test_other keeps failing.
PARTIALLY_FIXABLE = ORIGINAL + "\n\ndef test_other():\n    assert 1 == 2\n"

REPAIR_TASK = "Fix the failing test"

ACTIVE_STATES = {TASK_INITIALIZING, TASK_RUNNING, TASK_VERIFYING}


# ------------------------------------------------------------------
# Step 1 — Registry unification
# ------------------------------------------------------------------


def test_package_registry_is_the_single_action_registry():
    assert flycoder.create_action_registry is flycoder.actions.create_action_registry


def test_package_registry_includes_verification_and_rollback():
    registry = flycoder.create_action_registry()

    assert isinstance(registry, ActionRegistry)
    assert registry.list_actions() == flycoder.actions.create_action_registry().list_actions()
    assert {"verify_changes", "rollback_repair"} <= set(registry.list_actions())


def test_package_keeps_previously_exported_names():
    for name in (
        "ActionRegistry",
        "approve_repair_action",
        "create_action_registry",
        "explain_error_action",
        "fix_error_action",
        "improve_code_action",
        "inspect_files_action",
        "propose_repair_action",
        "read_file_action",
        "review_code_action",
        "run_tests_action",
        "write_file_action",
    ):
        assert hasattr(flycoder, name), name


# ------------------------------------------------------------------
# Step 2 — Failure isolation and interrupt handling
# ------------------------------------------------------------------


def quietly(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def write(tmp_path, content=ORIGINAL):
    (tmp_path / "test_sample.py").write_text(content)


def disk(tmp_path):
    return (tmp_path / "test_sample.py").read_text()


class StepAgent:
    """Agent stand-in: each step is (action, success, changes) or an
    exception instance to raise."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = 0

    @staticmethod
    def classify_task(task):
        return FlyCoderAgent.classify_task(task)

    def run_once(self, state):
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1

        if isinstance(step, BaseException):
            raise step

        action, success, changes = step

        for name, value in changes.items():
            setattr(state, name, value)

        return ActionResult(action=action, success=success, message=action)


def make_state():
    return CodingState(task="Run the tests", task_intent="test", files=["a.py"])


def run(agent, state, **kwargs):
    return quietly(run_autonomous_task, agent, state, **kwargs)


def start_session(tmp_path, content=ORIGINAL):
    write(tmp_path, content)
    return quietly(TaskSession.start, FlyCoderAgent(tmp_path), REPAIR_TASK)


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


def assert_consistent(result):
    """Lifecycle, stop status, outcome, and success agree."""

    assert result.lifecycle.status not in ACTIVE_STATES

    if result.outcome is not None:
        assert result.outcome.lifecycle_status == result.lifecycle.status
        assert result.outcome.stop_status == result.status
        assert outcome_consistency_problems(
            result.outcome,
            result.completion_evidence,
            result.lifecycle.status,
        ) == ()

    assert result.successful == (
        result.outcome is not None and result.outcome.successful
    )


def assert_session_failed(session):
    assert session.status == AUTONOMOUS_FAILED
    assert session.lifecycle.status == TASK_FAILED
    assert session.needs_human is False
    assert session.result.successful is False
    assert_consistent(session.result)


# --- first run -----------------------------------------------------


def test_component_exception_during_first_run_fails_consistently():
    agent = StepAgent(("inspect_files", True, {}), RuntimeError("boom"))

    result = run(agent, make_state())

    assert result.status == AUTONOMOUS_FAILED
    assert result.lifecycle.status == TASK_FAILED
    assert result.outcome.outcome == OUTCOME_EXECUTION_FAILED
    assert not result.successful
    assert_consistent(result)

    assert result.error.type_name == "RuntimeError"
    assert result.error.message == "boom"
    assert result.error.stage == "run_once"
    assert result.error.iteration == 2
    assert not result.interrupted

    # The completed step is kept; the failing iteration is recorded.
    assert [s.result.action for s in result.steps] == [
        "inspect_files",
        COMPONENT_ERROR_ACTION,
    ]
    assert result.steps[1].iteration == 2
    assert result.steps[1].result.data["component_error"]["stage"] == "run_once"
    assert [snapshot.iteration for snapshot in result.progress_history] == [0, 1]
    assert any("RuntimeError: boom" in e for e in result.lifecycle.errors)
    assert any("RuntimeError" in issue for issue in result.outcome.unresolved_issues)


@pytest.mark.parametrize(
    "target, stage, success",
    [
        ("recover_from_failure", "recovery", False),
        ("observe_iteration", "iteration_observation", True),
        ("assess_confidence", "step", True),
    ],
)
def test_loop_internal_exceptions_fail_consistently(monkeypatch, target, stage, success):
    def broken(*args, **kwargs):
        raise ValueError(f"{target} broke")

    monkeypatch.setattr(autonomous, target, broken)

    result = run(StepAgent(("run_tests", success, {})), make_state())

    assert result.status == AUTONOMOUS_FAILED
    assert result.error.stage == stage
    assert result.steps[-1].result.action == COMPONENT_ERROR_ACTION
    assert not result.successful
    assert_consistent(result)


def test_precondition_errors_are_raised_before_anything_changes():
    with pytest.raises(ValueError):
        run(StepAgent(("inspect_files", True, {})), make_state(), max_iterations=0)

    with pytest.raises(TypeError):
        run(StepAgent(("inspect_files", True, {})), make_state(), on_step="print")


# --- resumed run ---------------------------------------------------


def test_component_exception_during_resume_does_not_wedge_the_session(tmp_path):
    # Regression: this used to leave the lifecycle RUNNING while the
    # session still reported NEEDS_HUMAN, and refused resume and cancel.
    session = start_session(tmp_path)
    first_steps = list(session.steps)
    recovery_history = session.result.recovery_history
    recovery_attempts = len(recovery_history)
    approve(session)

    def crash(state):
        raise RuntimeError("component crashed")

    session.agent.run_once = crash

    result = quietly(session.resume)

    assert result is session.result
    assert_session_failed(session)
    assert session.outcome.outcome == OUTCOME_EXECUTION_FAILED
    assert session.result.error.message == "component crashed"

    # History and human decisions are preserved.
    assert session.steps[: len(first_steps)] == first_steps
    assert session.steps[-1].result.action == COMPONENT_ERROR_ACTION
    assert session.result.recovery_history is recovery_history
    assert len(recovery_history) == recovery_attempts
    assert [d.kind for d in session.decisions] == [DECISION_APPROVE]
    assert disk(tmp_path) == ORIGINAL

    with pytest.raises(ValueError, match="failed"):
        session.resume()

    with pytest.raises(HumanDecisionError):
        session.decide(HumanDecision(kind=DECISION_CANCEL_TASK, intervention_id="x"))


def test_verification_history_survives_a_later_component_failure(tmp_path):
    session = start_session(tmp_path, PARTIALLY_FIXABLE)
    approve(session)
    quietly(session.resume)  # verification fails; human input needed

    failed_attempt = session.outcome.verification_attempts[-1]
    assert failed_attempt.passed is False

    decide(session, DECISION_REJECT)  # authorizes rollback

    def crash(state):
        raise RuntimeError("crash after rejection")

    session.agent.run_once = crash
    quietly(session.resume)

    assert_session_failed(session)
    assert failed_attempt in session.outcome.verification_attempts
    assert session.outcome.outcome == OUTCOME_VERIFICATION_FAILED


def test_resume_precondition_failure_restores_the_session(monkeypatch, tmp_path):
    session = start_session(tmp_path)
    approve(session)
    finished_before = session.state.finished

    def broken(agent, state):
        raise RuntimeError("cannot resolve intent")

    monkeypatch.setattr(autonomous, "_resolve_intent", broken)

    with pytest.raises(RuntimeError, match="cannot resolve intent"):
        session.resume()

    # Nothing started: the task is still paused and resumable.
    assert session.lifecycle.status == TASK_NEEDS_HUMAN
    assert session.needs_human
    assert session.state.finished == finished_before


# --- evidence, finalization, and report failures --------------------


def test_evidence_failure_yields_a_degraded_result_never_success(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("evidence unavailable")

    monkeypatch.setattr(autonomous, "evaluate_completion_evidence", broken)

    result = run(StepAgent(("inspect_files", True, {})), make_state())

    assert result.status == AUTONOMOUS_FAILED
    assert result.lifecycle.status == TASK_FAILED
    assert result.outcome is None and not result.classified
    assert result.completion_evidence is None
    assert result.progress_report is None
    assert not result.successful
    assert result.error.stage == "evidence"
    assert any("Finalization failed" in note for note in result.diagnostics)
    assert_consistent(result)


def test_evidence_failure_mid_run_keeps_completed_steps(monkeypatch):
    original = autonomous.evaluate_completion_evidence
    calls = {"count": 0}

    def flaky(*args, **kwargs):
        calls["count"] += 1

        if calls["count"] >= 3:
            raise RuntimeError("evidence broke")

        return original(*args, **kwargs)

    monkeypatch.setattr(autonomous, "evaluate_completion_evidence", flaky)

    result = run(
        StepAgent(("inspect_files", True, {}), ("run_tests", True, {})),
        make_state(),
    )

    assert [s.result.action for s in result.steps] == [
        "inspect_files",
        COMPONENT_ERROR_ACTION,
    ]
    assert result.steps[0].completion_evidence is not None
    assert result.outcome is None
    assert result.lifecycle.status == TASK_FAILED
    assert not result.successful


def test_classification_failure_never_reports_completion(monkeypatch):
    original = autonomous.classify_completion_outcome

    def refuse_completion(*args, **kwargs):
        if kwargs.get("stop_status") == AUTONOMOUS_COMPLETED:
            raise RuntimeError("classifier broke")

        return original(*args, **kwargs)

    monkeypatch.setattr(autonomous, "classify_completion_outcome", refuse_completion)

    result = run(FlyCoderAgent(PLAYGROUND), CodingState(task="Inspect playground files"))

    # Classification happens before the transition, so the lifecycle
    # never reached COMPLETED.
    assert TASK_COMPLETED not in [t.to_state for t in result.lifecycle.transitions]
    assert result.status == AUTONOMOUS_FAILED
    assert result.lifecycle.status == TASK_FAILED
    assert result.outcome.outcome == OUTCOME_EXECUTION_FAILED
    assert result.error.stage == "finalization"
    assert not result.successful
    assert_consistent(result)


def test_report_failure_is_a_diagnostic_only(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("report broke")

    monkeypatch.setattr(autonomous, "build_progress_report", broken)

    result = run(FlyCoderAgent(PLAYGROUND), CodingState(task="Inspect playground files"))

    assert result.progress_report is None
    assert any("Progress report unavailable" in note for note in result.diagnostics)
    # The report is not authoritative: the verified outcome stands.
    assert result.status == AUTONOMOUS_COMPLETED
    assert result.outcome.outcome == OUTCOME_VERIFIED_COMPLETE
    assert result.successful
    assert result.error is None
    assert_consistent(result)


def test_cancel_finalization_failure_still_cancels(monkeypatch, tmp_path):
    session = start_session(tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError("evidence unavailable")

    monkeypatch.setattr(task_session_module, "evaluate_completion_evidence", broken)

    decide(session, DECISION_CANCEL_TASK)

    assert session.status == AUTONOMOUS_CANCELLED
    assert session.lifecycle.status == TASK_CANCELLED
    assert session.outcome is None
    assert not session.result.successful
    assert any("Finalization failed" in note for note in session.diagnostics)

    with pytest.raises(ValueError, match="cancelled"):
        session.resume()


def test_refresh_failure_keeps_the_previous_consistent_outcome(monkeypatch, tmp_path):
    session = start_session(tmp_path)
    outcome_before = session.outcome
    evidence_before = session.result.completion_evidence

    def broken(*args, **kwargs):
        raise RuntimeError("classifier broke")

    monkeypatch.setattr(task_session_module, "_classify_outcome", broken)

    decide(session, DECISION_REJECT)

    assert session.outcome is outcome_before
    assert session.result.completion_evidence is evidence_before
    assert any("Outcome refresh failed" in note for note in session.diagnostics)


# --- KeyboardInterrupt and SystemExit -------------------------------


@pytest.mark.parametrize(
    "interrupt",
    [KeyboardInterrupt(), SystemExit(3)],
    ids=["KeyboardInterrupt", "SystemExit"],
)
def test_interrupts_propagate_as_the_same_object(interrupt):
    agent = StepAgent(("inspect_files", True, {}), interrupt)

    with pytest.raises(type(interrupt)) as raised:
        run(agent, make_state())

    assert raised.value is interrupt

    result = attached_result(interrupt)

    assert result is not None
    assert result.interrupted
    assert result.status == AUTONOMOUS_FAILED
    assert result.lifecycle.status == TASK_FAILED
    assert result.outcome.outcome == OUTCOME_EXECUTION_FAILED
    assert not result.successful
    assert [s.result.action for s in result.steps] == [
        "inspect_files",
        COMPONENT_ERROR_ACTION,
    ]
    assert_consistent(result)


def test_system_exit_code_is_preserved():
    exit_request = SystemExit(7)

    with pytest.raises(SystemExit) as raised:
        run(StepAgent(exit_request), make_state())

    assert raised.value is exit_request
    assert raised.value.code == 7


def test_interrupt_during_finalization_is_classified_before_transition(monkeypatch):
    interrupt = KeyboardInterrupt()
    original = autonomous.classify_completion_outcome
    calls = {"count": 0}

    def interrupt_once(*args, **kwargs):
        calls["count"] += 1

        if calls["count"] == 1:
            raise interrupt

        return original(*args, **kwargs)

    monkeypatch.setattr(autonomous, "classify_completion_outcome", interrupt_once)

    with pytest.raises(KeyboardInterrupt) as raised:
        run(FlyCoderAgent(PLAYGROUND), CodingState(task="Inspect playground files"))

    assert raised.value is interrupt

    result = attached_result(interrupt)

    assert result.lifecycle.status == TASK_FAILED
    assert TASK_COMPLETED not in [t.to_state for t in result.lifecycle.transitions]
    assert not result.successful
    assert_consistent(result)


def test_repeated_interrupt_still_leaves_a_terminal_lifecycle(monkeypatch):
    interrupt = KeyboardInterrupt()

    def always_interrupt(*args, **kwargs):
        raise interrupt

    monkeypatch.setattr(autonomous, "classify_completion_outcome", always_interrupt)

    lifecycle = TaskLifecycle(task="Run the tests")

    for status in (TASK_INITIALIZING, TASK_RUNNING, TASK_NEEDS_HUMAN):
        lifecycle.transition(status)

    history = ProgressHistory()
    history.capture(0, make_state(), "test")
    resume = ResumeContext(
        lifecycle=lifecycle,
        progress_history=history,
        recovery_history=RecoveryHistory(),
        iteration_offset=0,
    )

    with pytest.raises(KeyboardInterrupt) as raised:
        run(StepAgent(interrupt), make_state(), resume=resume)

    assert raised.value is interrupt
    # The finally guard: never left RUNNING or VERIFYING.
    assert lifecycle.status == TASK_FAILED


def test_interrupt_during_session_resume_keeps_the_session_consistent(tmp_path):
    session = start_session(tmp_path)
    first_steps = list(session.steps)
    approve(session)
    interrupt = KeyboardInterrupt()

    def interrupted(state):
        raise interrupt

    session.agent.run_once = interrupted

    with pytest.raises(KeyboardInterrupt) as raised:
        session.resume()

    assert raised.value is interrupt
    assert_session_failed(session)
    assert session.result.interrupted
    assert session.steps[: len(first_steps)] == first_steps
    assert session.steps[-1].result.action == COMPONENT_ERROR_ACTION
    assert disk(tmp_path) == ORIGINAL


def test_session_records_an_interrupt_whose_result_was_not_attached(monkeypatch, tmp_path):
    session = start_session(tmp_path)
    approve(session)
    interrupt = KeyboardInterrupt()

    def interrupted(state):
        raise interrupt

    session.agent.run_once = interrupted
    monkeypatch.setattr(autonomous, "attach_result", lambda error, result: None)

    with pytest.raises(KeyboardInterrupt) as raised:
        session.resume()

    assert raised.value is interrupt
    assert_session_failed(session)
    assert session.result.interrupted
    assert any("interrupted before a result" in note for note in session.diagnostics)


# --- observer callbacks ---------------------------------------------


def test_observer_failure_does_not_change_a_successful_run():
    def broken_observer(step):
        raise ValueError("printer broke")

    result = run(
        FlyCoderAgent(PLAYGROUND),
        CodingState(task="Inspect playground files"),
        on_step=broken_observer,
    )

    assert result.status == AUTONOMOUS_COMPLETED
    assert result.outcome.outcome == OUTCOME_VERIFIED_COMPLETE
    assert result.successful
    assert result.error is None
    assert any("Step observer failed" in note for note in result.diagnostics)
    assert_consistent(result)


def test_observer_sees_every_recorded_step_in_order():
    seen = []
    agent = StepAgent(("inspect_files", True, {}), RuntimeError("boom"))

    result = run(agent, make_state(), on_step=seen.append)

    assert seen == result.steps
    assert seen[-1].result.action == COMPONENT_ERROR_ACTION


def test_observer_interrupt_still_propagates():
    interrupt = KeyboardInterrupt()

    def interrupting_observer(step):
        raise interrupt

    with pytest.raises(KeyboardInterrupt) as raised:
        run(StepAgent(("inspect_files", True, {})), make_state(), on_step=interrupting_observer)

    assert raised.value is interrupt

    result = attached_result(interrupt)

    assert result.interrupted
    assert result.lifecycle.status == TASK_FAILED
    # The observed step itself completed and is kept.
    assert [s.result.action for s in result.steps] == ["inspect_files"]


# ------------------------------------------------------------------
# Step 3 — Unified TaskSession entry point
# ------------------------------------------------------------------


def inspect_session():
    return TaskSession(
        FlyCoderAgent(PLAYGROUND),
        CodingState(task="Inspect playground files"),
    )


def failed_session():
    session = TaskSession(
        StepAgent(("inspect_files", True, {}), RuntimeError("boom")),
        make_state(),
    )
    quietly(session.run)
    return session


# --- run() ---------------------------------------------------------


def test_run_executes_the_initial_task():
    session = inspect_session()

    assert session.result is None and not session.started

    result = quietly(session.run)

    assert result is session.result
    assert session.started
    assert session.status == AUTONOMOUS_COMPLETED
    assert session.outcome.outcome == OUTCOME_VERIFIED_COMPLETE
    assert session.steps == result.steps


def test_run_classifies_the_intent_once_before_execution():
    seen = []

    class RecordingAgent(StepAgent):
        def run_once(self, state):
            seen.append(state.task_intent)
            return super().run_once(state)

    state = CodingState(task="Run the tests", files=["a.py"])
    session = TaskSession(
        RecordingAgent(("finish", True, {"finished": True})),
        state,
    )

    quietly(session.run)

    assert seen == ["test"]
    assert state.task_intent == "test"


def test_run_classification_failure_leaves_the_session_unstarted():
    class BrokenClassifier(StepAgent):
        @staticmethod
        def classify_task(task):
            raise RuntimeError("cannot classify")

    session = TaskSession(
        BrokenClassifier(("inspect_files", True, {})),
        CodingState(task="Anything"),
    )

    with pytest.raises(RuntimeError, match="cannot classify"):
        session.run()

    assert not session.started
    assert session.result is None


@pytest.mark.parametrize("make_session", ["completed", "failed", "paused"])
def test_a_session_cannot_run_twice(make_session, tmp_path):
    if make_session == "completed":
        session = inspect_session()
        quietly(session.run)
    elif make_session == "failed":
        session = failed_session()
    else:
        session = start_session(tmp_path)

    steps_before = list(session.steps)

    with pytest.raises(ValueError, match="already run"):
        session.run()

    assert session.steps == steps_before


def test_an_interrupted_run_keeps_the_session_handle_and_cannot_rerun():
    interrupt = KeyboardInterrupt()
    session = TaskSession(
        StepAgent(("inspect_files", True, {}), interrupt),
        make_state(),
    )

    with pytest.raises(KeyboardInterrupt) as raised:
        quietly(session.run)

    assert raised.value is interrupt
    assert session.result is attached_result(interrupt)
    assert_session_failed(session)
    assert session.result.interrupted

    with pytest.raises(ValueError, match="already run"):
        session.run()


def test_start_accepts_an_observer_and_resume_passes_it_on(tmp_path):
    seen = []
    write(tmp_path)
    session = quietly(
        TaskSession.start,
        FlyCoderAgent(tmp_path),
        REPAIR_TASK,
        on_step=seen.append,
    )

    assert seen == session.steps

    approve(session)
    quietly(session.resume, on_step=seen.append)

    assert seen == session.steps
    assert session.outcome.outcome == OUTCOME_VERIFIED_COMPLETE


# --- close() -------------------------------------------------------


def test_closing_a_paused_session_refuses_resume_and_decisions(tmp_path):
    session = start_session(tmp_path)

    assert session.outcome.can_resume

    session.close("Controller exited.")

    assert session.closed
    assert session.close_reason == "Controller exited."
    # Closing changes neither the lifecycle nor the classification.
    assert session.lifecycle.status == TASK_NEEDS_HUMAN
    assert session.outcome.outcome == "AWAITING_HUMAN"
    assert session.outcome.can_resume is False
    assert "Can resume: no" in session.report().render()

    request = session.intervention_request()
    state_before = copy.deepcopy(session.state)

    with pytest.raises(HumanDecisionError, match="closed"):
        session.decide(
            HumanDecision(
                kind=DECISION_APPROVE,
                intervention_id=request.intervention_id,
                proposal_fingerprint=request.proposal_fingerprint,
                confirmed=True,
            )
        )

    with pytest.raises(ValueError, match="closed"):
        session.resume()

    assert session.state == state_before
    assert session.decisions == ()
    assert disk(tmp_path) == ORIGINAL


def test_closing_after_a_decision_still_refuses_resume(tmp_path):
    session = start_session(tmp_path)
    approve(session)
    session.close()

    with pytest.raises(ValueError, match="closed"):
        session.resume()

    assert disk(tmp_path) == ORIGINAL


def test_closing_a_completed_session_keeps_its_outcome():
    session = inspect_session()
    quietly(session.run)
    outcome = session.outcome

    session.close()

    assert session.outcome == outcome
    assert session.result.successful
    assert session.lifecycle.status == TASK_COMPLETED


def test_closing_a_failed_session_keeps_it_failed():
    session = failed_session()

    session.close()

    assert_session_failed(session)
    assert session.outcome.outcome == OUTCOME_EXECUTION_FAILED


def test_closing_a_cancelled_session_keeps_it_cancelled(tmp_path):
    session = start_session(tmp_path)
    decide(session, DECISION_CANCEL_TASK)

    session.close()

    assert session.status == AUTONOMOUS_CANCELLED
    assert session.lifecycle.status == TASK_CANCELLED


def test_closing_before_running_refuses_the_run():
    session = inspect_session()
    session.close()

    with pytest.raises(ValueError, match="closed"):
        session.run()

    assert session.result is None


def test_closing_an_active_session_takes_effect_when_the_run_stops(tmp_path):
    write(tmp_path)
    session = TaskSession(FlyCoderAgent(tmp_path), CodingState(task=REPAIR_TASK))

    def close_mid_run(step):
        if step.iteration == 2:
            session.close("Closed while running.")

    quietly(session.run, on_step=close_mid_run)

    # The run itself was not cut short; it stopped at the human boundary.
    assert session.lifecycle.status == TASK_NEEDS_HUMAN
    assert session.closed
    assert session.outcome.can_resume is False

    with pytest.raises(ValueError, match="closed"):
        session.resume()


def test_close_is_idempotent_and_requires_a_reason():
    session = inspect_session()

    with pytest.raises(ValueError):
        session.close("  ")

    session.close("first")
    session.close("second")

    assert session.close_reason == "first"


# --- agent entry points --------------------------------------------


def test_start_task_returns_a_running_task_session():
    agent = FlyCoderAgent(PLAYGROUND)
    seen = []

    session = quietly(
        agent.start_task,
        "Inspect playground files",
        max_iterations=5,
        on_step=seen.append,
    )

    assert isinstance(session, TaskSession)
    assert session.agent is agent
    assert session.max_iterations == 5
    assert session.started and not session.closed
    assert session.status == AUTONOMOUS_COMPLETED
    assert seen == session.steps


def test_start_task_keeps_a_paused_task_resumable(tmp_path):
    write(tmp_path)
    session = quietly(FlyCoderAgent(tmp_path).start_task, REPAIR_TASK)

    assert session.outcome.can_resume

    approve(session)
    quietly(session.resume)

    assert session.outcome.outcome == OUTCOME_VERIFIED_COMPLETE


def test_start_task_accepts_prepared_state():
    state = CodingState(task="Inspect playground files")
    session = quietly(
        FlyCoderAgent(PLAYGROUND).start_task,
        "Inspect playground files",
        state=state,
    )

    assert session.state is state

    with pytest.raises(ValueError, match="different task"):
        FlyCoderAgent(PLAYGROUND).start_task("Other task", state=state)


def test_run_task_routes_through_a_closed_session(monkeypatch):
    agent = FlyCoderAgent(PLAYGROUND)
    sessions = []
    original = agent.start_task

    def spy(*args, **kwargs):
        session = original(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(agent, "start_task", spy)

    result = quietly(agent.run_task, "Inspect playground files", max_iterations=5)

    assert isinstance(result, autonomous.AutonomousTaskResult)
    assert result.status == AUTONOMOUS_COMPLETED
    assert result.successful
    assert len(sessions) == 1
    assert sessions[0].result is result
    assert sessions[0].closed


def test_run_task_paused_result_reports_it_cannot_resume(tmp_path):
    write(tmp_path)

    result = quietly(FlyCoderAgent(tmp_path).run_task, REPAIR_TASK)

    assert result.status == "NEEDS_HUMAN"
    assert result.outcome.outcome == "AWAITING_HUMAN"
    assert result.outcome.can_resume is False
    assert "Can resume: no" in result.progress_report.render()
    assert disk(tmp_path) == ORIGINAL


@pytest.mark.parametrize("entry", ["start_task", "run_task"])
def test_agent_entry_points_propagate_interrupts_unchanged(entry):
    agent = FlyCoderAgent(PLAYGROUND)
    interrupt = KeyboardInterrupt()

    def interrupted(state):
        raise interrupt

    agent.run_once = interrupted

    with pytest.raises(KeyboardInterrupt) as raised:
        quietly(getattr(agent, entry), "Inspect playground files")

    assert raised.value is interrupt

    result = attached_result(interrupt)

    assert result.interrupted
    assert result.lifecycle.status == TASK_FAILED
    assert not result.successful


# ------------------------------------------------------------------
# Step 4 — Deeply immutable OrchestrationSnapshot
# ------------------------------------------------------------------

import dataclasses  # noqa: E402
import datetime  # noqa: E402
import enum  # noqa: E402

from flycoder import cli as cli_module  # noqa: E402
from flycoder.actions.registry import ActionRegistry as _Registry  # noqa: E402
from flycoder.agent import cli as agent_cli  # noqa: E402
from flycoder.tools.completion_outcome import (  # noqa: E402
    OUTCOME_AWAITING_HUMAN,
    OUTCOME_BUDGET_EXHAUSTED,
    OUTCOME_CANCELLED,
)
from flycoder.tools.human_input import HUMAN_INPUT_NO_SAFE_REPAIR  # noqa: E402
from flycoder.tools.orchestration import (  # noqa: E402
    FrozenMapping,
    OrchestrationSnapshot,
    freeze,
    is_deeply_immutable,
)

REPAIRED = ORIGINAL.replace("assert False", "assert True")

_ALLOWED_LEAVES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    datetime.date,
    datetime.time,
    datetime.timedelta,
    datetime.timezone,
    enum.Enum,
)


def assert_no_mutable_values(value, path="snapshot"):
    """Walk everything reachable; fail on any mutable value."""

    if isinstance(value, _ALLOWED_LEAVES):
        return

    if isinstance(value, (tuple, frozenset)):
        for index, item in enumerate(value):
            assert_no_mutable_values(item, f"{path}[{index}]")
        return

    if isinstance(value, FrozenMapping):
        for key, item in value.items():
            assert_no_mutable_values(key, f"{path}<key {key!r}>")
            assert_no_mutable_values(item, f"{path}[{key!r}]")
        return

    assert dataclasses.is_dataclass(value) and not isinstance(value, type), (
        f"{path} is mutable: {type(value).__name__}"
    )
    assert value.__dataclass_params__.frozen, (
        f"{path} is a mutable dataclass: {type(value).__name__}"
    )

    for field in dataclasses.fields(value):
        assert_no_mutable_values(getattr(value, field.name), f"{path}.{field.name}")


def test_snapshot_before_execution():
    session = TaskSession(FlyCoderAgent(PLAYGROUND), CodingState(task="Inspect"), max_iterations=7)

    snap = session.snapshot()

    assert isinstance(snap, OrchestrationSnapshot)
    assert snap.started is False and snap.running is False
    assert snap.lifecycle is None and snap.lifecycle_status is None
    assert snap.outcome is None and snap.successful is False
    assert snap.steps == () and snap.progress_history == ()
    assert snap.iterations_used == 0 and snap.remaining_iterations == 7
    assert snap.can_resume is False
    assert snap.next_step == "Run the task."
    assert snap.workspace == str(PLAYGROUND.resolve())
    assert_no_mutable_values(snap)


def test_snapshot_of_a_paused_session_is_complete(tmp_path):
    session = start_session(tmp_path)
    result = session.result

    snap = session.snapshot()

    assert snap.task_id == session.task_id
    assert snap.task == REPAIR_TASK and snap.intent == "repair"
    assert snap.workspace == str(tmp_path.resolve())
    assert snap.max_iterations == session.max_iterations
    assert snap.iterations_used == len(session.steps)
    assert snap.remaining_iterations == session.remaining_iterations
    assert snap.started and not snap.closed

    assert snap.lifecycle_status == TASK_NEEDS_HUMAN
    assert [(t.from_state, t.to_state) for t in snap.lifecycle.transitions] == [
        (t.from_state, t.to_state) for t in result.lifecycle.transitions
    ]
    assert snap.lifecycle.actions == tuple(result.lifecycle.actions)
    assert snap.stop_status == result.status
    assert snap.stopped_reason == result.stopped_reason

    assert snap.outcome == session.outcome
    assert snap.outcome.outcome == OUTCOME_AWAITING_HUMAN
    assert snap.evidence.status == result.completion_evidence.status
    assert snap.evidence.reasons == result.completion_evidence.reasons

    assert [p.iteration for p in snap.progress_history] == [
        p.iteration for p in result.progress_history
    ]
    assert [s.action for s in snap.steps] == [s.result.action for s in session.steps]
    assert snap.steps[0].data == freeze(session.steps[0].result.data)
    assert snap.steps[0].data["files"] == tuple(session.steps[0].result.data["files"])
    assert snap.steps[1].evidence.status == session.steps[1].completion_evidence.status
    assert set(snap.steps[0].details) >= {"observation", "recovery", "confidence"}
    assert snap.recovery_attempts == result.recovery_history.attempts
    assert snap.verification_attempts == ()

    assert snap.pending_intervention == session.intervention_request()
    assert snap.decisions == ()
    assert snap.can_resume is True and snap.human_action_required is True
    assert snap.next_step == session.outcome.next_step
    assert snap.unresolved_issues == session.outcome.unresolved_issues
    assert snap.diagnostics == session.diagnostics
    assert snap.progress_report == result.progress_report.render()
    assert_no_mutable_values(snap)


def test_snapshot_after_completion_failure_and_cancellation(tmp_path):
    completed = inspect_session()
    quietly(completed.run)
    snap = completed.snapshot()

    assert snap.successful and snap.lifecycle_status == TASK_COMPLETED
    assert snap.outcome.outcome == OUTCOME_VERIFIED_COMPLETE
    assert_no_mutable_values(snap)

    failed = failed_session().snapshot()

    assert not failed.successful
    assert failed.outcome.outcome == OUTCOME_EXECUTION_FAILED
    assert failed.error.type_name == "RuntimeError"
    assert failed.steps[-1].is_component_error
    assert failed.component_errors == (failed.steps[-1].component_error,)
    assert_no_mutable_values(failed)

    cancelled = start_session(tmp_path)
    decide(cancelled, DECISION_CANCEL_TASK)
    snap = cancelled.snapshot()

    assert snap.outcome.outcome == OUTCOME_CANCELLED
    assert [d.kind for d in snap.decisions] == [DECISION_CANCEL_TASK]
    assert snap.can_resume is False
    assert_no_mutable_values(snap)


def test_snapshot_of_a_degraded_result(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("evidence unavailable")

    monkeypatch.setattr(autonomous, "evaluate_completion_evidence", broken)
    monkeypatch.setattr(task_session_module, "evaluate_completion_evidence", broken)

    session = TaskSession(StepAgent(("inspect_files", True, {})), make_state())
    quietly(session.run)
    snap = session.snapshot()

    assert snap.outcome is None and snap.evidence is None
    assert snap.successful is False
    assert snap.error is not None
    assert "component error" in snap.next_step
    assert_no_mutable_values(snap)


def test_snapshot_during_a_run_is_marked_running(tmp_path):
    write(tmp_path)
    session = TaskSession(FlyCoderAgent(tmp_path), CodingState(task=REPAIR_TASK))
    snapshots = []

    quietly(session.run, on_step=lambda step: snapshots.append(session.snapshot()))

    assert snapshots and all(s.running for s in snapshots)
    assert not session.snapshot().running

    for snap in snapshots:
        assert_no_mutable_values(snap)


def test_snapshot_types_are_frozen(tmp_path):
    snap = start_session(tmp_path).snapshot()

    for target, name in (
        (snap, "task"),
        (snap.lifecycle, "status"),
        (snap.lifecycle.transitions[0], "to_state"),
        (snap.steps[0], "action"),
        (snap.evidence, "status"),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(target, name, "changed")


def test_snapshot_nested_collections_cannot_be_mutated(tmp_path):
    snap = start_session(tmp_path).snapshot()
    data = snap.steps[0].data

    assert isinstance(snap.steps, tuple)
    assert isinstance(data, FrozenMapping)

    with pytest.raises(AttributeError):
        snap.steps.append(None)

    with pytest.raises(TypeError):
        data["files"] = []

    with pytest.raises(AttributeError):
        data._proxy = {}

    with pytest.raises(AttributeError):
        data["files"].append("x")  # a tuple

    with pytest.raises(TypeError):
        snap.steps[0].details["observation"]["risks"] = ()

    assert isinstance(snap.lifecycle.errors, tuple)
    assert isinstance(snap.evidence.reasons, tuple)


def test_snapshot_rejects_mutable_content_and_unearned_success(tmp_path):
    snap = failed_session().snapshot()

    with pytest.raises(TypeError):
        dataclasses.replace(snap, diagnostics=["mutable"])

    with pytest.raises(ValueError):
        dataclasses.replace(snap, successful=True)


def test_freeze_converts_deeply_nested_structures():
    @dataclasses.dataclass
    class Mutable:
        items: list

    original = {
        "a": [{"b": {1, 2}}, {"c": [1, [2, {"d": []}]]}],
        "e": Mutable(items=[{"f": "g"}]),
        "path": PLAYGROUND,
    }

    frozen = freeze(original)

    assert is_deeply_immutable(frozen)
    assert_no_mutable_values(frozen, "frozen")
    assert frozen["a"][0]["b"] == frozenset({1, 2})
    assert frozen["a"][1]["c"] == (1, (2, FrozenMapping({"d": ()})))
    assert frozen["e"]["__type__"] == "Mutable"
    assert frozen["e"]["items"][0]["f"] == "g"
    assert frozen["path"] == str(PLAYGROUND)

    # The frozen copy shares nothing mutable with the original.
    original["a"][1]["c"][1][1]["d"].append("later")
    original["e"].items.clear()

    assert frozen["a"][1]["c"][1][1]["d"] == ()
    assert frozen["e"]["items"][0]["f"] == "g"


def test_freeze_handles_cycles():
    loop = []
    loop.append(loop)

    frozen = freeze({"loop": loop})

    assert is_deeply_immutable(frozen)


def test_snapshot_is_independent_of_later_session_changes(tmp_path):
    session = start_session(tmp_path)
    snap = session.snapshot()
    steps_before = [s.action for s in snap.steps]

    # Change everything the snapshot was taken from.
    session.state.files.append("injected.py")
    session.steps[0].result.data["files"].append("injected.py")
    approve(session)
    quietly(session.resume)

    assert session.outcome.outcome == OUTCOME_VERIFIED_COMPLETE
    assert snap.outcome.outcome == OUTCOME_AWAITING_HUMAN
    assert snap.lifecycle_status == TASK_NEEDS_HUMAN
    assert [s.action for s in snap.steps] == steps_before
    assert "injected.py" not in snap.steps[0].data["files"]
    assert snap.decisions == ()
    assert len(snap.lifecycle.transitions) < len(session.lifecycle.transitions)


def test_snapshot_does_not_change_the_session(tmp_path):
    session = start_session(tmp_path)
    state_before = copy.deepcopy(session.state)
    outcome_before = session.outcome
    diagnostics_before = session.diagnostics

    first = session.snapshot()
    second = session.snapshot()

    assert first == second
    assert session.state == state_before
    assert session.outcome is outcome_before
    assert session.diagnostics == diagnostics_before


# ------------------------------------------------------------------
# Step 5 — CLI migration through TaskSession
# ------------------------------------------------------------------


def run_cli(tmp_path, argv, *, answer=None, entry=None, content=ORIGINAL):
    write(tmp_path, content)

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


@pytest.fixture
def captured_sessions(monkeypatch):
    """Record every session the CLIs drive through run_cli_session."""

    sessions = []
    original = cli_module.run_cli_session

    def spy(session, **kwargs):
        sessions.append(session)
        return original(session, **kwargs)

    monkeypatch.setattr(cli_module, "run_cli_session", spy)
    return sessions


ENTRIES = pytest.mark.parametrize(
    "entry",
    [cli_module.main, agent_cli],
    ids=["flycoder.cli", "agent.cli"],
)


@ENTRIES
def test_both_clis_use_the_shared_session_driver(entry, tmp_path, captured_sessions):
    run_cli(tmp_path, ["--task", REPAIR_TASK], entry=entry)

    assert len(captured_sessions) == 1

    session = captured_sessions[0]

    assert isinstance(session, TaskSession)
    assert session.started and session.closed
    assert session.close_reason == cli_module.CLI_CLOSE_REASON


def test_flycoder_cli_output_contract(tmp_path):
    output = run_cli(tmp_path, ["--task", REPAIR_TASK])
    lines = output.splitlines()

    assert lines[0] == f"Workspace: {tmp_path.resolve()}"
    assert lines[1] == f"Task: {REPAIR_TASK}"
    assert lines[2] == "Task Intent: repair"
    assert "[Step 0] inspect_files | success=True" in lines
    assert "  Files:" in lines
    assert "    - test_sample.py" in lines
    assert "[Step 1] run_tests | success=False" in lines
    assert "  Test output:" in lines
    assert "[Step 4] propose_repair | success=True" in lines
    assert "Proposed diff:" in output
    assert "Repair requires approval. Re-run with --approve to review and confirm it." in lines
    assert "Final state:" in lines
    assert "  Repair applied: False" in lines
    assert "Outcome: AWAITING_HUMAN — Execution is paused until a human records a decision." in lines
    # No step after the proposal: the session stops at the human boundary.
    assert not any(line.startswith("[Step 5]") for line in lines)


def test_agent_cli_output_contract(tmp_path):
    output = run_cli(tmp_path, ["--task", REPAIR_TASK], entry=agent_cli)
    lines = output.splitlines()

    assert lines[0] == f"Task: {REPAIR_TASK}"
    assert lines[1] == "Task Intent: repair"
    assert "[Step 0] inspect_files | success=True" in lines
    assert any(line.startswith("  data={") for line in lines)
    assert "[Step 4] propose_repair | success=True" in lines
    assert "A repair proposal is waiting for approval." in lines
    assert "Outcome: AWAITING_HUMAN — Execution is paused until a human records a decision." in lines


@ENTRIES
def test_cli_approval_applies_and_verifies(entry, tmp_path, captured_sessions):
    output = run_cli(tmp_path, ["--task", REPAIR_TASK, "--approve"], answer="yes", entry=entry)

    decision_line = (
        "[Decision] APPROVE" if entry is cli_module.main else "[Step 4] decision | APPROVE"
    )

    assert decision_line in output
    assert output.index("Proposed diff:") < output.index(decision_line)
    assert "[Step 5] approve_repair | success=True" in output
    assert "verify_changes | success=True" in output
    assert "Outcome: VERIFIED_COMPLETE" in output
    assert disk(tmp_path) == REPAIRED
    assert captured_sessions[0].outcome.successful


@ENTRIES
def test_cli_rejection_resumes_with_one_read_only_follow_up(entry, tmp_path, captured_sessions):
    output = run_cli(tmp_path, ["--task", REPAIR_TASK, "--approve"], answer="no", entry=entry)
    session = captured_sessions[0]
    rejection = session.decisions[0]

    assert rejection.kind == DECISION_REJECT

    # Exactly one follow-up step after the decision, and it changed nothing.
    decision_index = 4
    follow_up = session.steps[decision_index + 1:]

    assert [s.result.action for s in follow_up] == ["propose_repair"]
    assert follow_up[0].result.success is False
    assert len(session.steps) == decision_index + 2
    assert output.count("[Step 5]") == 1
    assert "[Step 6]" not in output

    # The rejected repair is never offered again; the run stops at
    # no_safe_repair; the protected file is untouched.
    assert rejection.proposal_fingerprint in session.state.rejected_proposals
    assert session.state.repair_proposed is False
    assert session.state.proposal_fingerprint is None
    assert session.state.human_input_reason == HUMAN_INPUT_NO_SAFE_REPAIR
    assert session.outcome.outcome == OUTCOME_AWAITING_HUMAN
    assert "approve_repair" not in output
    assert disk(tmp_path) == ORIGINAL


@ENTRIES
def test_cli_without_confirmation_fails_closed(entry, tmp_path):
    output = run_cli(tmp_path, ["--task", REPAIR_TASK, "--approve"], answer="", entry=entry)

    assert "No decision recorded" in output
    assert "approve_repair" not in output
    assert disk(tmp_path) == ORIGINAL


@ENTRIES
def test_zero_steps_is_a_parser_error(entry, tmp_path, capsys):
    write(tmp_path)

    with pytest.raises(SystemExit) as raised:
        entry([str(tmp_path), "--max-steps", "0"])

    assert raised.value.code == 2
    assert "--max-steps must be at least 1" in capsys.readouterr().err


@ENTRIES
def test_cli_help_lists_the_existing_arguments(entry, capsys):
    with pytest.raises(SystemExit) as raised:
        entry(["--help"])

    assert raised.value.code == 0

    help_text = capsys.readouterr().out

    for option in ("--task", "--max-steps", "--approve", "workspace"):
        assert option in help_text


@ENTRIES
def test_cli_interrupt_prints_the_outcome_and_reraises(entry, tmp_path, monkeypatch, captured_sessions):
    interrupt = KeyboardInterrupt()
    original = FlyCoderAgent.run_once
    calls = {"count": 0}

    def interrupt_second_step(self, state):
        calls["count"] += 1

        if calls["count"] == 2:
            raise interrupt

        return original(self, state)

    monkeypatch.setattr(FlyCoderAgent, "run_once", interrupt_second_step)
    write(tmp_path)
    buffer = io.StringIO()

    with pytest.raises(KeyboardInterrupt) as raised:
        with contextlib.redirect_stdout(buffer):
            entry([str(tmp_path), "--task", REPAIR_TASK], input_fn=input, is_interactive=False)

    assert raised.value is interrupt

    session = captured_sessions[0]

    assert session.closed
    assert_session_failed(session)
    assert session.result.interrupted
    assert "Outcome: EXECUTION_FAILED" in buffer.getvalue()
    assert disk(tmp_path) == ORIGINAL


# ------------------------------------------------------------------
# Step 6 — Pipeline and classification regressions
# ------------------------------------------------------------------


def test_actions_only_execute_inside_run_once(monkeypatch, tmp_path):
    depth = {"value": 0}
    inside, outside = [], []
    original_run_once = FlyCoderAgent.run_once
    original_execute = _Registry.execute

    def run_once(self, state):
        depth["value"] += 1
        try:
            return original_run_once(self, state)
        finally:
            depth["value"] -= 1

    def execute(self, name, **kwargs):
        (inside if depth["value"] else outside).append(name)
        return original_execute(self, name, **kwargs)

    monkeypatch.setattr(FlyCoderAgent, "run_once", run_once)
    monkeypatch.setattr(_Registry, "execute", execute)

    run_cli(tmp_path, ["--task", REPAIR_TASK, "--approve"], answer="yes")
    run_cli(tmp_path, ["--task", REPAIR_TASK], entry=agent_cli)

    session = start_session(tmp_path)
    session.snapshot()
    decide(session, DECISION_CANCEL_TASK)
    session.snapshot()
    session.close()

    quietly(FlyCoderAgent(PLAYGROUND).run_task, "Inspect playground files")

    assert outside == []
    assert {"inspect_files", "run_tests", "propose_repair", "approve_repair", "verify_changes"} <= set(inside)


def test_every_executed_action_passed_the_guardrail(monkeypatch, tmp_path, captured_sessions):
    import flycoder.agent as agent_module

    checked = []
    original = agent_module.check_guardrails

    def spy(result, state):
        decision = original(result, state)
        checked.append((result.action, decision.allowed))
        return decision

    monkeypatch.setattr(agent_module, "check_guardrails", spy)

    run_cli(tmp_path, ["--task", REPAIR_TASK, "--approve"], answer="yes")

    executed = [
        step.result.action
        for step in captured_sessions[0].steps
        if step.result.action != "finish"
    ]

    assert [action for action, _ in checked] == executed
    assert all(allowed for _, allowed in checked)

    for step in captured_sessions[0].steps:
        if step.result.action != "finish":
            assert step.result.data["guardrail"]["allowed"] is True


def test_classifications_are_unchanged_through_the_session_entry_point(tmp_path):
    agent = FlyCoderAgent(PLAYGROUND)

    completed = quietly(agent.start_task, "Inspect playground files")
    budget = quietly(agent.start_task, "Inspect playground files", max_iterations=1)
    failed = failed_session()
    paused = start_session(tmp_path)

    cases = [
        (completed, OUTCOME_VERIFIED_COMPLETE, True),
        (budget, OUTCOME_BUDGET_EXHAUSTED, False),
        (failed, OUTCOME_EXECUTION_FAILED, False),
        (paused, OUTCOME_AWAITING_HUMAN, False),
    ]

    for session, expected, successful in cases:
        assert session.outcome.outcome == expected
        assert session.result.successful is successful
        assert session.snapshot().successful is successful
        assert_consistent(session.result)

    decide(paused, DECISION_CANCEL_TASK)

    assert paused.outcome.outcome == OUTCOME_CANCELLED
    assert_consistent(paused.result)


# ------------------------------------------------------------------
# Backward compatibility of module-level imports
# ------------------------------------------------------------------


def test_flycoder_cli_keeps_its_previous_re_exports():
    from flycoder.cli import (
        DECISION_APPROVE,
        CodingState as cli_coding_state,
        FlyCoderAgent as cli_agent,
        apply_human_decision,
        awaiting_repair_approval,
        build_intervention_request,
        build_parser,
        classify_stopped_run,
        evaluate_completion_evidence,
        main,
        print_repair_proposal,
        prompt_for_approval,
        verification_attempt,
    )
    from flycoder.tools import completion_evidence, completion_outcome
    from flycoder.tools import human_input, human_intervention

    # Each old import path still resolves to the original object.
    assert DECISION_APPROVE == human_intervention.DECISION_APPROVE
    assert apply_human_decision is human_intervention.apply_human_decision
    assert build_intervention_request is human_intervention.build_intervention_request
    assert prompt_for_approval is human_intervention.prompt_for_approval
    assert awaiting_repair_approval is human_input.awaiting_repair_approval
    assert evaluate_completion_evidence is completion_evidence.evaluate_completion_evidence
    assert classify_stopped_run is completion_outcome.classify_stopped_run
    assert verification_attempt is completion_outcome.verification_attempt
    assert cli_agent is FlyCoderAgent
    assert cli_coding_state is CodingState
    assert callable(build_parser) and callable(main) and callable(print_repair_proposal)


def test_flycoder_agent_keeps_run_autonomous_task_re_export():
    from flycoder.agent import AutonomousTaskResult, run_autonomous_task as agent_loop

    assert agent_loop is autonomous.run_autonomous_task
    assert AutonomousTaskResult is autonomous.AutonomousTaskResult

    # The old path still works as an entry point into the same loop.
    result = quietly(
        agent_loop,
        FlyCoderAgent(PLAYGROUND),
        CodingState(task="Inspect playground files"),
    )

    assert result.status == AUTONOMOUS_COMPLETED
    assert result.successful
