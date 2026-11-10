"""Tests for Phase 10.6: the autonomous recovery loop."""

import contextlib
import copy
import io

import pytest

from flycoder.actions import create_action_registry
from flycoder.actions.error_actions import fix_error_action
from flycoder.actions.registry import ActionResult
from flycoder.agent import FlyCoderAgent
from flycoder.state import CodingState
from flycoder.tools.autonomous import (
    AUTONOMOUS_COMPLETED,
    AUTONOMOUS_MAX_ITERATIONS,
    AUTONOMOUS_NEEDS_HUMAN,
    run_autonomous_task,
)
from flycoder.tools.completion_evidence import (
    EVIDENCE_NEEDS_APPROVAL,
    evaluate_completion_evidence,
    record_verification,
    verification_is_current,
)
from flycoder.tools.filesystem import Workspace
from flycoder.tools.guardrails import check_guardrails
from flycoder.tools.human_input import (
    HUMAN_INPUT_APPLIED_REPAIR_FAILED,
    HUMAN_INPUT_INSUFFICIENT_DIAGNOSIS,
    HUMAN_INPUT_NO_SAFE_REPAIR,
    HUMAN_INPUT_REPAIR_APPROVAL,
    HUMAN_INPUT_STALE_PROPOSAL,
    awaiting_repair_approval,
    clear_human_input,
    describe_human_input,
    human_input_violations,
    request_human_input,
)
from flycoder.tools.iteration_decision import decide_iteration
from flycoder.tools.recovery import (
    RECOVERY_CONFIDENCE_HIGH,
    RECOVERY_CONFIDENCE_LOW,
    RECOVERY_CONFIDENCE_MEDIUM,
    RECOVERY_HUMAN_REQUIRED,
    RECOVERY_INVESTIGATE,
    RECOVERY_NONE,
    RECOVERY_PROPOSE_REPAIR,
    RECOVERY_RETRY,
    RECOVERY_UNRESOLVABLE,
    RecoveryHistory,
    recover_from_failure,
    repair_evidence_sufficient,
)
from flycoder.tools.verification_pipeline import VerificationPipelineResult
from flycoder.tools.verification import ChangeVerificationResult
from flycoder.tools.proposals import record_human_approval


ORIGINAL = (
    "def test_sample():\n"
    '    assert False, "Intentional failure for testing"\n'
)
REPAIRED = ORIGINAL.replace("assert False", "assert True")

# A failing test with no safe automatic repair pattern.
UNREPAIRABLE = "def test_sample():\n    assert 1 == 2\n"

# The automatic repair fixes the first test only.
PARTIALLY_FIXABLE = ORIGINAL + "\n\ndef test_other():\n    assert 1 == 2\n"


def quietly(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def run_task(workspace_path, state, **kwargs):
    return quietly(
        run_autonomous_task,
        FlyCoderAgent(workspace_path),
        state,
        **kwargs,
    )


def failed(action, message="failed"):
    return ActionResult(action=action, success=False, message=message)


def actions_of(result):
    return [step.result.action for step in result.steps]


def diagnosed_state(tmp_path, content=ORIGINAL):
    """Run real actions up to and including diagnosis."""

    (tmp_path / "test_sample.py").write_text(content)

    workspace = Workspace(tmp_path)
    registry = create_action_registry()
    state = CodingState(task="Fix the failing test", task_intent="repair")

    for action in ("inspect_files", "run_tests", "fix_error"):
        registry.execute(action, workspace=workspace, state=state)

    return workspace, registry, state


class ScriptedAgent:
    """Agent stand-in returning scripted results and state changes."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = 0

    @staticmethod
    def classify_task(task):
        return FlyCoderAgent.classify_task(task)

    def run_once(self, state):
        result, changes = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1

        for name, value in changes(self.calls).items():
            setattr(state, name, value)

        return result


# ------------------------------------------------------------------
# Human input reasons
# ------------------------------------------------------------------


def test_request_and_clear_human_input():
    state = CodingState(task="x", repair_proposed=True)

    request_human_input(state, HUMAN_INPUT_REPAIR_APPROVAL)

    assert state.user_input_needed
    assert awaiting_repair_approval(state)
    assert "approval" in describe_human_input(state)

    clear_human_input(state)

    assert not state.user_input_needed
    assert state.human_input_reason is None
    assert describe_human_input(state) is None


def test_unknown_human_input_reason_rejected():
    state = CodingState(task="x")

    with pytest.raises(ValueError):
        request_human_input(state, "because")

    assert state.user_input_needed is False


def test_legacy_flag_without_reason_is_still_described():
    state = CodingState(task="x", user_input_needed=True)

    assert describe_human_input(state).startswith("Human input is required")
    assert not awaiting_repair_approval(state)
    assert human_input_violations(state) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"human_input_reason": "because", "user_input_needed": True},
        {"human_input_reason": HUMAN_INPUT_STALE_PROPOSAL},
        {
            "human_input_reason": HUMAN_INPUT_REPAIR_APPROVAL,
            "user_input_needed": True,
        },
    ],
)
def test_inconsistent_human_input_is_a_guardrail_violation(overrides):
    state = CodingState(task="x", **overrides)

    decision = check_guardrails(
        ActionResult(action="inspect_files", success=True, message=""),
        state,
    )

    assert human_input_violations(state)
    assert decision.allowed is False


# ------------------------------------------------------------------
# fix_error
# ------------------------------------------------------------------


def test_fix_error_without_recorded_error_is_controlled(tmp_path):
    state = CodingState(task="x")
    before = copy.deepcopy(state)

    result = fix_error_action(Workspace(tmp_path), state)

    assert result.success is False
    assert result.message == "No recorded error is available to diagnose."
    assert result.data["error"] is None
    assert state == before  # nothing claims a diagnosis happened


def test_fix_error_without_error_through_registry(tmp_path):
    result = create_action_registry().execute(
        "fix_error",
        workspace=Workspace(tmp_path),
        state=CodingState(task="x"),
    )

    assert result.success is False
    assert "not associated with a value" not in result.message


def test_successful_diagnosis_continues_without_human_input(tmp_path):
    _, _, state = diagnosed_state(tmp_path)

    assert state.error_inspected
    assert state.current_file == "test_sample.py"
    assert state.current_file_content is None
    assert state.user_input_needed is False
    assert state.human_input_reason is None
    assert state.repair_proposed is False


def test_diagnosis_without_failing_file_requires_human(tmp_path):
    state = CodingState(
        task="Fix the failing test",
        task_intent="repair",
        last_error="ImportError: something went wrong",
    )

    result = fix_error_action(Workspace(tmp_path), state)

    assert result.success is False
    assert state.error_inspected
    assert state.current_file is None
    assert state.human_input_reason == HUMAN_INPUT_INSUFFICIENT_DIAGNOSIS


def test_diagnosed_path_outside_workspace_is_insufficient(tmp_path):
    state = CodingState(
        task="Fix the failing test",
        task_intent="repair",
        last_error="missing_module.py:3: AssertionError",
    )

    fix_error_action(Workspace(tmp_path), state)

    assert state.current_file is None
    assert state.human_input_reason == HUMAN_INPUT_INSUFFICIENT_DIAGNOSIS


# ------------------------------------------------------------------
# Safe diagnostic continuation
# ------------------------------------------------------------------


@pytest.mark.parametrize("action", ["read_file", "inspect_files"])
def test_read_only_actions_allowed_after_diagnosis(tmp_path, action):
    _, _, state = diagnosed_state(tmp_path)

    decision = check_guardrails(
        ActionResult(action=action, success=True, message=""),
        state,
    )

    assert decision.allowed


def test_agent_reads_failing_file_after_diagnosis(tmp_path):
    _, _, state = diagnosed_state(tmp_path)
    agent = FlyCoderAgent(tmp_path)

    assert agent.decide(state) == "read_file"

    result = quietly(agent.run_once, state)

    assert result.success
    assert state.current_file_content == ORIGINAL
    assert repair_evidence_sufficient(state)
    assert agent.decide(state) == "propose_repair"


def test_repair_run_prepares_proposal_and_stops_for_approval(tmp_path):
    (tmp_path / "test_sample.py").write_text(ORIGINAL)
    state = CodingState(task="Fix the failing test")

    result = run_task(tmp_path, state, max_iterations=10)

    assert actions_of(result) == [
        "inspect_files",
        "run_tests",
        "fix_error",
        "read_file",
        "propose_repair",
    ]
    assert result.status == AUTONOMOUS_NEEDS_HUMAN
    assert awaiting_repair_approval(state)
    assert result.completion_evidence.status == EVIDENCE_NEEDS_APPROVAL


# ------------------------------------------------------------------
# Repair proposals
# ------------------------------------------------------------------


def test_proposal_preserves_content_and_does_not_modify_files(tmp_path):
    (tmp_path / "test_sample.py").write_text(ORIGINAL)
    state = CodingState(task="Fix the failing test")

    run_task(tmp_path, state, max_iterations=10)

    assert (tmp_path / "test_sample.py").read_text() == ORIGINAL
    assert state.current_file_content == ORIGINAL
    assert state.proposed_content == REPAIRED
    assert state.repair_proposed
    assert state.repair_approved is False
    assert state.repair_applied is False
    assert state.code_version == 0
    assert not evaluate_completion_evidence(state).verified


def test_insufficient_evidence_never_manufactures_a_proposal(tmp_path):
    (tmp_path / "test_sample.py").write_text(UNREPAIRABLE)
    state = CodingState(task="Fix the failing test")

    result = run_task(tmp_path, state, max_iterations=10)

    assert actions_of(result)[-1] == "propose_repair"
    assert result.steps[-1].result.success is False
    assert result.status == AUTONOMOUS_NEEDS_HUMAN
    assert state.repair_proposed is False
    assert state.proposed_content is None
    assert state.human_input_reason == HUMAN_INPUT_NO_SAFE_REPAIR
    assert result.steps[-1].recovery.category == RECOVERY_HUMAN_REQUIRED
    assert (tmp_path / "test_sample.py").read_text() == UNREPAIRABLE


def test_no_safe_repair_does_not_loop_outside_the_autonomous_loop(tmp_path):
    # The CLI drives run_once directly without loop detection.
    _, _, state = diagnosed_state(tmp_path, UNREPAIRABLE)
    agent = FlyCoderAgent(tmp_path)
    executed = []

    for _ in range(10):
        if state.finished:
            break
        executed.append(quietly(agent.run_once, state).action)

    assert executed == ["read_file", "propose_repair", "finish"]


@pytest.mark.parametrize(
    "action",
    ["approve_repair", "write_code", "rollback_repair"],
)
def test_applying_a_pending_proposal_stays_blocked(tmp_path, action):
    (tmp_path / "test_sample.py").write_text(ORIGINAL)
    state = CodingState(task="Fix the failing test")
    run_task(tmp_path, state, max_iterations=10)

    decision = check_guardrails(
        ActionResult(action=action, success=True, message=""),
        state,
    )

    assert decision.allowed is False
    assert decision.requires_human_approval


def test_autonomous_loop_never_approves_or_applies(tmp_path):
    (tmp_path / "test_sample.py").write_text(ORIGINAL)
    agent = FlyCoderAgent(tmp_path)
    executed = []
    original = agent.actions.execute

    def tracking(name, **kwargs):
        executed.append(name)
        return original(name, **kwargs)

    agent.actions.execute = tracking

    quietly(
        run_autonomous_task,
        agent,
        CodingState(task="Fix the failing test"),
        max_iterations=20,
    )

    assert "approve_repair" not in executed
    assert "write_code" not in executed
    assert "rollback_repair" not in executed


# ------------------------------------------------------------------
# Recovery decisions
# ------------------------------------------------------------------


def test_ordinary_read_failure_recovers_by_investigation(tmp_path):
    state = CodingState(task="x", files=["a.py"], current_file="missing.py")
    result = create_action_registry().execute(
        "read_file",
        workspace=Workspace(tmp_path),
        state=state,
    )

    decision = recover_from_failure(result, state)

    assert result.success is False
    assert decision.category == RECOVERY_INVESTIGATE
    assert decision.recovery_action == "inspect_files"
    assert decision.failure_type == "read_failure"
    assert decision.reason


@pytest.mark.parametrize(
    "result, action",
    [
        (failed("inspect_files"), "inspect_files"),
        (failed("run_tests", "Command timed out after 60 seconds"), "run_tests"),
    ],
)
def test_safe_retry_for_read_only_transient_failures(result, action):
    decision = recover_from_failure(result, CodingState(task="x"))

    assert decision.category == RECOVERY_RETRY
    assert decision.retry_allowed is True
    assert decision.recovery_action == action


def test_recovery_categories_cover_outcomes():
    state = CodingState(task="x")

    assert recover_from_failure(
        ActionResult(action="run_tests", success=True, message=""), state
    ).category == RECOVERY_NONE
    assert recover_from_failure(failed("write_code"), state).category == (
        RECOVERY_UNRESOLVABLE
    )
    assert recover_from_failure(failed("other"), state).category == (
        RECOVERY_UNRESOLVABLE
    )
    assert recover_from_failure(failed("propose_repair"), state).category == (
        RECOVERY_HUMAN_REQUIRED
    )

    waiting = CodingState(task="x", user_input_needed=True)

    assert recover_from_failure(failed("run_tests"), waiting).category == (
        RECOVERY_HUMAN_REQUIRED
    )


def test_propose_repair_category_when_evidence_is_sufficient():
    state = CodingState(
        task="Fix",
        task_intent="repair",
        last_error="boom",
        error_inspected=True,
        current_file="test_sample.py",
        current_file_content=ORIGINAL,
    )

    decision = recover_from_failure(failed("run_tests"), state)

    assert decision.category == RECOVERY_PROPOSE_REPAIR
    assert decision.recovery_action == "propose_repair"
    assert decision.requires_human_input is False


def test_recovery_does_not_mutate_state():
    state = CodingState(task="x", last_error="boom", error_inspected=True)
    before = copy.deepcopy(state)

    for action in ("run_tests", "read_file", "verify_changes", "unknown"):
        recover_from_failure(failed(action), state)

    assert state == before


# ------------------------------------------------------------------
# Recovery history and loop prevention
# ------------------------------------------------------------------


def record(history, iteration, action="inspect_files", signature="s"):
    result = failed(action)
    decision = recover_from_failure(result, CodingState(task="x"))
    return history.record(iteration, result, decision, signature)


def test_first_failure_has_high_recovery_confidence():
    assessment = record(RecoveryHistory(), 1)

    assert assessment.confidence == RECOVERY_CONFIDENCE_HIGH
    assert assessment.new_information
    assert not assessment.should_escalate


def test_repeat_with_new_information_is_medium():
    history = RecoveryHistory()
    record(history, 1, signature="a")

    assessment = record(history, 2, signature="b")

    assert assessment.attempts == 2
    assert assessment.confidence == RECOVERY_CONFIDENCE_MEDIUM
    assert not assessment.should_escalate


def test_repeated_failure_without_new_information_escalates():
    history = RecoveryHistory()
    record(history, 1)

    assessment = record(history, 2)

    assert assessment.repeated_failure
    assert assessment.confidence == RECOVERY_CONFIDENCE_LOW
    assert assessment.should_escalate
    assert "without any new information" in assessment.reason


def test_repeated_recovery_decision_escalates():
    history = RecoveryHistory()

    for iteration, signature in enumerate("abc", start=1):
        assessment = record(history, iteration, signature=signature)

    assert assessment.repeated_decision
    assert assessment.should_escalate


def test_interleaved_failures_do_not_count_as_repeated_decisions():
    history = RecoveryHistory()
    record(history, 1, "inspect_files", "a")
    record(history, 2, "read_file", "b")
    assessment = record(history, 3, "inspect_files", "c")

    assert not assessment.repeated_decision
    assert assessment.attempts == 2
    assert assessment.confidence == RECOVERY_CONFIDENCE_MEDIUM


def test_history_is_preserved_and_validated():
    history = RecoveryHistory()
    record(history, 1)
    record(history, 2, signature="b")

    assert [a.iteration for a in history.attempts] == [1, 2]

    with pytest.raises(ValueError):
        record(history, 2)

    with pytest.raises(ValueError):
        history.record(
            3,
            ActionResult(action="x", success=True, message=""),
            recover_from_failure(failed("x"), CodingState(task="x")),
            "s",
        )

    assert len(history) == 2


def test_exhausted_recovery_escalates_the_iteration_decision():
    decision = decide_iteration(
        action_success=False,
        recovery_available=True,
        recovery_exhausted=True,
        recovery_reason="repeated without new information",
    )

    assert decision.should_escalate
    assert "repeated without new information" in decision.reason


def unchanged(_):
    return {}


def test_loop_retries_once_then_succeeds():
    agent = ScriptedAgent(
        (failed("inspect_files"), unchanged),
        (
            ActionResult(action="inspect_files", success=True, message=""),
            lambda _: {"files": ["a.py"]},
        ),
        (
            ActionResult(action="finish", success=True, message=""),
            lambda _: {"finished": True},
        ),
    )
    state = CodingState(task="Inspect the project", task_intent="inspect")

    result = quietly(run_autonomous_task, agent, state, max_iterations=10)

    assert result.status == AUTONOMOUS_COMPLETED
    assert result.steps[0].recovery.category == RECOVERY_RETRY
    assert result.steps[0].recovery_assessment.confidence == (
        RECOVERY_CONFIDENCE_HIGH
    )
    assert len(result.recovery_history) == 1


def test_loop_escalates_repeated_uninformative_failure():
    agent = ScriptedAgent((failed("inspect_files"), unchanged))
    state = CodingState(task="Inspect the project", task_intent="inspect")

    result = quietly(run_autonomous_task, agent, state, max_iterations=20)

    assert result.status == AUTONOMOUS_NEEDS_HUMAN
    assert result.iterations == 2
    assert result.steps[-1].recovery_assessment.repeated_failure
    assert result.stopped_reason.startswith("Recovery is exhausted")
    assert len(result.recovery_history) == 2


def test_no_infinite_recovery_loop_even_with_new_information():
    # Each failure changes the state, so it is never "uninformative",
    # but the attempt limit still stops recovery.
    agent = ScriptedAgent(
        (failed("inspect_files"), lambda call: {"last_action_message": str(call),
                                                 "current_file": f"f{call}.py"}),
    )
    state = CodingState(task="Inspect the project", task_intent="inspect")

    result = quietly(run_autonomous_task, agent, state, max_iterations=20)

    assert result.status == AUTONOMOUS_NEEDS_HUMAN
    assert result.iterations == 3
    assert [s.recovery_assessment.confidence for s in result.steps] == [
        RECOVERY_CONFIDENCE_HIGH,
        RECOVERY_CONFIDENCE_MEDIUM,
        RECOVERY_CONFIDENCE_LOW,
    ]


def test_steps_record_phase9_confidence_without_overriding_it():
    agent = ScriptedAgent((failed("inspect_files"), unchanged))
    state = CodingState(task="Inspect the project", task_intent="inspect")

    result = quietly(run_autonomous_task, agent, state, max_iterations=5)

    for step in result.steps:
        assert step.confidence is not None
        assert step.confidence.action == step.result.action


def test_maximum_iterations_respected_during_recovery(tmp_path):
    (tmp_path / "test_sample.py").write_text(ORIGINAL)
    state = CodingState(task="Fix the failing test")

    result = run_task(tmp_path, state, max_iterations=3)

    assert result.status == AUTONOMOUS_MAX_ITERATIONS
    assert actions_of(result) == ["inspect_files", "run_tests", "fix_error"]
    assert state.repair_proposed is False


# ------------------------------------------------------------------
# Verification integration
# ------------------------------------------------------------------


def applied_partial_repair(tmp_path):
    (tmp_path / "test_sample.py").write_text(PARTIALLY_FIXABLE)
    state = CodingState(task="Fix the failing test")

    first = run_task(tmp_path, state, max_iterations=10)
    assert awaiting_repair_approval(state), first.stopped_reason

    # Explicit human approval of the reviewed proposal.
    record_human_approval(state, state.proposal_fingerprint)
    approval = create_action_registry().execute(
        "approve_repair",
        workspace=Workspace(tmp_path),
        state=state,
    )
    assert approval.success

    return state


def test_failed_verification_is_diagnosed_and_escalated(tmp_path):
    state = applied_partial_repair(tmp_path)

    result = run_task(tmp_path, state, max_iterations=10)

    assert actions_of(result) == ["verify_changes", "fix_error"]
    assert result.steps[0].recovery.category == RECOVERY_INVESTIGATE
    assert result.steps[0].recovery.recovery_action == "fix_error"
    assert result.status == AUTONOMOUS_NEEDS_HUMAN
    assert state.human_input_reason == HUMAN_INPUT_APPLIED_REPAIR_FAILED
    assert state.code_version == 1  # no new repair was written


def test_stale_verification_is_not_accepted_and_reverified(tmp_path):
    (tmp_path / "test_sample.py").write_text(ORIGINAL)
    state = CodingState(task="Fix the failing test")
    run_task(tmp_path, state, max_iterations=10)

    record_human_approval(state, state.proposal_fingerprint)
    create_action_registry().execute(
        "approve_repair",
        workspace=Workspace(tmp_path),
        state=state,
    )

    stale = record_verification(
        VerificationPipelineResult(
            change=ChangeVerificationResult(
                verified=False,
                file="test_sample.py",
                reason="old",
            ),
            targeted=None,
            targeted_analysis=None,
            regression=None,
            reason="old code",
        ),
        CodingState(task="old", current_file="test_sample.py",
                    current_file_content=ORIGINAL),
    )
    state.verification_record = stale

    assert not verification_is_current(state)
    assert not evaluate_completion_evidence(state).verified
    assert FlyCoderAgent(tmp_path).decide(state) == "verify_changes"


# ------------------------------------------------------------------
# Guardrails and execution path stay authoritative
# ------------------------------------------------------------------


def test_recovery_recommendation_does_not_bypass_guardrails(tmp_path):
    (tmp_path / "test_sample.py").write_text(ORIGINAL)
    state = CodingState(task="Fix the failing test")
    run_task(tmp_path, state, max_iterations=10)

    decision = recover_from_failure(failed("run_tests"), state)

    assert decision.category == RECOVERY_HUMAN_REQUIRED
    assert decision.recovery_action is None

    blocked = check_guardrails(
        ActionResult(action="propose_repair", success=True, message=""),
        state,
    )

    assert blocked.allowed is False


class TrackingAgent(FlyCoderAgent):
    """Real agent that records whether actions run inside run_once."""

    def __init__(self, workspace):
        super().__init__(workspace)
        self.inside_run_once = False
        self.outside_calls = []
        self.inside_calls = []
        original = self.actions.execute

        def tracking(name, **kwargs):
            target = (
                self.inside_calls if self.inside_run_once else self.outside_calls
            )
            target.append(name)
            return original(name, **kwargs)

        self.actions.execute = tracking

    def run_once(self, state):
        self.inside_run_once = True
        try:
            return super().run_once(state)
        finally:
            self.inside_run_once = False


def test_recovery_runs_only_through_run_once(tmp_path):
    (tmp_path / "test_sample.py").write_text(UNREPAIRABLE)
    agent = TrackingAgent(tmp_path)

    result = quietly(
        run_autonomous_task,
        agent,
        CodingState(task="Fix the failing test"),
        max_iterations=10,
    )

    assert agent.outside_calls == []
    assert agent.inside_calls == [
        step.result.action for step in result.steps
    ]
