"""Tests for Phase 10.5 autonomous verification and test freshness."""

import contextlib
import copy
import io

import pytest

from flycoder.actions import create_action_registry
from flycoder.actions.registry import ActionResult
from flycoder.actions.verification_actions import verify_changes_action
from flycoder.agent import FlyCoderAgent
from flycoder.state import CodingState
from flycoder.tools.autonomous import (
    AUTONOMOUS_COMPLETED,
    AUTONOMOUS_MAX_ITERATIONS,
    AUTONOMOUS_NEEDS_HUMAN,
    run_autonomous_task,
)
from flycoder.tools.completion_evidence import (
    EVIDENCE_FAILED,
    EVIDENCE_INCONSISTENT,
    EVIDENCE_MISSING,
    EVIDENCE_STALE,
    EVIDENCE_VERIFIED,
    METHOD_VERIFICATION_PIPELINE,
    VerificationRecord,
    evaluate_completion_evidence,
    verification_is_current,
)
from flycoder.tools.filesystem import Workspace
from flycoder.tools import freshness
from flycoder.tools.freshness import (
    freshness_violations,
    record_code_change,
    record_test_run,
)
from flycoder.tools.guardrails import check_guardrails
from flycoder.tools.recovery import recover_from_failure
from flycoder.tools.task_lifecycle import (
    TASK_COMPLETED,
    TASK_RUNNING,
    TASK_VERIFYING,
)
from flycoder.tools.task_progress import TESTS_PASSED, TESTS_RUN, evaluate_milestones


ORIGINAL = (
    "def test_sample():\n"
    '    assert False, "Intentional failure for testing"\n'
)
REPAIRED = ORIGINAL.replace("assert False", "assert True")

# The automatic repair fixes only the first test; the second keeps
# failing, so verification of the applied change fails.
PARTIALLY_FIXABLE = ORIGINAL + "\n\ndef test_other():\n    assert 1 == 2\n"


def quietly(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def run(registry, workspace, state, action):
    return registry.execute(action, workspace=workspace, state=state)


def applied_repair(tmp_path, content=ORIGINAL):
    """Reach an explicitly approved, applied repair with real actions.

    The approve_repair call stands in for the human approval boundary.
    """

    (tmp_path / "test_sample.py").write_text(content)

    workspace = Workspace(tmp_path)
    registry = create_action_registry()
    state = CodingState(task="Fix the failing test", task_intent="repair")

    for action in (
        "inspect_files",
        "run_tests",
        "fix_error",
        "read_file",
        "propose_repair",
        "approve_repair",
    ):
        result = run(registry, workspace, state, action)
        assert result.action == action

    assert state.repair_applied
    return workspace, registry, state


def transitions(result):
    return [(t.from_state, t.to_state) for t in result.lifecycle.transitions]


# ------------------------------------------------------------------
# Freshness primitives
# ------------------------------------------------------------------


def test_record_code_change_invalidates_tests():
    state = CodingState(task="x", tests_run=True, tests_passed=True)

    record_code_change(state)

    assert state.code_version == 1
    assert state.tests_run is False
    assert state.tests_passed is False


def test_record_test_run_stamps_current_version():
    state = CodingState(task="x", code_version=3, error_inspected=True)

    record_test_run(state, passed=False, output="1 failed")

    assert state.tests_run and not state.tests_passed
    assert state.tests_code_version == 3
    assert state.last_error == "1 failed"
    assert state.error_inspected is False

    record_test_run(state, passed=True, output="ok")

    assert state.tests_passed
    assert state.last_error is None


def test_failed_run_without_output_still_records_error():
    state = CodingState(task="x")

    record_test_run(state, passed=False, output="")

    assert state.last_error == "Tests failed."


def test_unversioned_results_describe_version_zero():
    state = CodingState(task="x", tests_run=True, tests_passed=True)

    assert freshness.tested_code_version(state) == 0
    assert freshness.tests_current(state)

    state.code_version = 1

    assert freshness.tests_stale(state)
    assert not freshness.tests_current(state)


def test_freshness_violations():
    assert freshness_violations(CodingState(task="x")) == []
    assert freshness_violations(
        CodingState(task="x", code_version=1, tests_code_version=2)
    ) == ["tests_code_version cannot be newer than code_version."]
    assert freshness_violations(CodingState(task="x", code_version=-1))


# ------------------------------------------------------------------
# Every code-changing action invalidates; others preserve
# ------------------------------------------------------------------


def make_tested_state(**overrides):
    values = {
        "task": "x",
        "tests_run": True,
        "tests_passed": True,
        "tests_code_version": 0,
    }
    values.update(overrides)
    return CodingState(**values)


def test_write_code_invalidates_test_results(tmp_path):
    state = make_tested_state()

    result = create_action_registry().execute(
        "write_code",
        workspace=Workspace(tmp_path),
        state=state,
        file_path="module.py",
        content="x = 1\n",
    )

    assert result.success
    assert (tmp_path / "module.py").read_text() == "x = 1\n"
    assert state.code_version == 1
    assert not state.tests_run and not state.tests_passed


def test_failed_write_code_preserves_test_results(tmp_path):
    state = make_tested_state()
    before = copy.deepcopy(state)

    result = create_action_registry().execute(
        "write_code",
        workspace=Workspace(tmp_path),
        state=state,
        file_path="../outside.py",
        content="x = 1\n",
    )

    assert result.success is False
    assert state == before


def test_read_only_actions_preserve_test_results(tmp_path):
    (tmp_path / "module.py").write_text("x = 1\n")
    workspace = Workspace(tmp_path)
    registry = create_action_registry()
    state = make_tested_state()

    run(registry, workspace, state, "inspect_files")
    state.current_file = "module.py"
    run(registry, workspace, state, "read_file")

    assert state.code_version == 0
    assert freshness.tests_current(state) and state.tests_passed


def test_run_tests_action_stamps_version(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    state = CodingState(task="x", code_version=2)

    run(create_action_registry(), Workspace(tmp_path), state, "run_tests")

    assert state.tests_code_version == 2
    assert freshness.tests_current(state) and state.tests_passed


def test_approval_changes_code_but_does_not_verify(tmp_path):
    _, _, state = applied_repair(tmp_path)

    assert state.code_version == 1
    assert state.tests_run is False
    assert evaluate_completion_evidence(state).status == EVIDENCE_MISSING


def test_rejected_stale_approval_preserves_version_and_tests(tmp_path):
    (tmp_path / "test_sample.py").write_text(ORIGINAL)
    workspace = Workspace(tmp_path)
    registry = create_action_registry()
    state = CodingState(task="Fix the failing test", task_intent="repair")

    for action in ("inspect_files", "run_tests", "fix_error", "read_file"):
        run(registry, workspace, state, action)

    assert run(registry, workspace, state, "propose_repair").success

    (tmp_path / "test_sample.py").write_text(ORIGINAL + "# edited\n")
    version, tests = state.code_version, (state.tests_run, state.tests_passed)

    result = run(registry, workspace, state, "approve_repair")

    assert result.success is False
    assert state.code_version == version
    assert (state.tests_run, state.tests_passed) == tests


def test_rollback_bumps_version_and_rejected_rollback_does_not(tmp_path):
    workspace, registry, state = applied_repair(tmp_path)

    (tmp_path / "test_sample.py").write_text("changed = True\n")
    assert run(registry, workspace, state, "rollback_repair").success is False
    assert state.code_version == 1

    (tmp_path / "test_sample.py").write_text(REPAIRED)
    assert run(registry, workspace, state, "rollback_repair").success
    assert state.code_version == 2
    assert state.tests_run is False


# ------------------------------------------------------------------
# Evidence and progress honour freshness
# ------------------------------------------------------------------


def test_stale_state_tests_are_stale_evidence_and_pending_progress():
    state = CodingState(
        task="Run the tests",
        task_intent="test",
        files=["test_a.py"],
        tests_run=True,
        tests_passed=True,
        tests_code_version=0,
        code_version=1,
    )

    evidence = evaluate_completion_evidence(state)
    milestones = {m.key: m for m in evaluate_milestones(state)}

    assert evidence.status == EVIDENCE_STALE
    assert not evidence.verified
    assert evidence.tests_run is False
    assert "stale" in evidence.reasons[0]
    assert milestones[TESTS_RUN].pending
    assert milestones[TESTS_PASSED].pending


def test_future_test_version_is_inconsistent():
    state = CodingState(
        task="Run the tests",
        task_intent="test",
        tests_run=True,
        tests_passed=True,
        tests_code_version=5,
        code_version=1,
    )

    assert evaluate_completion_evidence(state).status == EVIDENCE_INCONSISTENT


def test_evidence_uses_state_verification_record_by_default(tmp_path):
    workspace, registry, state = applied_repair(tmp_path)

    assert quietly(verify_changes_action, workspace, state).success

    evidence = evaluate_completion_evidence(state)

    assert evidence.verified
    assert evidence.verification is state.verification_record
    assert evidence.verification_method == METHOD_VERIFICATION_PIPELINE
    assert evidence.verification_signature == evidence.code_signature


def test_wrong_state_record_type_rejected():
    state = CodingState(task="x", task_intent="test", verification_record="ok")

    with pytest.raises(TypeError):
        evaluate_completion_evidence(state)

    assert verification_is_current(state) is False


# ------------------------------------------------------------------
# verify_changes action
# ------------------------------------------------------------------


def test_verify_changes_passes_for_correct_repair(tmp_path):
    workspace, _, state = applied_repair(tmp_path)

    result = verify_changes_action(workspace, state)

    assert result.success
    assert result.data["targeted_tests"] == ["test_sample.py"]
    assert result.data["verification"]["verified"] is True
    assert isinstance(state.verification_record, VerificationRecord)
    assert verification_is_current(state)
    assert state.tests_run and state.tests_passed
    assert state.tests_code_version == state.code_version == 1
    assert state.last_error is None
    # Verification never modifies files.
    assert (tmp_path / "test_sample.py").read_text() == REPAIRED


def test_verify_changes_fails_when_tests_still_fail(tmp_path):
    workspace, _, state = applied_repair(tmp_path, PARTIALLY_FIXABLE)

    result = verify_changes_action(workspace, state)

    assert result.success is False
    assert result.data["verification"]["targeted_passed"] is False
    assert state.tests_run and not state.tests_passed
    assert state.tests_code_version == state.code_version
    assert state.last_error.startswith("Verification failed:")
    assert "test_other" in state.last_error
    assert evaluate_completion_evidence(state).status == EVIDENCE_FAILED


def test_verify_changes_change_check_failure_records_no_test_run(tmp_path):
    workspace, _, state = applied_repair(tmp_path)

    # The file on disk no longer matches the applied proposal.
    (tmp_path / "test_sample.py").write_text(REPAIRED + "# drift\n")

    result = verify_changes_action(workspace, state)

    assert result.success is False
    assert result.data["verification"]["change_verified"] is False
    assert state.tests_run is False
    assert "change verification failed" in state.last_error
    assert evaluate_completion_evidence(state).status == EVIDENCE_FAILED


def test_verify_changes_requires_applied_change(tmp_path):
    state = CodingState(task="x")

    result = verify_changes_action(Workspace(tmp_path), state)

    assert result.success is False
    assert state.verification_record is None
    assert state.tests_run is False


def test_guardrail_blocks_verification_without_applied_change():
    decision = check_guardrails(
        ActionResult(action="verify_changes", success=True, message=""),
        CodingState(task="x"),
    )

    assert decision.allowed is False
    assert "verify_changes requires an applied code change." in (
        decision.violations
    )


def test_guardrail_allows_verification_of_approved_change():
    state = CodingState(
        task="x",
        current_file="a.py",
        current_file_content="x",
        proposed_content="x",
        repair_original_content="y",
        repair_description="d",
        repair_proposed=True,
        repair_approved=True,
        repair_applied=True,
    )

    decision = check_guardrails(
        ActionResult(action="verify_changes", success=True, message=""),
        state,
    )

    assert decision.allowed
    assert decision.requires_human_approval is False


def test_verification_failure_recovery_only_inspects():
    failure = ActionResult(action="verify_changes", success=False, message="x")

    with_error = recover_from_failure(failure, CodingState(task="x", last_error="e"))
    without_error = recover_from_failure(failure, CodingState(task="x"))

    assert with_error.recovery_action == "fix_error"
    assert with_error.retry_allowed is False
    assert without_error.recovery_action is None


# ------------------------------------------------------------------
# Agent decision
# ------------------------------------------------------------------


def test_agent_verifies_applied_change_once(tmp_path):
    workspace, _, state = applied_repair(tmp_path)
    agent = FlyCoderAgent(tmp_path)

    assert agent.decide(state) == "verify_changes"

    quietly(agent.run_once, state)

    assert verification_is_current(state)
    assert agent.decide(state) == "finish"


def test_stale_record_triggers_reverification(tmp_path):
    workspace, _, state = applied_repair(tmp_path)
    agent = FlyCoderAgent(tmp_path)

    quietly(agent.run_once, state)

    state.current_file_content = REPAIRED + "# re-read\n"

    assert not verification_is_current(state)
    assert agent.decide(state) == "verify_changes"


def test_read_only_and_unchanged_tasks_never_verify(tmp_path):
    (tmp_path / "module.py").write_text("def f():\n    return 1\n")
    (tmp_path / "test_module.py").write_text("def test_f():\n    assert True\n")

    for task in ("Inspect the project", "Review the code", "Run the tests"):
        result = quietly(
            run_autonomous_task,
            FlyCoderAgent(tmp_path),
            CodingState(task=task),
            max_iterations=15,
        )

        actions = [step.result.action for step in result.steps]

        assert "verify_changes" not in actions
        assert result.status == AUTONOMOUS_COMPLETED


# ------------------------------------------------------------------
# Autonomous loop
# ------------------------------------------------------------------


def test_loop_verifies_resumed_repair_and_completes(tmp_path):
    _, _, state = applied_repair(tmp_path)

    result = quietly(
        run_autonomous_task,
        FlyCoderAgent(tmp_path),
        state,
        max_iterations=10,
    )

    assert [s.result.action for s in result.steps] == [
        "verify_changes",
        "finish",
    ]
    assert result.status == AUTONOMOUS_COMPLETED
    assert transitions(result)[2:] == [
        (TASK_RUNNING, TASK_VERIFYING),
        (TASK_VERIFYING, TASK_RUNNING),
        (TASK_RUNNING, TASK_VERIFYING),
        (TASK_VERIFYING, TASK_COMPLETED),
    ]
    assert result.steps[0].completion_evidence.status == EVIDENCE_VERIFIED
    assert result.progress_report.verified
    assert result.progress_report.consistent


def test_loop_failed_verification_returns_to_execution_then_human(tmp_path):
    _, _, state = applied_repair(tmp_path, PARTIALLY_FIXABLE)
    on_disk = (tmp_path / "test_sample.py").read_text()

    result = quietly(
        run_autonomous_task,
        FlyCoderAgent(tmp_path),
        state,
        max_iterations=10,
    )

    actions = [s.result.action for s in result.steps]

    assert actions == ["verify_changes", "fix_error"]
    assert actions.count("verify_changes") == 1
    assert result.status == AUTONOMOUS_NEEDS_HUMAN
    assert (TASK_VERIFYING, TASK_RUNNING) in transitions(result)
    assert "Verification failed" in result.lifecycle.transitions[3].reason
    assert result.completion_evidence.status == EVIDENCE_FAILED

    # No automatic repair: nothing new was proposed or written.
    assert (tmp_path / "test_sample.py").read_text() == on_disk
    assert state.code_version == 1

    report = result.progress_report

    assert report.verified is False
    assert report.human_intervention_required
    assert "Verification failed" in report.render()


def test_loop_respects_iteration_limit_during_verification(tmp_path):
    _, _, state = applied_repair(tmp_path)

    result = quietly(
        run_autonomous_task,
        FlyCoderAgent(tmp_path),
        state,
        max_iterations=1,
    )

    # Evidence is VERIFIED after verify_changes, but the agent never
    # finished, so the task is not reported as completed.
    assert result.status == AUTONOMOUS_MAX_ITERATIONS
    assert result.completion_evidence.verified
    assert result.lifecycle.status != TASK_COMPLETED
    assert result.progress_report.inconsistencies


def test_loop_never_approves_while_verifying(tmp_path):
    _, _, state = applied_repair(tmp_path, PARTIALLY_FIXABLE)

    agent = FlyCoderAgent(tmp_path)
    executed = []
    agent_execute = agent.actions.execute

    def tracking_execute(name, **kwargs):
        executed.append(name)
        return agent_execute(name, **kwargs)

    agent.actions.execute = tracking_execute

    quietly(run_autonomous_task, agent, state, max_iterations=10)

    assert executed == ["verify_changes", "fix_error"]
    assert "approve_repair" not in executed
    assert "propose_repair" not in executed
    assert "write_code" not in executed
