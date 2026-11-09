"""Tests for Phase 10.4.6 autonomous loop progress integration."""

import contextlib
import copy
import io
from pathlib import Path

import pytest

from flycoder.actions import create_action_registry
from flycoder.actions.registry import ActionResult
from flycoder.agent import FlyCoderAgent
from flycoder.state import CodingState
from flycoder.tools.autonomous import (
    AUTONOMOUS_COMPLETED,
    AUTONOMOUS_MAX_ITERATIONS,
    AUTONOMOUS_NEEDS_HUMAN,
    AUTONOMOUS_UNVERIFIED,
    _resolve_intent,
    run_autonomous_task,
)
from flycoder.tools.changes import ChangeProposal
from flycoder.tools.completion_evidence import (
    EVIDENCE_FAILED,
    EVIDENCE_INCOMPLETE,
    EVIDENCE_INCONSISTENT,
    EVIDENCE_MISSING,
    EVIDENCE_NEEDS_APPROVAL,
    EVIDENCE_STALE,
    EVIDENCE_VERIFIED,
    METHOD_VERIFICATION_PIPELINE,
    record_verification,
)
from flycoder.tools.filesystem import Workspace
from flycoder.tools.progress import ProgressSnapshot as IterationSnapshot
from flycoder.tools.progress_history import ProgressHistory, ProgressSnapshot
from flycoder.tools.progress_report import ProgressReport
from flycoder.tools.task_lifecycle import (
    TASK_BLOCKED,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_NEEDS_HUMAN,
    TASK_VERIFYING,
)
from flycoder.tools.verification_pipeline import verify_change_pipeline


PLAYGROUND = Path(__file__).parent

ORIGINAL = (
    "def test_sample():\n"
    '    assert False, "Intentional failure for testing"\n'
)


def run_quietly(agent, state, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_autonomous_task(agent, state, **kwargs)


class ScriptedAgent:
    """Agent stand-in that applies scripted state changes per call.

    It exposes the same ``run_once``/``classify_task`` surface the loop
    uses, so loop behaviour can be tested for states the real decision
    engine cannot reach.
    """

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = 0

    @staticmethod
    def classify_task(task):
        return FlyCoderAgent.classify_task(task)

    def run_once(self, state):
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        action, changes = step

        for name, value in changes.items():
            setattr(state, name, value)

        return ActionResult(action=action, success=True, message=action)


def histories_iterations(result):
    return [snapshot.iteration for snapshot in result.progress_history]


# ------------------------------------------------------------------
# History integration
# ------------------------------------------------------------------


def test_inspect_task_records_baseline_and_iteration_snapshots():
    agent = FlyCoderAgent(PLAYGROUND)
    state = CodingState(task="Inspect playground files")

    result = run_quietly(agent, state, max_iterations=5)

    history = result.progress_history

    assert isinstance(history, ProgressHistory)
    assert histories_iterations(result) == list(range(result.iterations + 1))
    assert history.first.iteration == 0
    assert history.first.percent == 0
    assert history.latest.percent == 100
    assert history.regression_count == 0


def test_steps_carry_the_recorded_task_level_snapshots():
    agent = FlyCoderAgent(PLAYGROUND)
    state = CodingState(task="Inspect playground files")

    result = run_quietly(agent, state, max_iterations=5)

    for step, snapshot in zip(result.steps, result.progress_history.snapshots[1:]):
        assert step.progress_snapshot is snapshot
        assert isinstance(step.progress_snapshot, ProgressSnapshot)
        assert not isinstance(step.progress_snapshot, IterationSnapshot)
        assert step.progress_snapshot.iteration == step.iteration


def test_baseline_intent_resolution_does_not_mutate_state():
    agent = FlyCoderAgent(PLAYGROUND)
    state = CodingState(task="Fix the failing test")
    before = copy.deepcopy(state)

    assert _resolve_intent(agent, state) == "repair"
    assert state == before

    state.task_intent = "test"

    assert _resolve_intent(agent, state) == "test"


def test_loop_progress_tracking_does_not_mutate_state():
    # The scripted agent changes nothing, so any change to the state
    # would come from history, evidence, or reporting.
    state = CodingState(
        task="Run the tests",
        task_intent="test",
        files=["test_a.py"],
    )
    before = copy.deepcopy(state)

    result = run_quietly(
        ScriptedAgent(("read_file", {})),
        state,
        max_iterations=10,
    )

    assert state == before
    assert result.status == AUTONOMOUS_NEEDS_HUMAN  # stall escalation
    assert histories_iterations(result) == list(range(result.iterations + 1))


def test_regression_preserved_in_loop_history():
    state = CodingState(
        task="Run the tests",
        task_intent="test",
        files=["test_a.py"],
    )

    agent = ScriptedAgent(
        ("run_tests", {"tests_run": True, "tests_passed": True}),
        ("run_tests", {"tests_passed": False, "last_error": "boom"}),
        ("finish", {"finished": True}),
    )

    result = run_quietly(agent, state, max_iterations=5)

    history = result.progress_history

    assert history.regression_count == 1
    assert history.peak.iteration == 1
    assert history.below_peak
    assert result.progress_report.history.regression_count == 1


def test_iteration_limit_result_has_history_evidence_and_report():
    agent = FlyCoderAgent(PLAYGROUND)
    state = CodingState(task="Inspect playground files")

    result = run_quietly(agent, state, max_iterations=1)

    assert result.status == AUTONOMOUS_MAX_ITERATIONS
    assert histories_iterations(result) == [0, 1]
    assert isinstance(result.progress_report, ProgressReport)

    # inspect_files already satisfies the inspection criteria, but the
    # agent never finished. Both facts are reported, not reconciled.
    assert result.completion_evidence.status == EVIDENCE_VERIFIED
    assert result.lifecycle.status == TASK_FAILED
    assert result.completed is False
    assert (
        "Completion evidence is VERIFIED, but the lifecycle ended as FAILED."
        in result.progress_report.inconsistencies
    )


# ------------------------------------------------------------------
# Completion gated on evidence
# ------------------------------------------------------------------


def test_completed_requires_verified_evidence():
    agent = FlyCoderAgent(PLAYGROUND)
    state = CodingState(task="Inspect playground files")

    result = run_quietly(agent, state, max_iterations=5)

    assert result.status == AUTONOMOUS_COMPLETED
    assert result.completion_evidence.status == EVIDENCE_VERIFIED
    assert result.progress_report.verified
    assert result.progress_report.consistent
    assert result.lifecycle.status == TASK_COMPLETED
    assert "verified evidence" in result.stopped_reason


def test_finished_without_evidence_is_unverified_not_completed():
    state = CodingState(
        task="Run the tests",
        task_intent="test",
        files=["test_a.py"],
    )

    result = run_quietly(
        ScriptedAgent(("finish", {"finished": True})),
        state,
        max_iterations=5,
    )

    assert result.status == AUTONOMOUS_UNVERIFIED
    assert result.unverified and not result.completed
    assert result.completion_evidence.status == EVIDENCE_MISSING
    assert result.lifecycle.status == TASK_FAILED
    assert TASK_VERIFYING in [t.to_state for t in result.lifecycle.transitions]
    assert "MISSING" in result.stopped_reason
    assert result.progress_report.verified is False


def test_finished_with_failed_tests_is_unverified():
    state = CodingState(
        task="Run the tests",
        task_intent="test",
        files=["test_a.py"],
    )

    result = run_quietly(
        ScriptedAgent(
            (
                "finish",
                {
                    "finished": True,
                    "tests_run": True,
                    "tests_passed": False,
                    "last_error": "1 failed",
                },
            )
        ),
        state,
        max_iterations=5,
    )

    assert result.status == AUTONOMOUS_UNVERIFIED
    assert result.completion_evidence.status == EVIDENCE_FAILED


def test_finished_with_pending_approval_escalates():
    state = CodingState(
        task="Fix the failing test",
        task_intent="repair",
        files=["test_a.py"],
        current_file="test_a.py",
        current_file_content=ORIGINAL,
        proposed_content=ORIGINAL.replace("False", "True"),
        repair_description="d",
        error_inspected=True,
        last_error="boom",
        repair_proposed=True,
    )

    result = run_quietly(
        ScriptedAgent(("finish", {"finished": True})),
        state,
        max_iterations=5,
    )

    assert result.status == AUTONOMOUS_NEEDS_HUMAN
    assert result.completion_evidence.status == EVIDENCE_NEEDS_APPROVAL
    assert result.lifecycle.status == TASK_NEEDS_HUMAN
    assert state.repair_approved is False
    assert state.repair_applied is False


def test_already_finished_state_is_gated_too():
    incomplete = CodingState(
        task="Inspect playground files",
        task_intent="inspect",
        finished=True,
    )

    result = run_quietly(FlyCoderAgent(PLAYGROUND), incomplete)

    assert result.status == AUTONOMOUS_UNVERIFIED
    assert result.iterations == 0
    assert result.completion_evidence.status == EVIDENCE_INCOMPLETE
    assert histories_iterations(result) == [0]

    verified = CodingState(
        task="Inspect playground files",
        task_intent="inspect",
        files=["example.py"],
        finished=True,
    )

    result = run_quietly(FlyCoderAgent(PLAYGROUND), verified)

    assert result.status == AUTONOMOUS_COMPLETED
    assert result.iterations == 0


# ------------------------------------------------------------------
# Evidence re-evaluated after each iteration
# ------------------------------------------------------------------


def test_real_repair_run_evidence_sequence(tmp_path):
    (tmp_path / "test_sample.py").write_text(ORIGINAL)

    agent = FlyCoderAgent(tmp_path)
    state = CodingState(task="Fix the failing test")

    result = run_quietly(agent, state, max_iterations=10)

    assert [step.result.action for step in result.steps] == [
        "inspect_files",
        "run_tests",
        "fix_error",
    ]
    assert [step.completion_evidence.status for step in result.steps] == [
        EVIDENCE_MISSING,
        EVIDENCE_FAILED,
        EVIDENCE_FAILED,
    ]
    assert result.status == AUTONOMOUS_NEEDS_HUMAN
    assert result.completion_evidence is result.steps[-1].completion_evidence

    report = result.progress_report

    assert report.human_intervention_required
    assert report.verified is False
    assert report.percent == 50

    fractions = [s.completion_fraction for s in result.progress_history]

    assert fractions == sorted(fractions)


def test_pending_approval_start_reports_needs_approval():
    agent = FlyCoderAgent(PLAYGROUND)
    state = CodingState(
        task="Repair the failing code",
        current_file="example.py",
        current_file_content="print('hello')",
        proposed_content="print('hi')",
        repair_description="d",
        last_error="Test failed.",
        error_inspected=True,
        repair_proposed=True,
        user_input_needed=True,
    )

    result = run_quietly(agent, state, max_iterations=5)

    assert result.status == AUTONOMOUS_NEEDS_HUMAN
    assert result.completion_evidence.status == EVIDENCE_NEEDS_APPROVAL
    assert result.completion_evidence.human_approval_granted is False
    assert state.repair_approved is False
    assert state.repair_applied is False


# ------------------------------------------------------------------
# Human boundary, resumption, and verification records
# ------------------------------------------------------------------


def human_approves_repair(workspace, state):
    """What a human-driven flow does between autonomous runs."""

    registry = create_action_registry()

    for action in ("read_file", "propose_repair", "approve_repair"):
        result = registry.execute(action, workspace=workspace, state=state)
        assert result.success, (action, result.message)


def first_run_to_human_boundary(tmp_path):
    (tmp_path / "test_sample.py").write_text(ORIGINAL)

    agent = FlyCoderAgent(tmp_path)
    state = CodingState(task="Fix the failing test")

    first = run_quietly(agent, state, max_iterations=10)

    assert first.status == AUTONOMOUS_NEEDS_HUMAN

    return agent, state


def test_resumed_run_after_human_approval_verifies_with_pipeline(tmp_path):
    # Phase 10.5: an applied change is verified with the Phase 7
    # pipeline (verify_changes) instead of a plain run_tests.
    agent, state = first_run_to_human_boundary(tmp_path)

    human_approves_repair(agent.workspace, state)

    second = run_quietly(agent, state, max_iterations=10)

    assert [step.result.action for step in second.steps] == [
        "verify_changes",
        "finish",
    ]
    assert second.status == AUTONOMOUS_COMPLETED
    assert (
        second.completion_evidence.verification_method
        == METHOD_VERIFICATION_PIPELINE
    )
    assert second.completion_evidence.verification is state.verification_record
    assert second.completion_evidence.human_approval_granted
    assert second.progress_history.first.next_milestone == "tests_run"


def test_resumed_run_uses_fresh_pipeline_record(tmp_path):
    agent, state = first_run_to_human_boundary(tmp_path)

    human_approves_repair(agent.workspace, state)

    pipeline = verify_change_pipeline(
        agent.workspace,
        ChangeProposal(
            file=state.current_file,
            original_content=state.repair_original_content,
            proposed_content=state.proposed_content,
        ),
        targeted_tests=["test_sample.py"],
    )
    record = record_verification(pipeline, state)

    second = run_quietly(agent, state, max_iterations=10, verification=record)

    assert second.status == AUTONOMOUS_COMPLETED
    assert (
        second.completion_evidence.verification_method
        == METHOD_VERIFICATION_PIPELINE
    )

    # The agent's own verification run is newer than the supplied record
    # and takes precedence; both describe the same code.
    latest = second.completion_evidence.verification

    assert latest is state.verification_record
    assert latest is not record
    assert latest.code_signature == record.code_signature
    assert latest.result.verified


def test_stale_record_is_never_proof(tmp_path):
    (tmp_path / "test_sample.py").write_text(ORIGINAL)

    workspace = Workspace(tmp_path)
    state = CodingState(
        task="Run the tests",
        task_intent="test",
        files=["test_sample.py"],
        current_file="test_sample.py",
        current_file_content=ORIGINAL.replace("False", "True"),
    )

    # A passing pipeline result recorded for code that has since changed.
    (tmp_path / "test_sample.py").write_text(state.current_file_content)
    pipeline = verify_change_pipeline(
        workspace,
        ChangeProposal(
            file="test_sample.py",
            original_content=ORIGINAL,
            proposed_content=state.current_file_content,
        ),
        targeted_tests=["test_sample.py"],
    )
    record = record_verification(pipeline, state)

    assert pipeline.verified

    state.current_file_content = ORIGINAL

    result = run_quietly(
        ScriptedAgent(("finish", {"finished": True})),
        state,
        max_iterations=5,
        verification=record,
    )

    assert result.status == AUTONOMOUS_UNVERIFIED
    assert result.completion_evidence.status == EVIDENCE_STALE
    assert result.progress_history.first.percent < 100


def test_invalid_verification_rejected():
    with pytest.raises(TypeError):
        run_autonomous_task(
            FlyCoderAgent(PLAYGROUND),
            CodingState(task="Inspect"),
            verification={"verified": True},
        )


def test_agent_run_task_exposes_progress_report():
    agent = FlyCoderAgent(PLAYGROUND)

    with contextlib.redirect_stdout(io.StringIO()):
        result = agent.run_task("Inspect playground files", max_iterations=5)

    assert result.completed
    assert result.progress_report.verified
    assert "Status: VERIFIED" in result.progress_report.render()


# ------------------------------------------------------------------
# Phase 10.4.7 audit regressions
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "files, expected_status, expected_evidence",
    [
        ([], AUTONOMOUS_UNVERIFIED, EVIDENCE_INCOMPLETE),
        (["example.py"], AUTONOMOUS_COMPLETED, EVIDENCE_VERIFIED),
    ],
)
def test_already_finished_unclassified_task_does_not_crash(
    files,
    expected_status,
    expected_evidence,
):
    # Regression: the final report used state.task_intent, which is None
    # until the agent's first decide(), and raised ValueError.
    state = CodingState(
        task="Inspect playground files",
        files=list(files),
        finished=True,
    )

    result = run_quietly(FlyCoderAgent(PLAYGROUND), state)

    assert result.status == expected_status
    assert result.completion_evidence.status == expected_evidence
    assert result.progress_report.intent == "inspect"
    assert state.task_intent is None


def test_guardrail_blocked_outcome_has_report():
    # Contradictory state makes the real guardrail block the first action.
    state = CodingState(task="Run the tests", tests_passed=True)

    result = run_quietly(FlyCoderAgent(PLAYGROUND), state, max_iterations=5)

    assert result.blocked
    assert result.iterations == 1
    assert result.lifecycle.status == TASK_BLOCKED
    assert result.completion_evidence.status == EVIDENCE_INCONSISTENT
    assert result.progress_report.verified is False
    assert histories_iterations(result) == [0, 1]
    assert state.files == []  # the blocked action never ran


class FailingAgent:
    """Agent stand-in whose action fails with no recovery path."""

    @staticmethod
    def classify_task(task):
        return FlyCoderAgent.classify_task(task)

    def run_once(self, state):
        return ActionResult(
            action="review_code",
            success=False,
            message="Unexpected analysis failure.",
        )


def test_failed_outcome_without_recovery_has_report():
    state = CodingState(
        task="Review the code",
        task_intent="review",
        files=["a.py"],
    )

    result = run_quietly(FailingAgent(), state, max_iterations=5)

    step = result.steps[-1]

    assert result.failed
    assert result.iterations == 1
    assert step.recovery.recovery_action is None
    assert step.iteration_decision.should_stop
    assert result.lifecycle.status == TASK_FAILED
    assert result.lifecycle.errors == ["Unexpected analysis failure."]
    assert result.completion_evidence.status == EVIDENCE_INCOMPLETE
    assert result.progress_report.verified is False
    assert result.progress_report.lifecycle.error_count == 1
