"""Tests for Phase 10.7: human escalation and task resumption."""

import contextlib
import copy
import io

import pytest

from flycoder import cli as cli_module
from flycoder.actions import create_action_registry
from flycoder.actions.approval_actions import approve_repair_action
from flycoder.actions.registry import ActionResult
from flycoder.agent import FlyCoderAgent, cli as agent_cli
from flycoder.state import CodingState
from flycoder.tools.autonomous import (
    AUTONOMOUS_CANCELLED,
    AUTONOMOUS_COMPLETED,
    AUTONOMOUS_MAX_ITERATIONS,
    AUTONOMOUS_NEEDS_HUMAN,
)
from flycoder.tools.completion_evidence import (
    EVIDENCE_VERIFIED,
    METHOD_VERIFICATION_PIPELINE,
    record_verification,
)
from flycoder.tools.filesystem import Workspace
from flycoder.tools.guardrails import check_guardrails
from flycoder.tools.human_input import (
    HUMAN_INPUT_APPLIED_REPAIR_FAILED,
    HUMAN_INPUT_INSUFFICIENT_DIAGNOSIS,
    HUMAN_INPUT_NO_SAFE_REPAIR,
    HUMAN_INPUT_REPAIR_APPROVAL,
    HUMAN_INPUT_STALE_PROPOSAL,
)
from flycoder.tools.human_intervention import (
    DECISION_APPROVE,
    DECISION_CANCEL_TASK,
    DECISION_PROVIDE_INFORMATION,
    DECISION_REJECT,
    DECISION_REQUEST_CHANGES,
    HumanDecision,
    HumanDecisionError,
    apply_human_decision,
    build_intervention_request,
    prompt_for_approval,
)
from flycoder.tools.proposals import (
    ApprovalRecord,
    approval_problems,
    compute_proposal_fingerprint,
    record_human_approval,
)
from flycoder.tools.recovery import recover_from_failure
from flycoder.tools.task_lifecycle import (
    ALLOWED_TRANSITIONS,
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_NEEDS_HUMAN,
    TASK_RUNNING,
)
from flycoder.tools.task_session import TaskSession
from flycoder.tools.verification import ChangeVerificationResult
from flycoder.tools.verification_pipeline import VerificationPipelineResult


ORIGINAL = (
    "def test_sample():\n"
    '    assert False, "Intentional failure for testing"\n'
)
REPAIRED = ORIGINAL.replace("assert False", "assert True")
UNREPAIRABLE = "def test_sample():\n    assert 1 == 2\n"
PARTIALLY_FIXABLE = ORIGINAL + "\n\ndef test_other():\n    assert 1 == 2\n"

TASK = "Fix the failing test"


def quietly(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def write(tmp_path, content=ORIGINAL):
    (tmp_path / "test_sample.py").write_text(content)


def disk(tmp_path):
    return (tmp_path / "test_sample.py").read_text()


def start(tmp_path, content=ORIGINAL, *, max_iterations=20, state=None):
    write(tmp_path, content)
    return quietly(
        TaskSession.start,
        FlyCoderAgent(tmp_path),
        TASK,
        max_iterations=max_iterations,
        state=state,
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


def resume(session):
    return quietly(session.resume)


def actions(result):
    return [step.result.action for step in result.steps]


def proposal_state(tmp_path):
    """Real actions up to a proposal awaiting approval."""

    write(tmp_path)
    workspace = Workspace(tmp_path)
    registry = create_action_registry()
    state = CodingState(task=TASK, task_intent="repair")

    for action in (
        "inspect_files",
        "run_tests",
        "fix_error",
        "read_file",
        "propose_repair",
    ):
        registry.execute(action, workspace=workspace, state=state)

    assert state.human_input_reason == HUMAN_INPUT_REPAIR_APPROVAL
    return workspace, registry, state


# ------------------------------------------------------------------
# Proposal review and approval
# ------------------------------------------------------------------


def test_diff_is_shown_before_confirmation_is_requested(tmp_path):
    _, _, state = proposal_state(tmp_path)
    request = build_intervention_request(state)
    events = []

    def input_fn(prompt):
        events.append(("input", prompt))
        return "yes"

    decision = prompt_for_approval(
        request,
        input_fn=input_fn,
        output_fn=lambda text: events.append(("output", text)),
        interactive=True,
    )

    shown = "\n".join(text for kind, text in events if kind == "output")
    first_input = next(i for i, (kind, _) in enumerate(events) if kind == "input")

    assert '+    assert True, "Intentional failure for testing"' in shown
    assert "File: test_sample.py" in shown
    assert request.proposal_fingerprint in shown
    assert all(kind == "output" for kind, _ in events[:first_input])
    assert decision.kind == DECISION_APPROVE
    assert decision.confirmed
    assert decision.proposal_fingerprint == request.proposal_fingerprint


def run_cli(tmp_path, *, approve_flag, answer=None, interactive=True, entry=None):
    write(tmp_path)
    prompts = []

    def input_fn(prompt):
        prompts.append(prompt)
        if answer is None:
            raise AssertionError("input must not be requested")
        return answer

    argv = [str(tmp_path), "--task", TASK]
    if approve_flag:
        argv.append("--approve")

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        (entry or cli_module.main)(
            argv,
            input_fn=input_fn,
            is_interactive=interactive,
        )

    return buffer.getvalue(), prompts


def test_cli_explicit_confirmation_applies_and_verifies(tmp_path):
    output, prompts = run_cli(tmp_path, approve_flag=True, answer="yes")

    assert len(prompts) == 1
    assert output.index("Proposed diff:") < output.index("[Decision] APPROVE")
    assert "approve_repair | success=True" in output
    assert "verify_changes | success=True" in output
    assert disk(tmp_path) == REPAIRED


def test_cli_approve_flag_alone_never_applies_unseen_code(tmp_path):
    output, prompts = run_cli(tmp_path, approve_flag=True, answer="")

    assert len(prompts) == 1
    assert "No decision recorded" in output
    assert "approve_repair" not in output
    assert disk(tmp_path) == ORIGINAL


def test_cli_non_interactive_approval_fails_closed(tmp_path):
    output, prompts = run_cli(tmp_path, approve_flag=True, interactive=False)

    assert prompts == []
    assert "Proposed diff:" in output
    assert "Approval requires interactive confirmation" in output
    assert disk(tmp_path) == ORIGINAL


def test_cli_without_approve_shows_proposal_and_stops(tmp_path):
    output, prompts = run_cli(tmp_path, approve_flag=False)

    assert prompts == []
    assert "Proposed diff:" in output
    assert "Re-run with --approve to review and confirm it." in output
    assert disk(tmp_path) == ORIGINAL


def test_cli_answer_no_rejects(tmp_path):
    output, _ = run_cli(tmp_path, approve_flag=True, answer="no")

    assert "[Decision] REJECT" in output
    assert disk(tmp_path) == ORIGINAL


def test_cli_end_of_input_fails_closed(tmp_path):
    write(tmp_path)

    def input_fn(_):
        raise EOFError

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        cli_module.main(
            [str(tmp_path), "--task", TASK, "--approve"],
            input_fn=input_fn,
            is_interactive=True,
        )

    assert "No confirmation received" in buffer.getvalue()
    assert disk(tmp_path) == ORIGINAL


@pytest.mark.parametrize("answer, expected", [("yes", REPAIRED), ("", ORIGINAL)])
def test_agent_cli_requires_confirmation_too(tmp_path, answer, expected):
    run_cli(tmp_path, approve_flag=True, answer=answer, entry=agent_cli)

    assert disk(tmp_path) == expected


def test_fingerprint_binds_file_original_and_proposed(tmp_path):
    _, _, state = proposal_state(tmp_path)
    request = build_intervention_request(state)

    expected = compute_proposal_fingerprint("test_sample.py", ORIGINAL, REPAIRED)

    assert state.proposal_fingerprint == expected
    assert request.proposal_fingerprint == expected
    assert compute_proposal_fingerprint("other.py", ORIGINAL, REPAIRED) != expected
    assert compute_proposal_fingerprint("test_sample.py", ORIGINAL, ORIGINAL) != expected

    record = record_human_approval(state, expected)

    assert record == ApprovalRecord(
        fingerprint=expected,
        file="test_sample.py",
        code_version=0,
    )


def test_changed_proposal_invalidates_approval(tmp_path):
    workspace, registry, state = proposal_state(tmp_path)
    record_human_approval(state, state.proposal_fingerprint)
    state.user_input_needed = False
    state.human_input_reason = None

    state.proposed_content = "def test_sample():\n    pass\n"

    assert approval_problems(state)
    assert not check_guardrails(
        ActionResult(action="approve_repair", success=True, message=""), state
    ).allowed

    result = registry.execute("approve_repair", workspace=workspace, state=state)

    assert result.success is False
    assert result.data["stale_proposal"]
    assert state.repair_approval is None
    assert disk(tmp_path) == ORIGINAL


def test_changed_original_file_invalidates_approval(tmp_path):
    workspace, registry, state = proposal_state(tmp_path)
    record_human_approval(state, state.proposal_fingerprint)

    external = ORIGINAL + "# someone else's change\n"
    (tmp_path / "test_sample.py").write_text(external)

    result = registry.execute("approve_repair", workspace=workspace, state=state)

    assert result.success is False
    assert state.human_input_reason == HUMAN_INPUT_STALE_PROPOSAL
    assert disk(tmp_path) == external


def test_changed_code_version_invalidates_approval(tmp_path):
    workspace, registry, state = proposal_state(tmp_path)
    record_human_approval(state, state.proposal_fingerprint)

    state.code_version += 1

    assert any("code" in p.lower() for p in approval_problems(state))

    result = registry.execute("approve_repair", workspace=workspace, state=state)

    assert result.success is False
    assert disk(tmp_path) == ORIGINAL


def test_stale_proposal_cannot_be_applied_later(tmp_path):
    workspace, registry, state = proposal_state(tmp_path)
    record_human_approval(state, state.proposal_fingerprint)
    (tmp_path / "test_sample.py").write_text(ORIGINAL + "# drift\n")

    registry.execute("approve_repair", workspace=workspace, state=state)
    (tmp_path / "test_sample.py").write_text(ORIGINAL)
    retry = registry.execute("approve_repair", workspace=workspace, state=state)

    assert retry.success is False
    assert "No explicit human approval" in retry.message
    assert disk(tmp_path) == ORIGINAL


def test_reapproval_required_after_stale_proposal(tmp_path):
    session = start(tmp_path)
    old_request = session.intervention_request()
    approve(session)

    (tmp_path / "test_sample.py").write_text("# edited\n" + ORIGINAL)
    first = resume(session)

    assert actions(first) == ["approve_repair"]
    assert session.state.human_input_reason == HUMAN_INPUT_STALE_PROPOSAL

    decide(session, DECISION_REQUEST_CHANGES, information="Re-read the file.")
    second = resume(session)

    assert actions(second) == ["read_file", "propose_repair"]
    new_request = session.intervention_request()

    assert new_request.proposal_fingerprint != old_request.proposal_fingerprint

    with pytest.raises(HumanDecisionError):
        session.decide(
            HumanDecision(
                kind=DECISION_APPROVE,
                intervention_id=old_request.intervention_id,
                proposal_fingerprint=old_request.proposal_fingerprint,
                confirmed=True,
            )
        )

    approve(session)
    final = resume(session)

    assert final.status == AUTONOMOUS_COMPLETED
    assert disk(tmp_path) == "# edited\n" + REPAIRED


# ------------------------------------------------------------------
# Human decisions
# ------------------------------------------------------------------


def test_reject_leaves_file_unchanged_and_is_not_completed(tmp_path):
    session = start(tmp_path)
    fingerprint = session.state.proposal_fingerprint

    record = decide(session, DECISION_REJECT, information="Not this fix.")

    assert record.kind == DECISION_REJECT
    assert disk(tmp_path) == ORIGINAL
    assert session.state.repair_proposed is False
    assert fingerprint in session.state.rejected_proposals

    result = resume(session)

    # The rejected proposal is not recreated.
    assert actions(result) == ["propose_repair"]
    assert result.status == AUTONOMOUS_NEEDS_HUMAN
    assert session.state.human_input_reason == HUMAN_INPUT_NO_SAFE_REPAIR
    assert result.status != AUTONOMOUS_COMPLETED
    assert disk(tmp_path) == ORIGINAL


def test_request_changes_returns_to_safe_preparation(tmp_path):
    session = start(tmp_path)

    decide(session, DECISION_REQUEST_CHANGES, information="Keep the message.")

    state = session.state
    assert state.repair_proposed is False
    assert state.repair_approval is None
    assert state.current_file_content is None
    assert state.user_input_needed is False
    assert "Requested changes: Keep the message." in state.human_notes

    result = resume(session)

    assert actions(result) == ["read_file", "propose_repair"]
    # The only safe automatic repair is the one the human asked to change.
    assert state.human_input_reason == HUMAN_INPUT_NO_SAFE_REPAIR
    assert disk(tmp_path) == ORIGINAL


def test_provide_information_reaches_the_resumed_diagnosis(tmp_path):
    state = CodingState(
        task=TASK,
        task_intent="repair",
        last_error="RuntimeError: failure without a location",
    )
    session = start(tmp_path, state=state)

    assert session.state.human_input_reason == HUMAN_INPUT_INSUFFICIENT_DIAGNOSIS

    decide(
        session,
        DECISION_PROVIDE_INFORMATION,
        information="The failing test lives in test_sample.py.",
        file="test_sample.py",
    )
    result = resume(session)

    assert actions(result) == ["fix_error", "read_file", "propose_repair"]
    assert result.steps[0].result.data["human_hint_used"] is True
    assert "The failing test lives in test_sample.py." in session.state.human_notes
    assert session.state.human_input_reason == HUMAN_INPUT_REPAIR_APPROVAL


def test_cancel_stops_cleanly_and_is_not_completed(tmp_path):
    session = start(tmp_path)

    decide(session, DECISION_CANCEL_TASK, information="Not needed anymore.")

    assert session.cancelled
    assert session.status == AUTONOMOUS_CANCELLED
    assert session.lifecycle.status == TASK_CANCELLED
    assert session.result.progress_report is not None
    assert session.result.completed is False
    assert disk(tmp_path) == ORIGINAL

    report = session.report()

    assert report.human_decisions[-1].kind == DECISION_CANCEL_TASK
    assert "Human decisions:" in report.render()

    with pytest.raises(ValueError):
        session.resume()


def test_invalid_decisions_are_rejected(tmp_path):
    session = start(tmp_path)
    request = session.intervention_request()
    before = copy.deepcopy(session.state)

    invalid = [
        HumanDecision(
            kind=DECISION_PROVIDE_INFORMATION,
            intervention_id=request.intervention_id,
            information="x",
        ),
        HumanDecision(
            kind=DECISION_APPROVE,
            intervention_id="wrong-id",
            proposal_fingerprint=request.proposal_fingerprint,
            confirmed=True,
        ),
        HumanDecision(
            kind=DECISION_APPROVE,
            intervention_id=request.intervention_id,
            proposal_fingerprint=request.proposal_fingerprint,
            confirmed=False,
        ),
        HumanDecision(
            kind=DECISION_APPROVE,
            intervention_id=request.intervention_id,
            proposal_fingerprint="0" * 64,
            confirmed=True,
        ),
        HumanDecision(
            kind=DECISION_REQUEST_CHANGES,
            intervention_id=request.intervention_id,
        ),
    ]

    for decision in invalid:
        with pytest.raises(HumanDecisionError):
            session.decide(decision)

    assert session.state == before
    assert session.decisions == ()

    with pytest.raises(HumanDecisionError):
        HumanDecision(kind="MERGE", intervention_id=request.intervention_id)


def test_stale_request_and_unsafe_file_hint_rejected(tmp_path):
    _, _, state = proposal_state(tmp_path)
    request = build_intervention_request(state)

    changed = copy.deepcopy(state)
    changed.code_version += 1

    with pytest.raises(HumanDecisionError, match="stale"):
        apply_human_decision(
            changed,
            request,
            HumanDecision(kind=DECISION_REJECT, intervention_id=request.intervention_id),
        )

    blind = CodingState(task="x", user_input_needed=True)
    blind_request = build_intervention_request(blind)

    with pytest.raises(HumanDecisionError):
        apply_human_decision(
            blind,
            blind_request,
            HumanDecision(
                kind=DECISION_PROVIDE_INFORMATION,
                intervention_id=blind_request.intervention_id,
                information="look here",
                file="../outside.py",
            ),
        )


SWAPPED = "def test_sample():\n    pass\n"


@pytest.mark.parametrize("field", ["proposed_content", "proposal_original_content"])
def test_changed_proposal_contents_make_request_stale(tmp_path, field):
    _, _, state = proposal_state(tmp_path)
    request = build_intervention_request(state)
    fingerprint = state.proposal_fingerprint

    assert request.has_reviewed_proposal
    assert request.stale_reasons(state) == []

    # The contents change while the stored fingerprint does not.
    setattr(state, field, SWAPPED)

    assert state.proposal_fingerprint == fingerprint == request.proposal_fingerprint
    reasons = request.stale_reasons(state)

    assert any("content changed" in reason for reason in reasons)
    assert any("reviewed fingerprint" in reason for reason in reasons)
    assert not request.is_valid(state)


def test_changed_selected_file_makes_request_stale(tmp_path):
    _, _, state = proposal_state(tmp_path)
    request = build_intervention_request(state)

    state.current_file = "other_test.py"

    reasons = request.stale_reasons(state)

    assert any("selected file" in reason for reason in reasons)
    assert not request.is_valid(state)


def test_non_proposal_request_skips_proposal_checks():
    state = CodingState(task="x", user_input_needed=True)
    request = build_intervention_request(state)

    assert not request.has_reviewed_proposal
    assert request.stale_reasons(state) == []


@pytest.mark.parametrize(
    "kind, information",
    [
        (DECISION_REJECT, ""),
        (DECISION_REQUEST_CHANGES, "Please use a different approach."),
    ],
)
def test_stale_request_cannot_discard_a_different_proposal(
    tmp_path, kind, information
):
    _, _, state = proposal_state(tmp_path)
    request = build_intervention_request(state)
    reviewed_fingerprint = request.proposal_fingerprint

    # A different proposal now sits in state under the old fingerprint.
    state.proposed_content = SWAPPED
    before = copy.deepcopy(state)

    with pytest.raises(HumanDecisionError, match="stale"):
        apply_human_decision(
            state,
            request,
            HumanDecision(
                kind=kind,
                intervention_id=request.intervention_id,
                information=information,
            ),
        )

    # Nothing was discarded, rejected, or annotated.
    assert state == before
    assert state.repair_proposed is True
    assert state.proposed_content == SWAPPED
    assert state.proposal_fingerprint == reviewed_fingerprint
    assert state.rejected_proposals == []
    assert state.human_notes == []
    assert state.user_input_needed is True
    assert disk(tmp_path) == ORIGINAL


def test_already_applied_repair_cannot_be_approved_again(tmp_path):
    workspace, registry, state = proposal_state(tmp_path)
    request = build_intervention_request(state)

    apply_human_decision(
        state,
        request,
        HumanDecision(
            kind=DECISION_APPROVE,
            intervention_id=request.intervention_id,
            proposal_fingerprint=request.proposal_fingerprint,
            confirmed=True,
        ),
    )
    applied = registry.execute("approve_repair", workspace=workspace, state=state)

    assert applied.success is True
    assert state.repair_applied is True
    assert disk(tmp_path) == REPAIRED

    before = copy.deepcopy(state)
    again = approve_repair_action(workspace, state)

    assert again.success is False
    assert "already been applied" in again.message
    assert state == before
    assert state.repair_applied is True
    assert disk(tmp_path) == REPAIRED


def test_no_decision_without_pending_intervention(tmp_path):
    (tmp_path / "module.py").write_text("x = 1\n")
    session = quietly(
        TaskSession.start,
        FlyCoderAgent(tmp_path),
        "Inspect the project",
    )

    assert session.status == AUTONOMOUS_COMPLETED
    assert session.intervention_request() is None

    with pytest.raises(HumanDecisionError):
        session.decide(HumanDecision(kind=DECISION_CANCEL_TASK, intervention_id="x"))


# ------------------------------------------------------------------
# Task resumption
# ------------------------------------------------------------------


def test_resume_after_approval_completes_with_verification(tmp_path):
    session = start(tmp_path)
    first_history = session.result.progress_history
    first_recovery = session.result.recovery_history

    approve(session)
    result = resume(session)

    assert actions(result) == ["approve_repair", "verify_changes", "finish"]
    assert result.status == AUTONOMOUS_COMPLETED
    assert result.completion_evidence.status == EVIDENCE_VERIFIED
    assert result.completion_evidence.verification_method == (
        METHOD_VERIFICATION_PIPELINE
    )
    assert disk(tmp_path) == REPAIRED

    # Context is preserved, not restarted.
    assert session.state.task_intent == "repair"
    assert result.progress_history is first_history
    assert result.recovery_history is first_recovery
    iterations = [s.iteration for s in result.progress_history]
    assert iterations == sorted(set(iterations))
    assert iterations == list(range(session.iterations_used + 1))
    assert [s.iteration for s in session.steps] == list(
        range(1, session.iterations_used + 1)
    )

    transitions = [(t.from_state, t.to_state) for t in session.lifecycle.transitions]

    assert (TASK_NEEDS_HUMAN, TASK_RUNNING) in transitions
    assert transitions[-1][1] == TASK_COMPLETED
    assert all(b in ALLOWED_TRANSITIONS[a] for a, b in transitions[1:])

    report = session.report()

    assert report.verified
    assert report.intervention is None
    assert report.human_decisions[0].kind == DECISION_APPROVE


def test_resume_after_failed_verification_rolls_back_on_rejection(tmp_path):
    session = start(tmp_path, PARTIALLY_FIXABLE)
    approve(session)

    after_apply = resume(session)

    assert actions(after_apply) == ["approve_repair", "verify_changes", "fix_error"]
    assert session.state.human_input_reason == HUMAN_INPUT_APPLIED_REPAIR_FAILED
    assert session.intervention_request().allowed_decisions == (
        DECISION_REJECT,
        DECISION_CANCEL_TASK,
    )

    decide(session, DECISION_REJECT, information="Roll it back.")
    after_reject = resume(session)

    acted = actions(after_reject)

    assert acted[0] == "rollback_repair"
    assert disk(tmp_path) == PARTIALLY_FIXABLE
    assert session.state.code_version == 2
    # The rolled-back proposal is not offered again.
    assert acted[-1] == "propose_repair"
    assert session.state.human_input_reason == HUMAN_INPUT_NO_SAFE_REPAIR
    assert after_reject.status != AUTONOMOUS_COMPLETED


def test_iteration_limit_spans_the_whole_task(tmp_path):
    session = start(tmp_path, max_iterations=6)

    assert session.iterations_used == 5
    assert session.remaining_iterations == 1

    approve(session)
    result = resume(session)

    assert actions(result) == ["approve_repair"]
    assert result.status == AUTONOMOUS_MAX_ITERATIONS
    assert session.remaining_iterations == 0
    assert result.completed is False

    with pytest.raises(ValueError):
        session.resume()


def test_resume_requires_a_decision_first(tmp_path):
    session = start(tmp_path)

    with pytest.raises(ValueError, match="human decision"):
        session.resume()


def test_stale_verification_is_not_accepted_after_resume(tmp_path):
    session = start(tmp_path)
    old_state = copy.deepcopy(session.state)

    session.state.verification_record = record_verification(
        VerificationPipelineResult(
            change=ChangeVerificationResult(
                verified=True,
                file="test_sample.py",
                reason="old",
                expected_content=REPAIRED,
                actual_content=REPAIRED,
            ),
            targeted=None,
            targeted_analysis=None,
            regression=None,
            reason="recorded for older code",
        ),
        old_state,
    )
    stale_record = session.state.verification_record

    approve(session)
    result = resume(session)

    assert "verify_changes" in actions(result)
    assert result.completion_evidence.verification is not stale_record
    assert result.completion_evidence.verified


def test_repeated_intervention_only_allows_cancellation(tmp_path):
    session = start(tmp_path, UNREPAIRABLE)

    for _ in range(2):
        assert session.state.human_input_reason == HUMAN_INPUT_NO_SAFE_REPAIR
        decide(session, DECISION_PROVIDE_INFORMATION, information="Try again.")
        resume(session)

    request = session.intervention_request()

    assert request.repeated
    assert request.allowed_decisions == (DECISION_CANCEL_TASK,)

    with pytest.raises(HumanDecisionError):
        decide(session, DECISION_PROVIDE_INFORMATION, information="Again.")

    decide(session, DECISION_CANCEL_TASK)

    assert session.cancelled


# ------------------------------------------------------------------
# Safety
# ------------------------------------------------------------------


def test_approval_remains_subject_to_guardrails(tmp_path):
    _, _, state = proposal_state(tmp_path)
    approve_action = ActionResult(action="approve_repair", success=True, message="")

    # Without an approval record: blocked.
    assert not check_guardrails(approve_action, state).allowed

    record_human_approval(state, state.proposal_fingerprint)

    # Approved, but the human boundary is still set: blocked.
    assert not check_guardrails(approve_action, state).allowed

    state.user_input_needed = False
    state.human_input_reason = None

    assert check_guardrails(approve_action, state).allowed

    # A record for a different proposal is not accepted.
    state.repair_approval = ApprovalRecord(
        fingerprint="f" * 64,
        file="test_sample.py",
        code_version=0,
    )

    assert not check_guardrails(approve_action, state).allowed


def test_rollback_requires_human_authorization(tmp_path):
    state = CodingState(
        task="x",
        repair_proposed=True,
        repair_approved=True,
        repair_applied=True,
        current_file="a.py",
        current_file_content="x",
        repair_description="d",
        proposed_content="x",
        repair_original_content="y",
    )
    rollback = ActionResult(action="rollback_repair", success=True, message="")

    assert not check_guardrails(rollback, state).allowed

    state.rollback_authorized = True

    assert check_guardrails(rollback, state).allowed


def test_recovery_cannot_approve_repairs(tmp_path):
    _, _, state = proposal_state(tmp_path)
    before = copy.deepcopy(state)

    for action in ("approve_repair", "propose_repair", "run_tests"):
        decision = recover_from_failure(
            ActionResult(action=action, success=False, message="x"),
            state,
        )
        assert decision.recovery_action != "approve_repair"

    assert state == before
    assert state.repair_approval is None


def test_human_decisions_never_run_actions_outside_run_once(tmp_path):
    write(tmp_path)
    agent = FlyCoderAgent(tmp_path)
    inside = {"flag": False}
    outside = []
    original_execute = agent.actions.execute
    original_run_once = agent.run_once

    def tracking_execute(name, **kwargs):
        if not inside["flag"]:
            outside.append(name)
        return original_execute(name, **kwargs)

    def tracking_run_once(state):
        inside["flag"] = True
        try:
            return original_run_once(state)
        finally:
            inside["flag"] = False

    agent.actions.execute = tracking_execute
    agent.run_once = tracking_run_once

    session = quietly(TaskSession.start, agent, TASK)
    approve(session)
    resume(session)

    assert outside == []
    assert session.status == AUTONOMOUS_COMPLETED


def test_only_verified_evidence_completes_after_resume(tmp_path):
    session = start(tmp_path, PARTIALLY_FIXABLE)
    approve(session)
    result = resume(session)

    # The approved repair was applied, but verification failed.
    assert session.state.repair_applied
    assert result.status == AUTONOMOUS_NEEDS_HUMAN
    assert result.completion_evidence.verified is False
    assert session.lifecycle.status != TASK_COMPLETED
