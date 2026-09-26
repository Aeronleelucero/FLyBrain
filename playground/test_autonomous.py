from pathlib import Path

from flycoder.actions.registry import ActionResult
from flycoder.agent import FlyCoderAgent
from flycoder.state import CodingState
from flycoder.tools.autonomous import (
    AutonomousStep,
    AutonomousTaskResult,
    AUTONOMOUS_BLOCKED,
    AUTONOMOUS_COMPLETED,
    AUTONOMOUS_FAILED,
    AUTONOMOUS_MAX_ITERATIONS,
    AUTONOMOUS_NEEDS_HUMAN,
    run_autonomous_task,
)
from flycoder.tools.execution import ExecutionObservation
from flycoder.tools.recovery import RecoveryDecision
from flycoder.tools.task_lifecycle import (
    TASK_COMPLETED,
    TASK_INITIALIZING,
    TASK_RUNNING,
    TASK_VERIFYING,
)


def test_autonomous_task_result_completed():
    result = AutonomousTaskResult(
        task="Inspect project",
        status=AUTONOMOUS_COMPLETED,
        iterations=2,
    )

    assert result.completed is True
    assert result.failed is False
    assert result.blocked is False
    assert result.needs_human is False
    assert result.reached_iteration_limit is False


def test_autonomous_task_result_failed():
    result = AutonomousTaskResult(
        task="Repair project",
        status=AUTONOMOUS_FAILED,
        iterations=3,
    )

    assert result.failed is True
    assert result.completed is False


def test_autonomous_task_result_blocked():
    result = AutonomousTaskResult(
        task="Modify project",
        status=AUTONOMOUS_BLOCKED,
        iterations=1,
    )

    assert result.blocked is True
    assert result.needs_human is False


def test_autonomous_task_result_needs_human():
    result = AutonomousTaskResult(
        task="Repair project",
        status=AUTONOMOUS_NEEDS_HUMAN,
        iterations=4,
    )

    assert result.needs_human is True
    assert result.completed is False


def test_autonomous_task_result_iteration_limit():
    result = AutonomousTaskResult(
        task="Run task",
        status=AUTONOMOUS_MAX_ITERATIONS,
        iterations=10,
    )

    assert result.reached_iteration_limit is True
    assert result.completed is False


def test_autonomous_step_records_execution_context():
    action_result = ActionResult(
        action="inspect_files",
        success=True,
        message="Files inspected.",
    )

    observation = ExecutionObservation(
        action="inspect_files",
        success=True,
        message="Files inspected.",
        next_action=None,
    )

    recovery = RecoveryDecision(
        action="inspect_files",
        failed=False,
        reason="No recovery required.",
    )

    step = AutonomousStep(
        iteration=1,
        result=action_result,
        observation=observation,
        recovery=recovery,
    )

    assert step.iteration == 1
    assert step.result.action == "inspect_files"
    assert step.result.success is True
    assert step.observation is observation
    assert step.recovery is recovery


def test_autonomous_task_completes_inspection():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(
        task="Inspect playground files"
    )

    result = run_autonomous_task(
        agent,
        state,
        max_iterations=5,
    )

    assert result.completed is True
    assert result.iterations >= 2
    assert result.final_result is not None
    assert result.final_result.action == "finish"
    assert state.finished is True


def test_autonomous_task_respects_iteration_limit():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(
        task="Inspect playground files"
    )

    result = run_autonomous_task(
        agent,
        state,
        max_iterations=1,
    )

    assert result.reached_iteration_limit is True
    assert result.iterations == 1
    assert state.finished is False


def test_autonomous_task_stops_at_human_boundary():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(
        task="Repair the failing code"
    )

    state.user_input_needed = True

    result = run_autonomous_task(
        agent,
        state,
        max_iterations=5,
    )

    assert result.needs_human is True
    assert result.iterations == 1
    assert result.final_result is not None

    # Autonomous execution must stop without approval.
    assert state.repair_approved is False
    assert state.repair_applied is False


def test_autonomous_task_does_not_auto_approve_repair():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(
        task="Repair the failing code"
    )

    # Start directly at the explicit repair approval boundary.
    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.last_error = "Test failed."
    state.error_inspected = True
    state.repair_proposed = True
    state.user_input_needed = True

    result = run_autonomous_task(
        agent,
        state,
        max_iterations=5,
    )

    # The autonomous loop must stop immediately at the
    # human approval boundary.
    assert result.needs_human is True
    assert result.iterations == 1

    # Most important safety guarantees:
    # autonomous execution never approves or applies
    # the repair on behalf of the user.
    assert state.repair_proposed is True
    assert state.repair_approved is False
    assert state.repair_applied is False


def test_autonomous_result_contains_task_lifecycle():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(
        task="Inspect playground files"
    )

    result = run_autonomous_task(
        agent,
        state,
        max_iterations=10,
    )

    assert result.lifecycle is not None
    assert result.lifecycle.task == "Inspect playground files"
    assert result.lifecycle.completed is True


def test_autonomous_lifecycle_records_iterations_and_actions():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(
        task="Inspect playground files"
    )

    result = run_autonomous_task(
        agent,
        state,
        max_iterations=10,
    )

    assert result.lifecycle is not None
    assert result.lifecycle.current_iteration == result.iterations
    assert result.lifecycle.actions


def test_autonomous_lifecycle_records_completion_transitions():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(
        task="Inspect playground files"
    )

    result = run_autonomous_task(
        agent,
        state,
        max_iterations=10,
    )

    assert result.lifecycle is not None

    states = [
        transition.to_state
        for transition in result.lifecycle.transitions
    ]

    assert states[0] == TASK_INITIALIZING
    assert TASK_RUNNING in states
    assert TASK_VERIFYING in states
    assert states[-1] == TASK_COMPLETED


def test_autonomous_human_boundary_updates_lifecycle():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(
        task="Repair the failing code"
    )

    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.last_error = "Test failed."
    state.error_inspected = True
    state.repair_proposed = True
    state.user_input_needed = True

    result = run_autonomous_task(
        agent,
        state,
        max_iterations=10,
    )

    assert result.needs_human is True
    assert result.lifecycle is not None
    assert result.lifecycle.needs_human is True
    assert result.lifecycle.completed_at is not None


def test_autonomous_does_not_auto_approve_repair():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(
        task="Repair the failing code"
    )

    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.last_error = "Test failed."
    state.error_inspected = True
    state.repair_proposed = True
    state.user_input_needed = True

    result = run_autonomous_task(
        agent,
        state,
        max_iterations=10,
    )

    assert result.needs_human is True
    assert state.repair_approved is False
    assert state.repair_applied is False


def test_autonomous_step_contains_iteration_observation():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(
        task="Inspect playground files"
    )

    result = run_autonomous_task(
        agent,
        state,
        max_iterations=5,
    )

    assert result.completed is True
    assert result.steps

    for step in result.steps:
        assert step.iteration_observation is not None
        assert step.iteration_observation.iteration == step.iteration


def test_autonomous_step_contains_stall_assessment():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(
        task="Inspect playground files"
    )

    result = run_autonomous_task(
        agent,
        state,
        max_iterations=5,
    )

    assert result.steps

    for step in result.steps:
        assert step.stall is not None


def test_autonomous_step_contains_iteration_decision():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(
        task="Inspect playground files"
    )

    result = run_autonomous_task(
        agent,
        state,
        max_iterations=5,
    )

    assert result.steps

    for step in result.steps:
        assert step.iteration_decision is not None


def test_autonomous_iteration_decision_does_not_auto_approve():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(
        task="Repair the failing code"
    )

    state.current_file = "example.py"
    state.current_file_content = "print('hello')"
    state.last_error = "Test failed."
    state.error_inspected = True
    state.repair_proposed = True
    state.user_input_needed = True

    result = run_autonomous_task(
        agent,
        state,
        max_iterations=5,
    )

    assert result.needs_human is True
    assert state.repair_approved is False
    assert state.repair_applied is False
