from pathlib import Path

from flycoder.agent import FlyCoderAgent
from flycoder.state import CodingState


def test_agent_can_inspect_workspace():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(task="Inspect playground files")

    result = agent.run_once(state)

    assert result.success is True
    assert result.action == "inspect_files"
    assert "example.py" in state.files
    assert "test_example.py" in state.files

def test_agent_records_last_action():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(task="Inspect playground files")

    result = agent.run_once(state)

    assert result.success is True
    assert state.last_action == "inspect_files"
    assert state.last_action_success is True
    assert state.last_action_message == result.message


def test_agent_attaches_execution_observation():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(task="Inspect playground files")

    result = agent.run_once(state)

    observation = result.data["execution_observation"]

    assert observation["action"] == "inspect_files"
    assert observation["success"] is True
    assert observation["next_action"] is None


def test_agent_does_not_execute_adaptive_next_action_automatically():
    workspace = Path(__file__).parent

    agent = FlyCoderAgent(workspace)
    state = CodingState(task="Run tests")

    result = agent.run_once(state)

    # The first cycle still performs only one action.
    assert result.action == "inspect_files"
    assert state.last_action == "inspect_files"

