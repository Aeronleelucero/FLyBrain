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

