"""Integration tests for the Phase 6 change proposal and approval boundary."""

from flycoder.actions.approval_actions import approve_repair_action
from flycoder.actions.repair_actions import propose_repair_action
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace


def _create_workspace(tmp_path):
    workspace = Workspace(tmp_path)
    file_path = tmp_path / "test_example.py"
    file_path.write_text(
        'def test_example():\n'
        '    assert False, "Intentional failure for testing"\n',
        encoding="utf-8",
    )
    return workspace


def test_propose_repair_generates_diff_without_modifying_file(tmp_path):
    workspace = _create_workspace(tmp_path)
    state = CodingState(
        task="repair failing test",
        current_file="test_example.py",
        current_file_content=workspace.read_file("test_example.py"),
    )

    original = workspace.read_file("test_example.py")

    result = propose_repair_action(workspace, state)

    assert result.success is True
    assert result.data["diff"]
    assert result.data["requires_approval"] is True
    assert workspace.read_file("test_example.py") == original
    assert state.repair_proposed is True
    assert state.repair_approved is False
    assert state.repair_applied is False
    assert state.user_input_needed is True


def test_approval_applies_proposed_change(tmp_path):
    workspace = _create_workspace(tmp_path)
    state = CodingState(
        task="repair failing test",
        current_file="test_example.py",
        current_file_content=workspace.read_file("test_example.py"),
    )

    proposal = propose_repair_action(workspace, state)

    assert proposal.success is True
    assert workspace.read_file("test_example.py") == state.current_file_content

    result = approve_repair_action(workspace, state)

    assert result.success is True
    assert "assert True" in workspace.read_file("test_example.py")
    assert state.repair_approved is True
    assert state.repair_applied is True
    assert state.user_input_needed is False


def test_approval_rejects_stale_proposal(tmp_path):
    workspace = _create_workspace(tmp_path)
    state = CodingState(
        task="repair failing test",
        current_file="test_example.py",
        current_file_content=workspace.read_file("test_example.py"),
    )

    proposal = propose_repair_action(workspace, state)

    assert proposal.success is True

    changed_content = (
        'def test_example():\n'
        '    print("Someone changed this file")\n'
        '    assert False, "Intentional failure for testing"\n'
    )
    workspace.write_file("test_example.py", changed_content)

    result = approve_repair_action(workspace, state)

    assert result.success is False
    assert result.data["stale_proposal"] is True
    assert result.data["requires_new_proposal"] is True
    assert "changed" in result.message.lower()

    # The external change must remain intact.
    assert workspace.read_file("test_example.py") == changed_content

    assert state.repair_approved is False
    assert state.repair_applied is False
    assert state.user_input_needed is True


def test_approval_requires_proposal(tmp_path):
    workspace = _create_workspace(tmp_path)
    state = CodingState(task="repair failing test")

    result = approve_repair_action(workspace, state)

    assert result.success is False
    assert "No repair proposal" in result.message


def test_approval_requires_selected_file(tmp_path):
    workspace = _create_workspace(tmp_path)
    state = CodingState(
        task="repair failing test",
        repair_proposed=True,
        proposed_content="changed",
    )

    result = approve_repair_action(workspace, state)

    assert result.success is False
    assert "No file is selected" in result.message


def test_approval_requires_proposed_content(tmp_path):
    workspace = _create_workspace(tmp_path)
    state = CodingState(
        task="repair failing test",
        repair_proposed=True,
        current_file="test_example.py",
    )

    result = approve_repair_action(workspace, state)

    assert result.success is False
    assert "No proposed content" in result.message
