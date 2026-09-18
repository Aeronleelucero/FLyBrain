"""Tests for safe repair rollback and recovery."""

from flycoder.actions.approval_actions import approve_repair_action
from flycoder.actions.repair_actions import propose_repair_action
from flycoder.actions.rollback_actions import rollback_repair_action
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace


def _create_workspace(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.write_file(
        "test_example.py",
        'def test_example():\n'
        '    assert False, "Intentional failure for testing"\n',
    )
    return workspace


def _create_state(workspace):
    return CodingState(
        task="repair failing test",
        current_file="test_example.py",
        current_file_content=workspace.read_file("test_example.py"),
    )


def _apply_repair(workspace, state):
    proposal = propose_repair_action(workspace, state)
    assert proposal.success is True

    approval = approve_repair_action(workspace, state)
    assert approval.success is True


def test_rollback_restores_original_content(tmp_path):
    workspace = _create_workspace(tmp_path)
    state = _create_state(workspace)

    original = workspace.read_file("test_example.py")

    _apply_repair(workspace, state)

    assert workspace.read_file("test_example.py") != original

    result = rollback_repair_action(workspace, state)

    assert result.success is True
    assert workspace.read_file("test_example.py") == original


def test_rollback_clears_repair_state(tmp_path):
    workspace = _create_workspace(tmp_path)
    state = _create_state(workspace)

    _apply_repair(workspace, state)

    result = rollback_repair_action(workspace, state)

    assert result.success is True
    assert state.repair_applied is False
    assert state.repair_approved is False
    assert state.repair_proposed is False
    assert state.proposed_content is None
    assert state.repair_original_content is None
    assert state.user_input_needed is False


def test_rollback_updates_current_file_content(tmp_path):
    workspace = _create_workspace(tmp_path)
    state = _create_state(workspace)

    original = state.current_file_content

    _apply_repair(workspace, state)

    rollback_repair_action(workspace, state)

    assert state.current_file_content == original


def test_rollback_requires_applied_repair(tmp_path):
    workspace = _create_workspace(tmp_path)
    state = _create_state(workspace)

    result = rollback_repair_action(workspace, state)

    assert result.success is False
    assert "No applied repair" in result.message


def test_rollback_requires_selected_file(tmp_path):
    workspace = _create_workspace(tmp_path)
    state = CodingState(
        task="repair failing test",
        repair_applied=True,
        repair_original_content="original",
    )

    result = rollback_repair_action(workspace, state)

    assert result.success is False
    assert "No file is selected" in result.message


def test_rollback_requires_original_content(tmp_path):
    workspace = _create_workspace(tmp_path)
    state = CodingState(
        task="repair failing test",
        repair_applied=True,
        current_file="test_example.py",
    )

    result = rollback_repair_action(workspace, state)

    assert result.success is False
    assert "No original content" in result.message


def test_rollback_rejects_file_changed_after_apply(tmp_path):
    workspace = _create_workspace(tmp_path)
    state = _create_state(workspace)

    _apply_repair(workspace, state)

    external_content = (
        'def test_example():\n'
        '    print("New external change")\n'
        '    assert True\n'
    )
    workspace.write_file("test_example.py", external_content)

    result = rollback_repair_action(workspace, state)

    assert result.success is False
    assert result.data["changed_after_apply"] is True
    assert result.data["requires_manual_review"] is True

    # The external modification must not be overwritten.
    assert workspace.read_file("test_example.py") == external_content


def test_rollback_uses_atomic_write(tmp_path, monkeypatch):
    workspace = _create_workspace(tmp_path)
    state = _create_state(workspace)

    _apply_repair(workspace, state)

    calls = []
    original_atomic_write = workspace.atomic_write_file

    def tracked_atomic_write(relative_path, content):
        calls.append((relative_path, content))
        original_atomic_write(relative_path, content)

    monkeypatch.setattr(
        workspace,
        "atomic_write_file",
        tracked_atomic_write,
    )

    result = rollback_repair_action(workspace, state)

    assert result.success is True
    assert len(calls) == 1
    assert calls[0][0] == "test_example.py"
    assert calls[0][1] == state.current_file_content
