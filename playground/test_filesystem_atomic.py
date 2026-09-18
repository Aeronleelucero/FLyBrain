"""Tests for safe atomic workspace writes."""

from flycoder.tools.filesystem import Workspace


def test_atomic_write_replaces_existing_file(tmp_path):
    workspace = Workspace(tmp_path)

    workspace.write_file("example.py", "original\n")
    workspace.atomic_write_file("example.py", "updated\n")

    assert workspace.read_file("example.py") == "updated\n"


def test_atomic_write_creates_missing_file(tmp_path):
    workspace = Workspace(tmp_path)

    workspace.atomic_write_file("nested/example.py", "created\n")

    assert workspace.read_file("nested/example.py") == "created\n"


def test_atomic_write_preserves_complete_content(tmp_path):
    workspace = Workspace(tmp_path)

    content = (
        "def example():\n"
        "    value = 42\n"
        "    return value\n"
    )

    workspace.atomic_write_file("example.py", content)

    assert workspace.read_file("example.py") == content


def test_atomic_write_rejects_path_outside_workspace(tmp_path):
    workspace = Workspace(tmp_path)

    try:
        workspace.atomic_write_file("../outside.py", "unsafe\n")
    except ValueError as exc:
        assert "outside the workspace" in str(exc)
    else:
        raise AssertionError("Expected path traversal to be rejected")


def test_atomic_write_does_not_leave_temporary_files(tmp_path):
    workspace = Workspace(tmp_path)

    workspace.atomic_write_file("example.py", "content\n")

    temporary_files = list(tmp_path.glob(".example.py.*.tmp"))

    assert temporary_files == []
