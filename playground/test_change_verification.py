"""Tests for Phase 7.5 change verification."""

from pathlib import Path

from flycoder.tools.changes import ChangeProposal
from flycoder.tools.filesystem import Workspace
from flycoder.tools.verification import (
    build_change_verification_report,
    verify_change,
)


def create_workspace(
    tmp_path: Path,
    content: str = "original\n",
) -> Workspace:
    """Create a workspace containing one source file."""
    source = tmp_path / "example.py"
    source.write_text(
        content,
        encoding="utf-8",
    )

    return Workspace(tmp_path)


def test_matching_content_is_verified(tmp_path):
    workspace = create_workspace(
        tmp_path,
        "updated\n",
    )

    proposal = ChangeProposal(
        file="example.py",
        original_content="original\n",
        proposed_content="updated\n",
        description="update example",
    )

    result = verify_change(
        workspace,
        proposal,
    )

    assert result.verified is True
    assert result.file == "example.py"
    assert result.reason == (
        "actual file content matches proposed content"
    )
    assert result.actual_content == "updated\n"
    assert result.expected_content == "updated\n"


def test_mismatched_content_is_rejected(tmp_path):
    workspace = create_workspace(
        tmp_path,
        "different\n",
    )

    proposal = ChangeProposal(
        file="example.py",
        original_content="original\n",
        proposed_content="updated\n",
        description="update example",
    )

    result = verify_change(
        workspace,
        proposal,
    )

    assert result.verified is False
    assert result.file == "example.py"
    assert result.reason == (
        "actual file content does not match "
        "the proposed content"
    )
    assert result.actual_content == "different\n"
    assert result.expected_content == "updated\n"


def test_missing_file_is_rejected(tmp_path):
    workspace = Workspace(tmp_path)

    proposal = ChangeProposal(
        file="missing.py",
        original_content="original\n",
        proposed_content="updated\n",
        description="update missing file",
    )

    result = verify_change(
        workspace,
        proposal,
    )

    assert result.verified is False
    assert result.file == "missing.py"
    assert "unable to read target file" in result.reason


def test_empty_file_name_is_rejected(tmp_path):
    workspace = create_workspace(tmp_path)

    proposal = ChangeProposal(
        file="",
        original_content="original\n",
        proposed_content="updated\n",
        description="invalid proposal",
    )

    result = verify_change(
        workspace,
        proposal,
    )

    assert result.verified is False
    assert result.file is None
    assert result.reason == (
        "change proposal has no target file"
    )


def test_verification_does_not_modify_file(tmp_path):
    workspace = create_workspace(
        tmp_path,
        "actual\n",
    )

    proposal = ChangeProposal(
        file="example.py",
        original_content="original\n",
        proposed_content="updated\n",
        description="update example",
    )

    verify_change(
        workspace,
        proposal,
    )

    assert workspace.read_file("example.py") == "actual\n"


def test_verification_report_is_serializable(tmp_path):
    workspace = create_workspace(
        tmp_path,
        "updated\n",
    )

    proposal = ChangeProposal(
        file="example.py",
        original_content="original\n",
        proposed_content="updated\n",
    )

    result = verify_change(
        workspace,
        proposal,
    )

    report = build_change_verification_report(
        result
    )

    assert report == {
        "verified": True,
        "file": "example.py",
        "reason": (
            "actual file content matches proposed content"
        ),
        "expected_content": "updated\n",
        "actual_content": "updated\n",
        "changed": True,
    }


def test_noop_proposal_can_be_verified(tmp_path):
    workspace = create_workspace(
        tmp_path,
        "same\n",
    )

    proposal = ChangeProposal(
        file="example.py",
        original_content="same\n",
        proposed_content="same\n",
        description="no-op",
    )

    result = verify_change(
        workspace,
        proposal,
    )

    assert result.verified is True
    assert result.changed is True


def test_outside_workspace_file_is_rejected(tmp_path):
    workspace = create_workspace(tmp_path)

    outside_file = tmp_path.parent / "outside.py"
    outside_file.write_text(
        "outside\n",
        encoding="utf-8",
    )

    proposal = ChangeProposal(
        file="../outside.py",
        original_content="original\n",
        proposed_content="updated\n",
    )

    result = verify_change(
        workspace,
        proposal,
    )

    assert result.verified is False
    assert result.file == "../outside.py"
    