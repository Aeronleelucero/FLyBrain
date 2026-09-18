"""Tests for Phase 6.1 change proposal tools."""

from flycoder.tools.changes import (
    ChangeProposal,
    build_change_diff,
    create_change_proposal,
)


def test_create_change_proposal():
    proposal = create_change_proposal(
        file="app.py",
        original_content="print('old')\n",
        proposed_content="print('new')\n",
        description="Update output.",
    )

    assert isinstance(proposal, ChangeProposal)
    assert proposal.file == "app.py"
    assert proposal.original_content == "print('old')\n"
    assert proposal.proposed_content == "print('new')\n"
    assert proposal.description == "Update output."


def test_proposal_detects_change():
    proposal = create_change_proposal(
        file="app.py",
        original_content="old\n",
        proposed_content="new\n",
    )

    assert proposal.changed is True


def test_identical_content_is_not_a_change():
    proposal = create_change_proposal(
        file="app.py",
        original_content="same\n",
        proposed_content="same\n",
    )

    assert proposal.changed is False


def test_diff_contains_file_names():
    proposal = create_change_proposal(
        file="app.py",
        original_content="old\n",
        proposed_content="new\n",
    )

    diff = build_change_diff(proposal)

    assert "--- a/app.py" in diff
    assert "+++ b/app.py" in diff


def test_diff_contains_removed_content():
    proposal = create_change_proposal(
        file="app.py",
        original_content="old\n",
        proposed_content="new\n",
    )

    diff = build_change_diff(proposal)

    assert "-old" in diff


def test_diff_contains_added_content():
    proposal = create_change_proposal(
        file="app.py",
        original_content="old\n",
        proposed_content="new\n",
    )

    diff = build_change_diff(proposal)

    assert "+new" in diff


def test_unchanged_proposal_has_empty_diff():
    proposal = create_change_proposal(
        file="app.py",
        original_content="same\n",
        proposed_content="same\n",
    )

    assert build_change_diff(proposal) == ""


def test_multiline_diff():
    proposal = create_change_proposal(
        file="app.py",
        original_content=(
            "def hello():\n"
            "    return 'old'\n"
        ),
        proposed_content=(
            "def hello():\n"
            "    return 'new'\n"
        ),
    )

    diff = build_change_diff(proposal)

    assert "-    return 'old'" in diff
    assert "+    return 'new'" in diff


def test_proposal_does_not_modify_content():
    original = "old\n"
    proposed = "new\n"

    proposal = create_change_proposal(
        file="app.py",
        original_content=original,
        proposed_content=proposed,
    )

    build_change_diff(proposal)

    assert proposal.original_content == original
    assert proposal.proposed_content == proposed
