"""Change proposal tools for FLY-CODER."""

from dataclasses import dataclass
from difflib import unified_diff


@dataclass
class ChangeProposal:
    """Represent a proposed modification without applying it."""

    file: str
    original_content: str
    proposed_content: str
    description: str = ""

    @property
    def changed(self) -> bool:
        """Return whether the proposal changes the file."""
        return self.original_content != self.proposed_content


def create_change_proposal(
    file: str,
    original_content: str,
    proposed_content: str,
    description: str = "",
) -> ChangeProposal:
    """Create a change proposal without modifying any file."""

    return ChangeProposal(
        file=file,
        original_content=original_content,
        proposed_content=proposed_content,
        description=description,
    )


def build_change_diff(
    proposal: ChangeProposal,
) -> str:
    """Build a unified diff for a change proposal."""

    if not proposal.changed:
        return ""

    diff = unified_diff(
        proposal.original_content.splitlines(
            keepends=True
        ),
        proposal.proposed_content.splitlines(
            keepends=True
        ),
        fromfile=f"a/{proposal.file}",
        tofile=f"b/{proposal.file}",
    )

    return "".join(diff)
