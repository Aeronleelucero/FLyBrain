"""Change verification tools for FLY-CODER."""

from dataclasses import dataclass
from pathlib import Path

from flycoder.tools.changes import ChangeProposal
from flycoder.tools.filesystem import Workspace


@dataclass
class ChangeVerificationResult:
    """Represent the result of verifying an applied change."""

    __test__ = False

    verified: bool
    file: str | None
    reason: str
    expected_content: str | None = None
    actual_content: str | None = None

    @property
    def changed(self) -> bool:
        """Return whether the expected content differs from the original."""
        if self.expected_content is None:
            return False

        if self.actual_content is None:
            return False

        return self.actual_content == self.expected_content


def verify_change(
    workspace: Workspace,
    proposal: ChangeProposal,
) -> ChangeVerificationResult:
    """
    Verify that a workspace file contains the proposed content.

    Verification is read-only. It never modifies the workspace.
    """
    if not proposal.file:
        return ChangeVerificationResult(
            verified=False,
            file=None,
            reason="change proposal has no target file",
            expected_content=proposal.proposed_content,
        )

    if proposal.proposed_content is None:
        return ChangeVerificationResult(
            verified=False,
            file=proposal.file,
            reason="change proposal has no proposed content",
        )

    try:
        actual_content = workspace.read_file(
            proposal.file
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        return ChangeVerificationResult(
            verified=False,
            file=proposal.file,
            reason=(
                f"unable to read target file: {exc}"
            ),
            expected_content=proposal.proposed_content,
        )

    if actual_content != proposal.proposed_content:
        return ChangeVerificationResult(
            verified=False,
            file=proposal.file,
            reason=(
                "actual file content does not match "
                "the proposed content"
            ),
            expected_content=proposal.proposed_content,
            actual_content=actual_content,
        )

    return ChangeVerificationResult(
        verified=True,
        file=proposal.file,
        reason="actual file content matches proposed content",
        expected_content=proposal.proposed_content,
        actual_content=actual_content,
    )


def build_change_verification_report(
    result: ChangeVerificationResult,
) -> dict:
    """Build a serializable change-verification report."""
    return {
        "verified": result.verified,
        "file": result.file,
        "reason": result.reason,
        "expected_content": result.expected_content,
        "actual_content": result.actual_content,
        "changed": result.changed,
    }
