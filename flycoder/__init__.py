"""Available FLY-CODER actions.

Phase 10.9: ``flycoder.actions.create_action_registry`` is the single
definition of the default action registry. This package re-exports it
(and the action functions it previously exposed) instead of keeping a
second, divergent registry definition.
"""

from flycoder.actions import create_action_registry
from flycoder.actions.analysis_actions import (
    explain_error_action,
    improve_code_action,
    review_code_action,
)
from flycoder.actions.approval_actions import approve_repair_action
from flycoder.actions.error_actions import fix_error_action
from flycoder.actions.filesystem_actions import (
    inspect_files_action,
    read_file_action,
    write_file_action,
)
from flycoder.actions.registry import ActionRegistry
from flycoder.actions.repair_actions import propose_repair_action
from flycoder.actions.testing_actions import run_tests_action

__all__ = [
    "ActionRegistry",
    "approve_repair_action",
    "create_action_registry",
    "explain_error_action",
    "fix_error_action",
    "improve_code_action",
    "inspect_files_action",
    "propose_repair_action",
    "read_file_action",
    "review_code_action",
    "run_tests_action",
    "write_file_action",
]
