"""Available FLY-CODER actions."""

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
from flycoder.actions.rollback_actions import rollback_repair_action
from flycoder.actions.testing_actions import run_tests_action


def create_action_registry() -> ActionRegistry:
    """Create the default action registry."""

    registry = ActionRegistry()

    registry.register("inspect_files", inspect_files_action)
    registry.register("read_file", read_file_action)
    registry.register("write_code", write_file_action)

    registry.register("run_tests", run_tests_action)

    registry.register("fix_error", fix_error_action)
    registry.register("propose_repair", propose_repair_action)
    registry.register("approve_repair", approve_repair_action)
    registry.register("rollback_repair", rollback_repair_action)

    registry.register("explain_error", explain_error_action)
    registry.register("review_code", review_code_action)
    registry.register("improve_code", improve_code_action)

    return registry
