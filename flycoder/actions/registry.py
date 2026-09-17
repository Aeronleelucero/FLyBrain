"""Action registry for FLY-CODER."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ActionResult:
    """Result returned after executing an action."""

    action: str
    success: bool
    message: str
    data: Any = None


class ActionRegistry:
    """Store and execute available agent actions."""

    def __init__(self):
        self._actions: dict[str, Callable[..., ActionResult]] = {}

    def register(
        self,
        name: str,
        function: Callable[..., ActionResult],
    ) -> None:
        """Register an action."""

        if name in self._actions:
            raise ValueError(f"Action already registered: {name}")

        self._actions[name] = function

    def execute(
        self,
        name: str,
        **kwargs: Any,
    ) -> ActionResult:
        """Execute a registered action."""

        if name not in self._actions:
            return ActionResult(
                action=name,
                success=False,
                message=f"Unknown action: {name}",
            )

        try:
            return self._actions[name](**kwargs)

        except Exception as error:
            return ActionResult(
                action=name,
                success=False,
                message=str(error),
            )

    def list_actions(self) -> list[str]:
        """Return all registered action names."""

        return sorted(self._actions)