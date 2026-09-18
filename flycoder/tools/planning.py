"""Task decomposition and planning tools for FLY-CODER."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskItem:
    """One decomposed item from a coding task."""

    description: str
    category: str


@dataclass
class TaskPlan:
    """Structured decomposition of a coding task."""

    task: str
    items: list[TaskItem] = field(default_factory=list)


def _normalize_task(task: str) -> str:
    """Normalize task text without changing its meaning."""

    return " ".join(task.strip().split())


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    """Return whether text contains any of the supplied phrases."""

    return any(phrase in text for phrase in phrases)


def decompose_task(task: str) -> TaskPlan:
    """
    Decompose a coding task into broad, deterministic work items.

    This function intentionally does not identify files, symbols,
    dependencies, or code changes. Those responsibilities belong to
    later Phase 5 planning stages.
    """

    normalized_task = _normalize_task(task)

    if not normalized_task:
        return TaskPlan(task="", items=[])

    text = normalized_task.lower()

    items: list[TaskItem] = []

    # Every non-empty coding task begins with understanding the
    # requested behavior.
    items.append(
        TaskItem(
            description="Understand the requested behavior.",
            category="understand",
        )
    )

    if _contains_any(
        text,
        (
            "fix",
            "repair",
            "debug",
            "bug",
            "error",
            "broken",
            "failing",
        ),
    ):
        items.append(
            TaskItem(
                description="Investigate the reported problem.",
                category="repair",
            )
        )

    if _contains_any(
        text,
        (
            "add",
            "create",
            "implement",
            "build",
            "introduce",
            "support",
        ),
    ):
        items.append(
            TaskItem(
                description="Implement the requested functionality.",
                category="implement",
            )
        )

    if _contains_any(
        text,
        (
            "change",
            "update",
            "modify",
            "replace",
            "remove",
            "delete",
        ),
    ):
        items.append(
            TaskItem(
                description="Apply the requested code changes.",
                category="modify",
            )
        )

    if _contains_any(
        text,
        (
            "refactor",
            "restructure",
            "clean up",
            "simplify",
        ),
    ):
        items.append(
            TaskItem(
                description="Refactor the affected implementation.",
                category="refactor",
            )
        )

    if _contains_any(
        text,
        (
            "test",
            "tests",
            "pytest",
            "verify",
            "validation",
        ),
    ):
        items.append(
            TaskItem(
                description="Verify the requested behavior with tests.",
                category="test",
            )
        )
    else:
        # Verification is a planning concern even when the user
        # does not explicitly mention tests.
        items.append(
            TaskItem(
                description="Verify the resulting behavior.",
                category="verify",
            )
        )

    return TaskPlan(
        task=normalized_task,
        items=items,
    )


def find_plan_items_by_category(
    plan: TaskPlan,
    category: str,
) -> list[TaskItem]:
    """Return plan items matching a category."""

    normalized_category = category.strip().lower()

    return [
        item
        for item in plan.items
        if item.category.lower() == normalized_category
    ]


def build_plan_report(plan: TaskPlan) -> str:
    """Build a readable report for a decomposed task."""

    lines = [
        "FLY-CODER TASK PLAN",
        "=" * 24,
        "",
        f"Task: {plan.task}",
        "",
        "Work items:",
    ]

    if not plan.items:
        lines.append("  No work items identified.")
        return "\n".join(lines)

    for index, item in enumerate(plan.items, start=1):
        lines.append(
            f"  {index}. [{item.category}] "
            f"{item.description}"
        )

    return "\n".join(lines)
