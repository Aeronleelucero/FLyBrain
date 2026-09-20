"""Strategy selection tools for FLY-CODER."""

from __future__ import annotations

from dataclasses import dataclass, field

from flycoder.tools.learning import (
    LearningContext,
    LearningMemory,
)


STRATEGIES = (
    "inspect_only",
    "modify_existing",
    "create_new",
    "refactor",
    "repair",
    "test_first",
    "research_first",
)


@dataclass
class StrategyDecision:
    """Deterministic strategy recommendation for a coding task."""

    strategy: str
    reason: str
    confidence: float
    supporting_memories: list[LearningMemory] = field(
        default_factory=list
    )
    risks: list[str] = field(default_factory=list)


def _normalize_task(task: str) -> str:
    """Normalize task text consistently."""
    return " ".join(task.strip().split())


def _contains_any(task: str, terms: tuple[str, ...]) -> bool:
    """Return whether normalized task text contains any term."""
    return any(term in task for term in terms)


def _memory_support(
    context: LearningContext,
    strategy: str,
) -> list[LearningMemory]:
    """Return memories whose recorded action supports a strategy."""

    matches: list[LearningMemory] = []

    for memory in context.memories:
        action = memory.experience.action.lower()

        if strategy == "repair" and _contains_any(
            action,
            ("repair", "fix", "bug", "error"),
        ):
            matches.append(memory)

        elif strategy == "refactor" and _contains_any(
            action,
            ("refactor", "restructure", "clean up"),
        ):
            matches.append(memory)

        elif strategy == "create_new" and _contains_any(
            action,
            ("create", "add", "implement", "new"),
        ):
            matches.append(memory)

        elif strategy == "modify_existing" and _contains_any(
            action,
            ("update", "modify", "change", "edit"),
        ):
            matches.append(memory)

        elif strategy == "test_first" and _contains_any(
            action,
            ("test", "pytest", "verification", "verify"),
        ):
            matches.append(memory)

    return matches


def select_strategy(
    task: str,
    context: LearningContext | None = None,
) -> StrategyDecision:
    """
    Select a deterministic coding strategy for a task.

    Strategy selection is advisory only. It does not modify files,
    execute tools, or mutate the supplied learning context.
    """

    normalized_task = _normalize_task(task)

    if not normalized_task:
        return StrategyDecision(
            strategy="inspect_only",
            reason=(
                "No task was provided, so inspection is "
                "the safest starting point."
            ),
            confidence=1.0,
            risks=[
                "A concrete task is required before modification."
            ],
        )

    learning = context or LearningContext(task=normalized_task)
    task_lower = normalized_task.lower()

    if _contains_any(
        task_lower,
        (
            "fix",
            "bug",
            "error",
            "broken",
            "failure",
            "failing",
            "repair",
        ),
    ):
        strategy = "repair"
        reason = (
            "The task explicitly describes a defect or failed behavior."
        )
        risks = [
            "The root cause should be confirmed before changing code."
        ]

    elif _contains_any(
        task_lower,
        (
            "refactor",
            "restructure",
            "clean up",
            "cleanup",
        ),
    ):
        strategy = "refactor"
        reason = (
            "The task explicitly requests structural improvement."
        )
        risks = [
            "Behavior should remain unchanged unless the task "
            "requires otherwise."
        ]

    elif _contains_any(
        task_lower,
        (
            "test",
            "testing",
            "verify",
            "verification",
        ),
    ):
        strategy = "test_first"
        reason = (
            "The task explicitly focuses on testing or verification."
        )
        risks = [
            "Existing behavior should be understood before "
            "changing tests."
        ]

    elif _contains_any(
        task_lower,
        (
            "create",
            "add",
            "new",
            "build",
            "implement",
        ),
    ):
        strategy = "create_new"
        reason = (
            "The task describes creating or implementing "
            "new functionality."
        )
        risks = [
            "Existing architecture and dependencies should "
            "be inspected first."
        ]

    elif _contains_any(
        task_lower,
        (
            "update",
            "modify",
            "change",
            "edit",
        ),
    ):
        strategy = "modify_existing"
        reason = (
            "The task describes changing existing functionality."
        )
        risks = [
            "Existing behavior and affected dependencies "
            "should be checked."
        ]

    else:
        strategy = "inspect_only"
        reason = (
            "The task does not provide enough evidence for a "
            "more specific modification strategy."
        )
        risks = [
            "Additional workspace inspection may be required."
        ]

    supporting = _memory_support(
        learning,
        strategy,
    )

    if supporting:
        verified_support = [
            memory
            for memory in supporting
            if memory.experience.verified
        ]

        if verified_support:
            confidence = min(
                1.0,
                0.75 + (0.05 * len(verified_support)),
            )
            reason += (
                " Similar verified experience supports this strategy."
            )
        else:
            confidence = min(
                1.0,
                0.60 + (0.05 * len(supporting)),
            )
            reason += (
                " Similar previous experience supports this strategy."
            )
    else:
        confidence = 0.55

    return StrategyDecision(
        strategy=strategy,
        reason=reason,
        confidence=confidence,
        supporting_memories=supporting,
        risks=risks,
    )
