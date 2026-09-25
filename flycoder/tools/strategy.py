"""Strategy selection tools for FLY-CODER."""

from __future__ import annotations

from dataclasses import dataclass, field

from flycoder.tools.learning import (
    LearningContext,
    LearningMemory,
    build_learning_guidance,
)
from flycoder.tools.learning_validation import (
    LearningValidationResult,
    build_validation_guidance,
    validate_learning_context,
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
    learning_guidance: list[str] = field(
        default_factory=list
    )
    validation_guidance: list[str] = field(
        default_factory=list
    )
    trusted_memory_count: int = 0
    rejected_memory_count: int = 0
    risks: list[str] = field(default_factory=list)


def _normalize_task(task: str) -> str:
    """Normalize task text consistently."""
    return " ".join(task.strip().split())


def _contains_any(
    task: str,
    terms: tuple[str, ...],
) -> bool:
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


def _validated_support(
    context: LearningContext,
    strategy: str,
    validation: LearningValidationResult,
) -> list[LearningMemory]:
    """
    Return strategy-supporting memories that were not rejected.

    LOW-trust memories remain available as cautionary evidence.
    Rejected memories are excluded entirely.
    """

    rejected = {
        id(memory.memory.experience)
        for memory in validation.rejected_memories
    }

    supporting = _memory_support(
        context,
        strategy,
    )

    return [
        memory
        for memory in supporting
        if id(memory.experience) not in rejected
    ]


def select_strategy(
    task: str,
    context: LearningContext | None = None,
) -> StrategyDecision:
    """
    Select a deterministic coding strategy for a task.

    Strategy selection is advisory only. It does not modify files,
    execute tools, mutate state, or grant execution permission.

    Learning validation is also advisory. It excludes structurally
    invalid memories while preserving low-trust memories as cautionary
    evidence.
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

    learning = (
        context
        if context is not None
        else LearningContext(task=normalized_task)
    )

    guidance = build_learning_guidance(
        learning
    )

    validation = validate_learning_context(
        learning
    )

    validation_guidance = build_validation_guidance(
        validation
    )

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

    supporting = _validated_support(
        learning,
        strategy,
        validation,
    )

    high_trust_support = [
        memory
        for memory in supporting
        if any(
            validated.memory is memory
            and validated.trust_level == "HIGH"
            for validated in validation.valid_memories
        )
    ]

    medium_trust_support = [
        memory
        for memory in supporting
        if any(
            validated.memory is memory
            and validated.trust_level == "MEDIUM"
            for validated in validation.valid_memories
        )
    ]

    if high_trust_support:
        confidence = min(
            1.0,
            0.75 + (0.05 * len(high_trust_support)),
        )
        reason += (
            " Similar verified experience supports this strategy."
        )

    elif medium_trust_support:
        confidence = min(
            1.0,
            0.60 + (0.05 * len(medium_trust_support)),
        )
        reason += (
            " Similar moderate-trust experience supports this strategy."
        )

    elif supporting:
        confidence = 0.55
        reason += (
            " Similar low-trust experience provides cautionary context."
        )

    else:
        confidence = 0.55

    if validation.rejected_memories:
        risks.append(
            "Some retrieved learning memories were rejected "
            "during validation."
        )

    if validation.low_trust_memories:
        risks.append(
            "Some learning evidence is low trust and should "
            "be treated cautiously."
        )

    return StrategyDecision(
        strategy=strategy,
        reason=reason,
        confidence=confidence,
        supporting_memories=supporting,
        learning_guidance=guidance,
        validation_guidance=validation_guidance,
        trusted_memory_count=len(
            validation.trusted_memories
        ),
        rejected_memory_count=len(
            validation.rejected_memories
        ),
        risks=risks,
    )
