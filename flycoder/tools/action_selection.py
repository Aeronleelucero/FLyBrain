"""Action selection tools for FLY-CODER Phase 9.2."""

from __future__ import annotations

from dataclasses import dataclass, field

from flycoder.actions import create_action_registry
from flycoder.tools.learning import LearningContext
from flycoder.tools.strategy import StrategyDecision


@dataclass
class ActionDecision:
    """Deterministic recommendation for the next agent action."""

    action: str
    reason: str
    confidence: float
    strategy: str
    risks: list[str] = field(default_factory=list)


def _available_actions() -> set[str]:
    """Return the names of actions currently registered by FLY-CODER."""

    return set(create_action_registry().list_actions())


def _contains_any(
    text: str,
    terms: tuple[str, ...],
) -> bool:
    """Return whether text contains any of the supplied terms."""

    return any(term in text for term in terms)


def _select_candidate(
    task: str,
    strategy: str,
) -> tuple[str, str, list[str]]:
    """Select a deterministic action candidate from the task and strategy."""

    task_lower = task.lower()

    if strategy == "repair":
        if _contains_any(
            task_lower,
            (
                "error",
                "failure",
                "failing",
                "broken",
                "bug",
            ),
        ):
            return (
                "explain_error",
                (
                    "The repair strategy requires understanding "
                    "the reported failure before proposing a fix."
                ),
                [
                    "The reported error should be inspected before "
                    "any repair is proposed."
                ],
            )

        return (
            "inspect_files",
            (
                "The repair strategy should begin by inspecting "
                "the existing workspace."
            ),
            [
                "The root cause should be confirmed before modification."
            ],
        )

    if strategy == "refactor":
        return (
            "inspect_files",
            (
                "Refactoring should begin with inspection of the "
                "existing workspace."
            ),
            [
                "Existing behavior and dependencies must be understood "
                "before structural changes."
            ],
        )

    if strategy == "test_first":
        return (
            "run_tests",
            (
                "The selected strategy explicitly prioritizes "
                "testing or verification."
            ),
            [
                "Test execution must occur before interpreting "
                "the resulting behavior."
            ],
        )

    if strategy == "create_new":
        return (
            "inspect_files",
            (
                "New functionality should begin by inspecting "
                "the existing workspace."
            ),
            [
                "Existing architecture and dependencies should be "
                "understood before creating code."
            ],
        )

    if strategy == "modify_existing":
        return (
            "inspect_files",
            (
                "Existing functionality should be inspected before "
                "it is modified."
            ),
            [
                "Existing behavior and dependencies should be checked "
                "before modification."
            ],
        )

    if strategy == "research_first":
        return (
            "inspect_files",
            (
                "Research should begin with inspection of the local "
                "workspace and available implementation context."
            ),
            [
                "External research may still be required if local "
                "information is insufficient."
            ],
        )

    return (
        "inspect_files",
        (
            "The task does not provide enough evidence for a more "
            "specific action, so workspace inspection is the safest "
            "starting point."
        ),
        [
            "Additional context may be required before taking a "
            "more specific action."
        ],
    )


def _apply_task_specific_action(
    task: str,
    strategy: str,
    candidate: str,
) -> tuple[str, str, list[str]]:
    """Refine an action candidate using explicit task evidence."""

    task_lower = task.lower()

    if strategy == "repair" and _contains_any(
        task_lower,
        (
            "error",
            "failure",
            "failing",
            "broken",
            "bug",
        ),
    ):
        return (
            "explain_error",
            (
                "The task identifies a failure condition, so the "
                "existing error should be explained before repair."
            ),
            [
                "Do not modify code until the failure is understood."
            ],
        )

    if _contains_any(
        task_lower,
        (
            "test",
            "testing",
            "verify",
            "verification",
        ),
    ):
        return (
            "run_tests",
            (
                "The task explicitly requests testing or verification."
            ),
            [
                "Test results should be analyzed before deciding "
                "whether further changes are needed."
            ],
        )

    return (
        candidate,
        "",
        [],
    )


def _memory_support(
    context: LearningContext,
    action: str,
) -> int:
    """Count relevant previous experiences for the selected action."""

    count = 0

    action_terms = {
        "inspect_files": (
            "inspect",
            "review",
            "analyze",
            "explore",
        ),
        "read_file": (
            "read",
            "inspect",
        ),
        "write_code": (
            "write",
            "create",
            "implement",
            "add",
        ),
        "run_tests": (
            "test",
            "pytest",
            "verify",
        ),
        "fix_error": (
            "fix",
            "repair",
            "bug",
            "error",
        ),
        "propose_repair": (
            "repair",
            "proposal",
            "fix",
        ),
        "explain_error": (
            "explain",
            "error",
            "failure",
        ),
        "review_code": (
            "review",
            "audit",
        ),
        "improve_code": (
            "improve",
            "optimize",
            "clean",
        ),
    }

    terms = action_terms.get(action, ())

    for memory in context.memories:
        recorded_action = memory.experience.action.lower()

        if _contains_any(recorded_action, terms):
            count += 1

    return count


def select_action(
    task: str,
    strategy: StrategyDecision,
    context: LearningContext | None = None,
) -> ActionDecision:
    """
    Select a deterministic action for a previously selected strategy.

    Action selection is advisory only. It does not execute actions,
    modify files, mutate state, or mutate learning memory.
    """

    normalized_task = " ".join(task.strip().split())

    available = _available_actions()

    if not normalized_task:
        action = "inspect_files"

        if action not in available:
            return ActionDecision(
                action=action,
                reason=(
                    "No task was supplied and the inspection action "
                    "is unavailable."
                ),
                confidence=0.0,
                strategy=strategy.strategy,
                risks=[
                    "A concrete task and an available inspection "
                    "action are required."
                ],
            )

        return ActionDecision(
            action=action,
            reason=(
                "No task was supplied, so workspace inspection is "
                "the safest available starting action."
            ),
            confidence=1.0,
            strategy=strategy.strategy,
            risks=[
                "A concrete task is required before modification."
            ],
        )

    candidate, reason, risks = _select_candidate(
        normalized_task,
        strategy.strategy,
    )

    candidate, refined_reason, refined_risks = _apply_task_specific_action(
        normalized_task,
        strategy.strategy,
        candidate,
    )

    if refined_reason:
        reason = refined_reason

    if refined_risks:
        risks = refined_risks

    if candidate not in available:
        fallback = "inspect_files"

        return ActionDecision(
            action=fallback,
            reason=(
                f"The preferred action '{candidate}' is not registered, "
                "so the selector falls back to workspace inspection."
            ),
            confidence=0.35,
            strategy=strategy.strategy,
            risks=[
                "The preferred action is unavailable.",
                "The fallback action should be reviewed before execution.",
            ],
        )

    learning = context or LearningContext(task=normalized_task)

    memory_count = _memory_support(
        learning,
        candidate,
    )

    confidence = min(
        1.0,
        0.75
        + (0.05 * memory_count)
        + (0.05 * strategy.confidence),
    )

    if memory_count:
        reason += (
            " Previous experience supports this action."
        )

    return ActionDecision(
        action=candidate,
        reason=reason,
        confidence=confidence,
        strategy=strategy.strategy,
        risks=risks,
    )
