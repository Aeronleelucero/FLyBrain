"""Action selection tools for FLY-CODER Phase 9.2."""

from __future__ import annotations

from dataclasses import dataclass, field

from flycoder.actions import create_action_registry
from flycoder.tools.learning import (
    LearningContext,
    build_learning_guidance,
)
from flycoder.tools.learning_validation import (
    LearningValidationResult,
    build_validation_guidance,
    validate_learning_context,
)
from flycoder.tools.strategy import StrategyDecision


@dataclass
class ActionDecision:
    """Deterministic recommendation for the next agent action."""

    action: str
    reason: str
    confidence: float
    strategy: str
    risks: list[str] = field(default_factory=list)
    learning_guidance: list[str] = field(
        default_factory=list
    )
    validation_guidance: list[str] = field(
        default_factory=list
    )
    trusted_memory_count: int = 0
    rejected_memory_count: int = 0


def _available_actions() -> set[str]:
    """Return the names of actions currently registered by FLY-CODER."""

    return set(
        create_action_registry().list_actions()
    )


def _contains_any(
    text: str,
    terms: tuple[str, ...],
) -> bool:
    """Return whether text contains any of the supplied terms."""

    return any(
        term in text
        for term in terms
    )


def _select_candidate(
    task: str,
    strategy: str,
) -> tuple[str, str, list[str]]:
    """Select a deterministic action candidate from task and strategy."""

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

    terms = action_terms.get(
        action,
        (),
    )

    for memory in context.memories:
        recorded_action = (
            memory.experience.action.lower()
        )

        if _contains_any(
            recorded_action,
            terms,
        ):
            count += 1

    return count


def _validated_memory_support(
    context: LearningContext,
    action: str,
    validation: LearningValidationResult,
) -> tuple[int, int]:
    """
    Count usable and trusted memories supporting an action.

    Rejected memories are excluded. Low-trust memories remain
    cautionary evidence and are therefore still counted as usable
    support, but they do not increase confidence.
    """

    rejected = {
        id(memory.memory.experience)
        for memory in validation.rejected_memories
    }

    support_count = 0
    trusted_count = 0

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

    terms = action_terms.get(
        action,
        (),
    )

    for memory in context.memories:
        if id(memory.experience) in rejected:
            continue

        recorded_action = (
            memory.experience.action.lower()
        )

        if not _contains_any(
            recorded_action,
            terms,
        ):
            continue

        support_count += 1

        if any(
            validated.memory is memory
            and validated.trust_level in {
                "HIGH",
                "MEDIUM",
            }
            for validated in validation.valid_memories
        ):
            trusted_count += 1

    return support_count, trusted_count


def select_action(
    task: str,
    strategy: StrategyDecision,
    context: LearningContext | None = None,
) -> ActionDecision:
    """
    Select a deterministic action for a previously selected strategy.

    Action selection is advisory only. It does not execute actions,
    modify files, mutate state, or grant execution permission.

    Learning validation excludes invalid memories and prevents
    low-trust evidence from increasing confidence.
    """

    normalized_task = " ".join(
        task.strip().split()
    )

    available = _available_actions()

    learning = (
        context
        if context is not None
        else LearningContext(
            task=normalized_task
        )
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
                learning_guidance=guidance,
                validation_guidance=validation_guidance,
                trusted_memory_count=len(
                    validation.trusted_memories
                ),
                rejected_memory_count=len(
                    validation.rejected_memories
                ),
            )

        return ActionDecision(
            action=action,
            reason=(
                "No task was supplied, so inspection is the safest "
                "available action."
            ),
            confidence=1.0,
            strategy=strategy.strategy,
            risks=[
                "A concrete task is required before modification."
            ],
            learning_guidance=guidance,
            validation_guidance=validation_guidance,
            trusted_memory_count=len(
                validation.trusted_memories
            ),
            rejected_memory_count=len(
                validation.rejected_memories
            ),
        )

    candidate, reason, risks = _select_candidate(
        normalized_task,
        strategy.strategy,
    )

    candidate, specific_reason, specific_risks = (
        _apply_task_specific_action(
            normalized_task,
            strategy.strategy,
            candidate,
        )
    )

    if specific_reason:
        reason = specific_reason

    if specific_risks:
        risks = specific_risks

    if candidate not in available:
        return ActionDecision(
            action=candidate,
            reason=(
                "The selected action is not currently registered."
            ),
            confidence=0.0,
            strategy=strategy.strategy,
            risks=[
                "The action must be registered before execution."
            ],
            learning_guidance=guidance,
            validation_guidance=validation_guidance,
            trusted_memory_count=len(
                validation.trusted_memories
            ),
            rejected_memory_count=len(
                validation.rejected_memories
            ),
        )

    support_count, trusted_support_count = (
        _validated_memory_support(
            learning,
            candidate,
            validation,
        )
    )

    confidence = min(
        1.0,
        0.55
        + (
            0.05
            * trusted_support_count
        ),
    )

    if trusted_support_count:
        reason += (
            " Previous experience supports this action."
        )

    elif support_count:
        reason += (
            " Previous low-trust experience provides cautionary context."
        )

    if guidance:
        reason += (
            " Previous execution outcomes were included "
            "as advisory learning context."
        )

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

    return ActionDecision(
        action=candidate,
        reason=reason,
        confidence=confidence,
        strategy=strategy.strategy,
        risks=risks,
        learning_guidance=guidance,
        validation_guidance=validation_guidance,
        trusted_memory_count=len(
            validation.trusted_memories
        ),
        rejected_memory_count=len(
            validation.rejected_memories
        ),
    )
