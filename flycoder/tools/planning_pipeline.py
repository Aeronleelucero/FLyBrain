"""Integrated planning pipeline for FLY-CODER Phase 9.2."""

from __future__ import annotations

from dataclasses import dataclass

from flycoder.state import CodingState
from flycoder.tools.action_selection import (
    ActionDecision,
    select_action,
)
from flycoder.tools.code_plan import (
    CodePlan,
    PlanStep,
    create_plan,
)
from flycoder.tools.filesystem import Workspace
from flycoder.tools.impact import (
    ImpactAnalysis,
    analyze_impact,
)
from flycoder.tools.learning import (
    LearningContext,
    build_learning_guidance,
    retrieve_learning_context,
)
from flycoder.tools.learning_validation import (
    LearningValidationResult,
    build_validation_guidance,
    validate_learning_context,
)
from flycoder.tools.memory import MemoryStore
from flycoder.tools.ordering import (
    OrderingResult,
    order_plan_steps,
)
from flycoder.tools.plan_validation import (
    PlanValidationResult,
    validate_code_plan,
)
from flycoder.tools.planning import (
    TaskPlan,
    decompose_task,
)
from flycoder.tools.risk import (
    RiskAnalysis,
    analyze_risk,
)
from flycoder.tools.strategy import (
    StrategyDecision,
    select_strategy,
)


@dataclass
class IntegratedPlanningResult:
    """Complete result produced by the integrated planning pipeline."""

    task: str
    learning: LearningContext
    learning_validation: LearningValidationResult
    strategy: StrategyDecision
    action: ActionDecision
    task_plan: TaskPlan
    impact: ImpactAnalysis
    plan: CodePlan
    ordering: OrderingResult
    validation: PlanValidationResult
    risk: RiskAnalysis

    @property
    def valid(self) -> bool:
        """Return whether the integrated plan is valid."""
        return (
            self.validation.valid
            and self.ordering.valid
        )

    @property
    def safe(self) -> bool:
        """Return whether the plan is valid and not high risk."""
        return (
            self.valid
            and self.risk.level != "HIGH"
        )


def _normalize_task(task: str) -> str:
    """Normalize task text consistently across the pipeline."""
    return " ".join(
        task.strip().split()
    )


def _build_plan_steps(
    task_plan: TaskPlan,
    impact: ImpactAnalysis,
) -> list[PlanStep]:
    """Convert decomposed task items into structured plan steps."""

    affected_files = [
        item.path
        for item in impact.items
        if item.kind == "file"
    ]

    affected_symbols = [
        item.name
        for item in impact.items
        if item.kind == "symbol"
        and item.name
    ]

    steps: list[PlanStep] = []

    for index, item in enumerate(
        task_plan.items,
        start=1,
    ):
        dependencies: list[str] = []

        if steps:
            dependencies.append(
                steps[-1].id
            )

        steps.append(
            PlanStep(
                id=f"step-{index}",
                description=item.description,
                category=item.category,
                affected_files=list(
                    affected_files
                ),
                affected_symbols=list(
                    affected_symbols
                ),
                dependencies=dependencies,
                rationale=(
                    "Generated from task decomposition and "
                    "workspace impact analysis."
                ),
            )
        )

    return steps


def _build_code_plan(
    task: str,
    task_plan: TaskPlan,
    impact: ImpactAnalysis,
) -> CodePlan:
    """Build a structured CodePlan from earlier planning stages."""

    plan = create_plan(
        task=task,
        objective=(
            f"Complete the requested task: {task}"
        ),
    )

    for step in _build_plan_steps(
        task_plan,
        impact,
    ):
        plan.add_step(step)

    for item in impact.items:
        if item.kind == "file":
            plan.add_impacted_file(
                item.path
            )

        elif (
            item.kind == "symbol"
            and item.name
        ):
            plan.add_impacted_symbol(
                f"{item.path}:{item.name}"
            )

    plan.add_validation_step(
        "Run the relevant automated tests."
    )

    plan.add_validation_step(
        "Review the final changes for correctness and regressions."
    )

    return plan


def create_integrated_plan(
    workspace: Workspace,
    state: CodingState,
    task: str | None = None,
    *,
    memory_store: MemoryStore | None = None,
) -> IntegratedPlanningResult:
    """
    Run the complete planning pipeline.

    This function performs planning and analysis only. It never
    modifies source files or executes registered actions.

    Learning, strategy selection, and action selection are advisory.
    They do not automatically execute tools, modify files, grant
    approval, or bypass execution guardrails.
    """

    requested_task = _normalize_task(
        task
        if task is not None
        else state.task
    )

    # --------------------------------------------------------------
    # Phase 9.7.4 — Learning-aware planning
    #
    # Use the caller's MemoryStore when one is supplied.
    #
    # IMPORTANT:
    # Do not use:
    #
    #     memory_store or MemoryStore()
    #
    # because an empty MemoryStore can evaluate as falsey.
    # That would silently replace the caller's store.
    # --------------------------------------------------------------

    active_memory_store = (
        memory_store
        if memory_store is not None
        else MemoryStore()
    )

    learning = retrieve_learning_context(
        active_memory_store,
        requested_task,
    )

    # --------------------------------------------------------------
    # Phase 9.7.5 — Learning validation
#
    # Validate retrieved learning before it influences planning.
    # Validation is advisory and does not mutate the memory store.
    # --------------------------------------------------------------

    learning_validation = validate_learning_context(
        learning
    )

    # --------------------------------------------------------------
    # Strategy selection receives the retrieved learning context.
    #
    # Learning is advisory evidence only.
    # --------------------------------------------------------------

    strategy = select_strategy(
        requested_task,
        learning,
    )

    # --------------------------------------------------------------
    # Action selection receives the same learning context.
    #
    # Learning can inform the recommendation but cannot authorize
    # execution.
    # --------------------------------------------------------------

    action = select_action(
        requested_task,
        strategy,
        learning,
    )

    # --------------------------------------------------------------
    # Continue with deterministic task planning.
    # --------------------------------------------------------------

    task_plan = decompose_task(
        requested_task
    )

    impact = analyze_impact(
        requested_task,
        workspace,
        symbols=state.symbols,
        symbol_graph=state.symbol_graph,
    )

    plan = _build_code_plan(
        requested_task,
        task_plan,
        impact,
    )

    ordering = order_plan_steps(
        plan
    )

    validation = validate_code_plan(
        plan
    )

    risk = analyze_risk(
        plan
    )

    return IntegratedPlanningResult(
        task=requested_task,
        learning=learning,
        learning_validation=learning_validation,
        strategy=strategy,
        action=action,
        task_plan=task_plan,
        impact=impact,
        plan=plan,
        ordering=ordering,
        validation=validation,
        risk=risk,
    )


def build_planning_report(
    result: IntegratedPlanningResult,
) -> str:
    """Build a readable report for the complete planning pipeline."""

    lines = [
        "FLY-CODER INTEGRATED PLANNING",
        "=" * 31,
        "",
        f"Task: {result.task}",
        "",
        f"Task items: {len(result.task_plan.items)}",
        f"Impact items: {len(result.impact.items)}",
        f"Plan steps: {len(result.plan.steps)}",
        (
            "Ordered steps: "
            f"{len(result.ordering.ordered_steps)}"
        ),
        (
            "Validation: "
            f"{'VALID' if result.validation.valid else 'INVALID'}"
        ),
        (
            "Risk: "
            f"{result.risk.level} "
            f"({result.risk.score})"
        ),
        "",
        "Learning:",
        (
            "  Relevant memories: "
            f"{len(result.learning.memories)}"
        ),
        (
            "  Verified memories: "
            f"{len(result.learning.verified_memories)}"
        ),
        (
            "  Previous failures: "
            f"{len(result.learning.failed_memories)}"
        ),
        (
            "  Trusted memories: "
            f"{len(result.learning_validation.trusted_memories)}"
        ),
        (
            "  Rejected memories: "
            f"{len(result.learning_validation.rejected_memories)}"
        ),
    ]

    # --------------------------------------------------------------
    # Learning guidance
    # --------------------------------------------------------------

    learning_guidance = build_learning_guidance(
        result.learning
    )

    if learning_guidance:
        lines.extend(
            [
                "",
                "Learning guidance:",
            ]
        )

        for guidance in learning_guidance:
            lines.append(
                f"  - {guidance}"
            )
    else:
        lines.extend(
            [
                "",
                "Learning guidance:",
                "  - No relevant previous experience found.",
            ]
        )

    # --------------------------------------------------------------
    # Learning validation guidance
    # --------------------------------------------------------------

    validation_guidance = build_validation_guidance(
        result.learning_validation
    )

    if validation_guidance:
        lines.extend(
            [
                "",
                "Learning validation:",
            ]
        )

        for guidance in validation_guidance:
            lines.append(
                f"  - {guidance}"
            )
    else:
        lines.extend(
            [
                "",
                "Learning validation:",
                "  - No validation warnings or trust guidance.",
            ]
        )

    # --------------------------------------------------------------
    # Strategy
    # --------------------------------------------------------------

    lines.extend(
        [
            "",
            "Strategy:",
            (
                "  Selected: "
                f"{result.strategy.strategy}"
            ),
            (
                "  Confidence: "
                f"{result.strategy.confidence:.2f}"
            ),
            (
                "  Reason: "
                f"{result.strategy.reason}"
            ),
        ]
    )

    # --------------------------------------------------------------
    # Action
    # --------------------------------------------------------------

    lines.extend(
        [
            "",
            "Action:",
            (
                "  Selected: "
                f"{result.action.action}"
            ),
            (
                "  Confidence: "
                f"{result.action.confidence:.2f}"
            ),
            (
                "  Reason: "
                f"{result.action.reason}"
            ),
        ]
    )

    lines.extend(
        [
            "",
            (
                "Plan executable: "
                f"{'YES' if result.valid else 'NO'}"
            ),
            (
                "Plan safe: "
                f"{'YES' if result.safe else 'NO'}"
            ),
        ]
    )

    # --------------------------------------------------------------
    # Action risks
    # --------------------------------------------------------------

    if result.action.risks:
        lines.extend(
            [
                "",
                "Action risks:",
            ]
        )

        for risk in result.action.risks:
            lines.append(
                f"  - {risk}"
            )

    # --------------------------------------------------------------
    # Action learning guidance
    # --------------------------------------------------------------

    if result.action.learning_guidance:
        lines.extend(
            [
                "",
                "Action learning guidance:",
            ]
        )

        for guidance in (
            result.action.learning_guidance
        ):
            lines.append(
                f"  - {guidance}"
            )

    # --------------------------------------------------------------
    # Strategy supporting memories
    # --------------------------------------------------------------

    if result.strategy.supporting_memories:
        lines.extend(
            [
                "",
                "Supporting strategy memories:",
            ]
        )

        for memory in (
            result.strategy.supporting_memories
        ):
            experience = memory.experience

            lines.append(
                f"  - {experience.action}"
                f" (verified={experience.verified})"
            )

    # --------------------------------------------------------------
    # Strategy learning guidance
    # --------------------------------------------------------------

    if result.strategy.learning_guidance:
        lines.extend(
            [
                "",
                "Strategy learning guidance:",
            ]
        )

        for guidance in (
            result.strategy.learning_guidance
        ):
            lines.append(
                f"  - {guidance}"
            )

    # --------------------------------------------------------------
    # Strategy risks
    # --------------------------------------------------------------

    if result.strategy.risks:
        lines.extend(
            [
                "",
                "Strategy risks:",
            ]
        )

        for risk in result.strategy.risks:
            lines.append(
                f"  - {risk}"
            )

    # --------------------------------------------------------------
    # Missing dependencies
    # --------------------------------------------------------------

    if result.ordering.missing_dependencies:
        lines.extend(
            [
                "",
                "Missing dependencies:",
            ]
        )

        for (
            step_id,
            dependencies,
        ) in result.ordering.missing_dependencies.items():
            lines.append(
                f"  - {step_id}: "
                + ", ".join(dependencies)
            )

    # --------------------------------------------------------------
    # Dependency cycles
    # --------------------------------------------------------------

    if result.ordering.cycles:
        lines.extend(
            [
                "",
                "Dependency cycles:",
            ]
        )

        for cycle in result.ordering.cycles:
            lines.append(
                "  - "
                + " -> ".join(cycle)
            )

    # --------------------------------------------------------------
    # Validation issues
    # --------------------------------------------------------------

    if result.validation.issues:
        lines.extend(
            [
                "",
                "Validation issues:",
            ]
        )

        for issue in result.validation.issues:
            location = (
                f" [{issue.step_id}]"
                if issue.step_id
                else ""
            )

            lines.append(
                f"  - [{issue.severity}] "
                f"{issue.code}{location}: "
                f"{issue.message}"
            )

    # --------------------------------------------------------------
    # Risk factors
    # --------------------------------------------------------------

    if result.risk.factors:
        lines.extend(
            [
                "",
                "Risk factors:",
            ]
        )

        for factor in result.risk.factors:
            lines.append(
                f"  - [{factor.code}] "
                f"+{factor.score}: "
                f"{factor.description}"
            )

    # --------------------------------------------------------------
    # Risk recommendations
    # --------------------------------------------------------------

    if result.risk.recommendations:
        lines.extend(
            [
                "",
                "Recommendations:",
            ]
        )

        for recommendation in (
            result.risk.recommendations
        ):
            lines.append(
                f"  - {recommendation}"
            )

    return "\n".join(lines)
