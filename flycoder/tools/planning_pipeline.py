"""Integrated planning pipeline for FLY-CODER Phase 9.1."""

from __future__ import annotations

from dataclasses import dataclass

from flycoder.state import CodingState
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
    retrieve_learning_context,
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
    strategy: StrategyDecision
    task_plan: TaskPlan
    impact: ImpactAnalysis
    plan: CodePlan
    ordering: OrderingResult
    validation: PlanValidationResult
    risk: RiskAnalysis

    @property
    def valid(self) -> bool:
        """Return whether the integrated plan is valid."""
        return self.validation.valid and self.ordering.valid

    @property
    def safe(self) -> bool:
        """Return whether the plan is valid and not high risk."""
        return self.valid and self.risk.level != "HIGH"


def _normalize_task(task: str) -> str:
    """Normalize task text consistently across the pipeline."""
    return " ".join(task.strip().split())


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

    for index, item in enumerate(task_plan.items, start=1):
        dependencies: list[str] = []

        if steps:
            dependencies.append(steps[-1].id)

        steps.append(
            PlanStep(
                id=f"step-{index}",
                description=item.description,
                category=item.category,
                affected_files=list(affected_files),
                affected_symbols=list(affected_symbols),
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
        objective=f"Complete the requested task: {task}",
    )

    for step in _build_plan_steps(task_plan, impact):
        plan.add_step(step)

    for item in impact.items:
        if item.kind == "file":
            plan.add_impacted_file(item.path)

        elif item.kind == "symbol" and item.name:
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
    modifies source files.

    When a MemoryStore is supplied, relevant previous experiences
    are retrieved as advisory learning context. Strategy selection
    is also advisory and does not execute tools or modify the plan
    automatically.
    """

    requested_task = _normalize_task(
        task if task is not None else state.task
    )

    learning = retrieve_learning_context(
        memory_store or MemoryStore(),
        requested_task,
    )

    strategy = select_strategy(
        requested_task,
        learning,
    )

    task_plan = decompose_task(requested_task)

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

    ordering = order_plan_steps(plan)

    validation = validate_code_plan(plan)

    risk = analyze_risk(plan)

    return IntegratedPlanningResult(
        task=requested_task,
        learning=learning,
        strategy=strategy,
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
        f"Ordered steps: {len(result.ordering.ordered_steps)}",
        f"Validation: {'VALID' if result.validation.valid else 'INVALID'}",
        f"Risk: {result.risk.level} ({result.risk.score})",
        "",
        "Learning:",
        f"  Relevant memories: {len(result.learning.memories)}",
        f"  Verified memories: {len(result.learning.verified_memories)}",
        f"  Previous failures: {len(result.learning.failed_memories)}",
        "",
        "Strategy:",
        f"  Selected: {result.strategy.strategy}",
        f"  Confidence: {result.strategy.confidence:.2f}",
        f"  Reason: {result.strategy.reason}",
        "",
        f"Plan executable: {'YES' if result.valid else 'NO'}",
        f"Plan safe: {'YES' if result.safe else 'NO'}",
    ]

    if result.strategy.supporting_memories:
        lines.extend(
            [
                "",
                "Supporting strategy memories:",
            ]
        )

        for memory in result.strategy.supporting_memories:
            experience = memory.experience

            lines.append(
                f"  - {experience.action}"
                f" (verified={experience.verified})"
            )

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

    if result.ordering.missing_dependencies:
        lines.extend(
            [
                "",
                "Missing dependencies:",
            ]
        )

        for step_id, dependencies in (
            result.ordering.missing_dependencies.items()
        ):
            lines.append(
                f"  - {step_id}: "
                + ", ".join(dependencies)
            )

    if result.ordering.cycles:
        lines.extend(
            [
                "",
                "Dependency cycles:",
            ]
        )

        for cycle in result.ordering.cycles:
            lines.append(
                "  - " + " -> ".join(cycle)
            )

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

    if result.risk.recommendations:
        lines.extend(
            [
                "",
                "Recommendations:",
            ]
        )

        for recommendation in result.risk.recommendations:
            lines.append(
                f"  - {recommendation}"
            )

    return "\n".join(lines)