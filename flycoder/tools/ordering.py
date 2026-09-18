"""Dependency-aware plan ordering tools for FLY-CODER."""

from __future__ import annotations

from dataclasses import dataclass

from flycoder.tools.code_plan import CodePlan, PlanStep


@dataclass
class OrderingResult:
    """Result of dependency-aware ordering."""

    ordered_steps: list[PlanStep]
    missing_dependencies: dict[str, list[str]]
    cycles: list[list[str]]

    @property
    def valid(self) -> bool:
        """Return whether the plan can be completely ordered."""
        return not self.missing_dependencies and not self.cycles


def _step_map(plan: CodePlan) -> dict[str, PlanStep]:
    """Build a step lookup table."""
    return {step.id: step for step in plan.steps}


def find_missing_dependencies(
    plan: CodePlan,
) -> dict[str, list[str]]:
    """Find dependencies that reference nonexistent steps."""
    steps = _step_map(plan)
    missing: dict[str, list[str]] = {}

    for step in plan.steps:
        unknown = [
            dependency
            for dependency in step.dependencies
            if dependency not in steps
        ]

        if unknown:
            missing[step.id] = unknown

    return missing


def find_dependency_cycles(
    plan: CodePlan,
) -> list[list[str]]:
    """Find dependency cycles using depth-first traversal."""
    steps = _step_map(plan)
    cycles: list[list[str]] = []
    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in active_set:
            start = active.index(step_id)
            cycle = active[start:] + [step_id]

            if cycle not in cycles:
                cycles.append(cycle)

            return

        if step_id in visited:
            return

        step = steps.get(step_id)

        if step is None:
            return

        active.append(step_id)
        active_set.add(step_id)

        for dependency in step.dependencies:
            if dependency in steps:
                visit(dependency)

        active.pop()
        active_set.remove(step_id)
        visited.add(step_id)

    for step in plan.steps:
        visit(step.id)

    return cycles


def order_plan_steps(
    plan: CodePlan,
) -> OrderingResult:
    """
    Return plan steps in dependency-first order.

    Steps with no dependency are ordered first. When multiple steps
    are available at the same time, their original plan order is used
    to keep the result deterministic.
    """
    missing = find_missing_dependencies(plan)
    cycles = find_dependency_cycles(plan)

    if missing or cycles:
        return OrderingResult(
            ordered_steps=[],
            missing_dependencies=missing,
            cycles=cycles,
        )

    steps = _step_map(plan)
    remaining = {
        step.id: set(step.dependencies)
        for step in plan.steps
    }

    ordered: list[PlanStep] = []

    while remaining:
        ready = [
            step
            for step in plan.steps
            if step.id in remaining
            and not remaining[step.id]
        ]

        if not ready:
            # Defensive fallback. A cycle should already have been
            # detected above.
            break

        for step in ready:
            ordered.append(step)
            del remaining[step.id]

            for dependencies in remaining.values():
                dependencies.discard(step.id)

    return OrderingResult(
        ordered_steps=ordered,
        missing_dependencies={},
        cycles=[],
    )


def build_ordering_report(
    plan: CodePlan,
    result: OrderingResult,
) -> str:
    """Build a readable dependency-ordering report."""
    lines = [
        "FLY-CODER DEPENDENCY ORDER",
        "=" * 28,
        "",
        f"Task: {plan.task}",
        "",
    ]

    if result.missing_dependencies:
        lines.append("Missing dependencies:")

        for step_id, dependencies in result.missing_dependencies.items():
            lines.append(
                f"  - {step_id}: "
                + ", ".join(dependencies)
            )

        lines.append("")

    if result.cycles:
        lines.append("Dependency cycles:")

        for cycle in result.cycles:
            lines.append(
                "  - " + " -> ".join(cycle)
            )

        lines.append("")

    if not result.valid:
        lines.append("Plan cannot be ordered.")
        return "\n".join(lines)

    lines.append("Execution order:")

    if not result.ordered_steps:
        lines.append("  No plan steps defined.")
    else:
        for index, step in enumerate(
            result.ordered_steps,
            start=1,
        ):
            lines.append(
                f"  {index}. [{step.id}] "
                f"[{step.category}] {step.description}"
            )

    return "\n".join(lines)
