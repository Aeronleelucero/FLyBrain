"""Structured planning models for FLY-CODER."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlanStep:
    """One actionable step in a coding plan."""

    id: str
    description: str
    category: str
    affected_files: list[str] = field(default_factory=list)
    affected_symbols: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class CodePlan:
    """Structured plan produced by FLY-CODER planning stages."""

    task: str
    objective: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    impacted_files: list[str] = field(default_factory=list)
    impacted_symbols: list[str] = field(default_factory=list)
    validation_steps: list[str] = field(default_factory=list)

    def add_step(self, step: PlanStep) -> None:
        """Append a plan step."""
        self.steps.append(step)

    def add_impacted_file(self, path: str) -> None:
        """Add an impacted file without creating duplicates."""
        if path not in self.impacted_files:
            self.impacted_files.append(path)

    def add_impacted_symbol(self, name: str) -> None:
        """Add an impacted symbol without creating duplicates."""
        if name not in self.impacted_symbols:
            self.impacted_symbols.append(name)

    def add_validation_step(self, description: str) -> None:
        """Add a validation step without creating duplicates."""
        if description not in self.validation_steps:
            self.validation_steps.append(description)


def create_plan(
    task: str,
    objective: str = "",
) -> CodePlan:
    """Create a normalized empty code plan."""
    return CodePlan(
        task=" ".join(task.strip().split()),
        objective=" ".join(objective.strip().split()),
    )


def find_plan_step(
    plan: CodePlan,
    step_id: str,
) -> PlanStep | None:
    """Find a plan step by its identifier."""
    normalized_id = step_id.strip()

    for step in plan.steps:
        if step.id == normalized_id:
            return step

    return None


def find_plan_steps_by_category(
    plan: CodePlan,
    category: str,
) -> list[PlanStep]:
    """Find plan steps matching a category."""
    normalized_category = category.strip().lower()

    return [
        step
        for step in plan.steps
        if step.category.lower() == normalized_category
    ]


def validate_plan_structure(
    plan: CodePlan,
) -> list[str]:
    """
    Validate the structural integrity of a code plan.

    This checks structure only. Semantic validation and dependency
    validation belong to later Phase 5 stages.
    """
    errors: list[str] = []

    if not plan.task:
        errors.append("Plan task must not be empty.")

    step_ids: set[str] = set()

    for index, step in enumerate(plan.steps, start=1):
        if not step.id:
            errors.append(f"Step {index} must have an id.")
        elif step.id in step_ids:
            errors.append(
                f"Duplicate step id: {step.id}"
            )
        else:
            step_ids.add(step.id)

        if not step.description.strip():
            errors.append(
                f"Step {index} must have a description."
            )

        if not step.category.strip():
            errors.append(
                f"Step {index} must have a category."
            )

        for dependency in step.dependencies:
            if dependency == step.id:
                errors.append(
                    f"Step {step.id} cannot depend on itself."
                )

    known_ids = step_ids

    for step in plan.steps:
        for dependency in step.dependencies:
            if dependency not in known_ids:
                errors.append(
                    f"Step {step.id} references unknown "
                    f"dependency: {dependency}"
                )

    return errors


def build_plan_report(plan: CodePlan) -> str:
    """Build a readable structured plan report."""
    lines = [
        "FLY-CODER CODE PLAN",
        "=" * 21,
        "",
        f"Task: {plan.task}",
    ]

    if plan.objective:
        lines.extend(
            [
                f"Objective: {plan.objective}",
            ]
        )

    lines.extend(
        [
            "",
            "Steps:",
        ]
    )

    if not plan.steps:
        lines.append("  No plan steps defined.")
    else:
        for index, step in enumerate(plan.steps, start=1):
            lines.append(
                f"  {index}. [{step.id}] "
                f"[{step.category}] {step.description}"
            )

            if step.affected_files:
                lines.append(
                    "     Files: "
                    + ", ".join(step.affected_files)
                )

            if step.affected_symbols:
                lines.append(
                    "     Symbols: "
                    + ", ".join(step.affected_symbols)
                )

            if step.dependencies:
                lines.append(
                    "     Depends on: "
                    + ", ".join(step.dependencies)
                )

            if step.rationale:
                lines.append(
                    f"     Rationale: {step.rationale}"
                )

    lines.extend(
        [
            "",
            "Impacted files:",
        ]
    )

    if plan.impacted_files:
        lines.extend(
            f"  - {path}"
            for path in plan.impacted_files
        )
    else:
        lines.append("  None identified.")

    lines.extend(
        [
            "",
            "Impacted symbols:",
        ]
    )

    if plan.impacted_symbols:
        lines.extend(
            f"  - {name}"
            for name in plan.impacted_symbols
        )
    else:
        lines.append("  None identified.")

    lines.extend(
        [
            "",
            "Validation:",
        ]
    )

    if plan.validation_steps:
        lines.extend(
            f"  - {step}"
            for step in plan.validation_steps
        )
    else:
        lines.append("  None defined.")

    return "\n".join(lines)
