"""Validation tools for FLY-CODER code plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

from flycoder.tools.code_plan import (
    CodePlan,
    PlanStep,
    validate_plan_structure,
)
from flycoder.tools.ordering import order_plan_steps


@dataclass
class PlanValidationIssue:
    """A single plan validation issue."""

    severity: str
    code: str
    message: str
    step_id: str | None = None


@dataclass
class PlanValidationResult:
    """Result of validating a CodePlan."""

    valid: bool
    issues: list[PlanValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[PlanValidationIssue]:
        """Return validation errors."""
        return [
            issue
            for issue in self.issues
            if issue.severity == "error"
        ]

    @property
    def warnings(self) -> list[PlanValidationIssue]:
        """Return validation warnings."""
        return [
            issue
            for issue in self.issues
            if issue.severity == "warning"
        ]


def _issue(
    severity: str,
    code: str,
    message: str,
    step_id: str | None = None,
) -> PlanValidationIssue:
    return PlanValidationIssue(
        severity=severity,
        code=code,
        message=message,
        step_id=step_id,
    )


def _normalize_path(path: str) -> str:
    """Normalize a plan path to POSIX form."""
    return PurePosixPath(path.replace("\\", "/")).as_posix()


def _validate_steps(plan: CodePlan) -> list[PlanValidationIssue]:
    """Validate individual plan steps."""
    issues: list[PlanValidationIssue] = []

    for step in plan.steps:
        if not step.id.strip():
            issues.append(
                _issue(
                    "error",
                    "empty_step_id",
                    "Plan step has an empty ID.",
                )
            )

        if not step.description.strip():
            issues.append(
                _issue(
                    "error",
                    "empty_description",
                    "Plan step has an empty description.",
                    step.id or None,
                )
            )

        if not step.category.strip():
            issues.append(
                _issue(
                    "error",
                    "empty_category",
                    "Plan step has an empty category.",
                    step.id or None,
                )
            )

        for path in step.affected_files:
            normalized = _normalize_path(path)

            if not normalized.strip() or normalized == ".":
                issues.append(
                    _issue(
                        "error",
                        "invalid_affected_file",
                        "Plan step contains an invalid affected file.",
                        step.id or None,
                    )
                )

        for symbol in step.affected_symbols:
            if not symbol.strip():
                issues.append(
                    _issue(
                        "error",
                        "invalid_affected_symbol",
                        "Plan step contains an empty affected symbol.",
                        step.id or None,
                    )
                )

    return issues


def _validate_impact_consistency(
    plan: CodePlan,
) -> list[PlanValidationIssue]:
    """Check that step impacts are represented by the plan-level impact."""
    issues: list[PlanValidationIssue] = []

    impacted_files = {
        _normalize_path(path)
        for path in plan.impacted_files
        if path.strip()
    }

    impacted_symbols = {
        symbol.strip()
        for symbol in plan.impacted_symbols
        if symbol.strip()
    }

    for step in plan.steps:
        for path in step.affected_files:
            normalized = _normalize_path(path)

            if normalized not in impacted_files:
                issues.append(
                    _issue(
                        "error",
                        "missing_impacted_file",
                        (
                            f"Affected file '{path}' from step "
                            f"'{step.id}' is missing from plan.impacted_files."
                        ),
                        step.id or None,
                    )
                )

        for symbol in step.affected_symbols:
            normalized = symbol.strip()

            if normalized not in impacted_symbols:
                issues.append(
                    _issue(
                        "error",
                        "missing_impacted_symbol",
                        (
                            f"Affected symbol '{symbol}' from step "
                            f"'{step.id}' is missing from plan.impacted_symbols."
                        ),
                        step.id or None,
                    )
                )

    return issues


def _validate_impact_entries(
    plan: CodePlan,
) -> list[PlanValidationIssue]:
    """Validate plan-level impact entries."""
    issues: list[PlanValidationIssue] = []

    for path in plan.impacted_files:
        normalized = _normalize_path(path)

        if not path.strip() or normalized == ".":
            issues.append(
                _issue(
                    "error",
                    "invalid_impacted_file",
                    "Plan contains an invalid impacted file.",
                )
            )

    for symbol in plan.impacted_symbols:
        if not symbol.strip():
            issues.append(
                _issue(
                    "error",
                    "invalid_impacted_symbol",
                    "Plan contains an empty impacted symbol.",
                )
            )

    return issues


def _validate_dependencies(
    plan: CodePlan,
) -> list[PlanValidationIssue]:
    """Validate dependency references and cycles."""
    issues: list[PlanValidationIssue] = []

    ordering = order_plan_steps(plan)

    for step_id, dependencies in ordering.missing_dependencies.items():
        for dependency in dependencies:
            issues.append(
                _issue(
                    "error",
                    "missing_dependency",
                    (
                        f"Step '{step_id}' depends on unknown "
                        f"step '{dependency}'."
                    ),
                    step_id,
                )
            )

    for cycle in ordering.cycles:
        issues.append(
            _issue(
                "error",
                "dependency_cycle",
                "Dependency cycle: " + " -> ".join(cycle),
            )
        )

    return issues


def _validate_validation_steps(
    plan: CodePlan,
) -> list[PlanValidationIssue]:
    """Check whether the plan contains meaningful validation coverage."""
    issues: list[PlanValidationIssue] = []

    validation_steps = [
        step
        for step in plan.steps
        if step.category.strip().lower()
        in {
            "validation",
            "testing",
            "test",
            "verification",
        }
    ]

    if plan.steps and not validation_steps and not plan.validation_steps:
        issues.append(
            _issue(
                "warning",
                "missing_validation",
                "Plan has no explicit validation or testing step.",
            )
        )

    for validation_step in plan.validation_steps:
        if not validation_step.strip():
            issues.append(
                _issue(
                    "error",
                    "empty_validation_step",
                    "Plan contains an empty validation step.",
                )
            )

    return issues


def validate_code_plan(plan: CodePlan) -> PlanValidationResult:
    """
    Validate whether a CodePlan is coherent and executable.

    This function does not modify the plan or source code.
    """
    issues: list[PlanValidationIssue] = []

    structural_issues = validate_plan_structure(plan)

    for message in structural_issues:
        issues.append(
            _issue(
                "error",
                "invalid_structure",
                message,
            )
        )

    issues.extend(_validate_steps(plan))
    issues.extend(_validate_impact_entries(plan))
    issues.extend(_validate_impact_consistency(plan))
    issues.extend(_validate_dependencies(plan))
    issues.extend(_validate_validation_steps(plan))

    return PlanValidationResult(
        valid=not any(
            issue.severity == "error"
            for issue in issues
        ),
        issues=issues,
    )


def build_validation_report(
    plan: CodePlan,
    result: PlanValidationResult,
) -> str:
    """Build a readable validation report."""
    lines = [
        "FLY-CODER PLAN VALIDATION",
        "=" * 26,
        "",
        f"Task: {plan.task}",
        "",
        f"Status: {'VALID' if result.valid else 'INVALID'}",
        "",
    ]

    if not result.issues:
        lines.append("No validation issues found.")
        return "\n".join(lines)

    lines.append("Issues:")

    for issue in result.issues:
        location = (
            f" [{issue.step_id}]"
            if issue.step_id
            else ""
        )

        lines.append(
            f"  - [{issue.severity.upper()}] "
            f"{issue.code}{location}: {issue.message}"
        )

    return "\n".join(lines)
