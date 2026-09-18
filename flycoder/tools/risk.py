"""Risk and safety analysis tools for FLY-CODER plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

from flycoder.tools.code_plan import CodePlan, PlanStep
from flycoder.tools.ordering import order_plan_steps
from flycoder.tools.plan_validation import validate_code_plan


@dataclass
class RiskFactor:
    """A factor contributing to plan risk."""

    code: str
    description: str
    score: int


@dataclass
class RiskAnalysis:
    """Deterministic risk analysis for a CodePlan."""

    level: str
    score: int
    factors: list[RiskFactor] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def _add_factor(
    factors: list[RiskFactor],
    code: str,
    description: str,
    score: int,
) -> None:
    """Add a risk factor."""
    factors.append(
        RiskFactor(
            code=code,
            description=description,
            score=score,
        )
    )


def _unique_files(plan: CodePlan) -> set[str]:
    """Return unique affected files."""
    files = set(plan.impacted_files)

    for step in plan.steps:
        files.update(step.affected_files)

    return {
        file_path.replace("\\", "/")
        for file_path in files
        if file_path.strip()
    }


def _unique_symbols(plan: CodePlan) -> set[str]:
    """Return unique affected symbols."""
    symbols = set(plan.impacted_symbols)

    for step in plan.steps:
        symbols.update(step.affected_symbols)

    return {
        symbol.strip()
        for symbol in symbols
        if symbol.strip()
    }


def _dependency_depth(plan: CodePlan) -> int:
    """Calculate the deepest dependency chain."""
    steps = {step.id: step for step in plan.steps}
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(step_id: str) -> int:
        if step_id in memo:
            return memo[step_id]

        if step_id in visiting:
            return 0

        step = steps.get(step_id)

        if step is None:
            return 0

        visiting.add(step_id)

        value = 1

        for dependency in step.dependencies:
            value = max(value, 1 + depth(dependency))

        visiting.remove(step_id)
        memo[step_id] = value

        return value

    return max(
        (depth(step.id) for step in plan.steps),
        default=0,
    )


def _categories(plan: CodePlan) -> set[str]:
    """Return normalized plan categories."""
    return {
        step.category.strip().lower()
        for step in plan.steps
        if step.category.strip()
    }


def _has_validation(plan: CodePlan) -> bool:
    """Return whether explicit validation exists."""
    validation_categories = {
        "validation",
        "testing",
        "test",
        "verification",
    }

    if plan.validation_steps:
        return True

    return any(
        step.category.strip().lower() in validation_categories
        for step in plan.steps
    )


def _plan_text(plan: CodePlan) -> str:
    """Build a flat searchable representation of a plan."""
    parts: list[str] = [
        plan.task,
        plan.objective,
    ]

    for step in plan.steps:
        parts.extend(
            [
                step.description,
                step.category,
                *step.affected_symbols,
                *step.affected_files,
            ]
        )

    parts.extend(plan.impacted_files)
    parts.extend(plan.impacted_symbols)
    parts.extend(plan.validation_steps)

    return " ".join(
        part
        for part in parts
        if isinstance(part, str)
    ).lower()


def _authentication_related(plan: CodePlan) -> bool:
    """Detect authentication and authorization work."""
    terms = {
        "auth",
        "authentication",
        "authorization",
        "authorize",
        "login",
        "logout",
        "permission",
        "permissions",
        "role",
        "roles",
        "security",
    }

    searchable = _plan_text(plan)

    return any(term in searchable for term in terms)


def _database_related(plan: CodePlan) -> bool:
    """Detect database and persistence work."""
    terms = {
        "database",
        "db",
        "migration",
        "schema",
        "sql",
        "sqlite",
        "mysql",
        "postgres",
        "postgresql",
        "query",
        "queries",
        "repository",
    }

    searchable = _plan_text(plan)

    return any(term in searchable for term in terms)


def _broad_change(plan: CodePlan) -> bool:
    """Detect plans that explicitly describe broad changes."""
    terms = {
        "refactor",
        "rewrite",
        "redesign",
        "migration",
        "global",
        "all files",
        "entire project",
        "system-wide",
        "system wide",
    }

    searchable = _plan_text(plan)

    return any(term in searchable for term in terms)


def _calculate_factors(plan: CodePlan) -> list[RiskFactor]:
    """Calculate deterministic risk factors."""
    factors: list[RiskFactor] = []

    file_count = len(_unique_files(plan))
    symbol_count = len(_unique_symbols(plan))
    step_count = len(plan.steps)
    dependency_depth = _dependency_depth(plan)

    if file_count >= 8:
        _add_factor(
            factors,
            "many_files",
            f"Plan affects {file_count} files.",
            3,
        )
    elif file_count >= 4:
        _add_factor(
            factors,
            "multiple_files",
            f"Plan affects {file_count} files.",
            2,
        )
    elif file_count >= 2:
        _add_factor(
            factors,
            "multiple_files",
            f"Plan affects {file_count} files.",
            1,
        )

    if symbol_count >= 12:
        _add_factor(
            factors,
            "many_symbols",
            f"Plan affects {symbol_count} symbols.",
            3,
        )
    elif symbol_count >= 6:
        _add_factor(
            factors,
            "multiple_symbols",
            f"Plan affects {symbol_count} symbols.",
            2,
        )
    elif symbol_count >= 3:
        _add_factor(
            factors,
            "multiple_symbols",
            f"Plan affects {symbol_count} symbols.",
            1,
        )

    if step_count >= 10:
        _add_factor(
            factors,
            "many_steps",
            f"Plan contains {step_count} steps.",
            2,
        )
    elif step_count >= 5:
        _add_factor(
            factors,
            "multiple_steps",
            f"Plan contains {step_count} steps.",
            1,
        )

    if dependency_depth >= 5:
        _add_factor(
            factors,
            "deep_dependencies",
            f"Dependency chain reaches depth {dependency_depth}.",
            2,
        )
    elif dependency_depth >= 3:
        _add_factor(
            factors,
            "deep_dependencies",
            f"Dependency chain reaches depth {dependency_depth}.",
            1,
        )

    if _authentication_related(plan):
        _add_factor(
            factors,
            "security_sensitive",
            (
                "Plan touches authentication, authorization, "
                "permissions, or security."
            ),
            3,
        )

    if _database_related(plan):
        _add_factor(
            factors,
            "data_sensitive",
            (
                "Plan touches database, schema, migration, "
                "or persistence behavior."
            ),
            2,
        )

    if _broad_change(plan):
        _add_factor(
            factors,
            "broad_change",
            (
                "Plan describes a broad refactor, rewrite, "
                "migration, or system-wide change."
            ),
            2,
        )

    validation = validate_code_plan(plan)

    if validation.errors:
        _add_factor(
            factors,
            "invalid_plan",
            f"Plan contains {len(validation.errors)} validation error(s).",
            5,
        )

    if not _has_validation(plan) and plan.steps:
        _add_factor(
            factors,
            "no_validation",
            "Plan has no explicit testing or validation coverage.",
            2,
        )

    ordering = order_plan_steps(plan)

    if not ordering.valid:
        _add_factor(
            factors,
            "unorderable_plan",
            (
                "Plan cannot be completely ordered because "
                "of dependency problems."
            ),
            5,
        )

    return factors


def _recommendations(
    plan: CodePlan,
    factors: list[RiskFactor],
) -> list[str]:
    """Build deterministic safety recommendations."""
    recommendations: list[str] = []

    codes = {factor.code for factor in factors}

    if "invalid_plan" in codes:
        recommendations.append(
            "Resolve all plan validation errors before implementation."
        )

    if "unorderable_plan" in codes:
        recommendations.append(
            (
                "Resolve missing dependencies or dependency cycles "
                "before implementation."
            )
        )

    if "security_sensitive" in codes:
        recommendations.append(
            "Run focused authentication and authorization tests."
        )

    if "data_sensitive" in codes:
        recommendations.append(
            "Verify database, migration, and data-integrity behavior."
        )

    if "many_files" in codes or "multiple_files" in codes:
        recommendations.append(
            "Review every affected file before applying changes."
        )

    if "many_symbols" in codes or "multiple_symbols" in codes:
        recommendations.append(
            "Review callers and relationships of affected symbols."
        )

    if "deep_dependencies" in codes:
        recommendations.append(
            (
                "Execute changes in dependency order and verify "
                "each dependency boundary."
            )
        )

    if "broad_change" in codes:
        recommendations.append(
            (
                "Split broad changes into smaller independently "
                "verifiable steps where possible."
            )
        )

    if "no_validation" in codes:
        recommendations.append(
            "Add explicit tests or validation steps before implementation."
        )

    if not recommendations and plan.steps:
        recommendations.append(
            (
                "Review the plan and run its declared validation "
                "steps after implementation."
            )
        )

    return recommendations


def analyze_risk(plan: CodePlan) -> RiskAnalysis:
    """
    Analyze the implementation risk of a CodePlan.

    The analysis is deterministic and does not modify the plan or source code.
    """
    factors = _calculate_factors(plan)
    score = sum(factor.score for factor in factors)

    if score >= 10:
        level = "HIGH"
    elif score >= 5:
        level = "MEDIUM"
    else:
        level = "LOW"

    return RiskAnalysis(
        level=level,
        score=score,
        factors=factors,
        recommendations=_recommendations(plan, factors),
    )


def build_risk_report(
    plan: CodePlan,
    analysis: RiskAnalysis,
) -> str:
    """Build a readable risk-analysis report."""
    lines = [
        "FLY-CODER RISK ANALYSIS",
        "=" * 23,
        "",
        f"Task: {plan.task}",
        "",
        f"Risk level: {analysis.level}",
        f"Risk score: {analysis.score}",
        "",
    ]

    if analysis.factors:
        lines.append("Risk factors:")

        for factor in analysis.factors:
            lines.append(
                f"  - [{factor.code}] "
                f"(+{factor.score}) {factor.description}"
            )

        lines.append("")

    if analysis.recommendations:
        lines.append("Recommendations:")

        for recommendation in analysis.recommendations:
            lines.append(f"  - {recommendation}")

    return "\n".join(lines)
