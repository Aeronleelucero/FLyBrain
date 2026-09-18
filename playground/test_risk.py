"""Tests for FLY-CODER risk analysis."""

from flycoder.tools.code_plan import CodePlan, PlanStep
from flycoder.tools.risk import (
    RiskAnalysis,
    RiskFactor,
    analyze_risk,
    build_risk_report,
)


def make_plan(
    *,
    task: str = "Add feature",
    objective: str = "Implement the requested change.",
    steps: list[PlanStep] | None = None,
    impacted_files: list[str] | None = None,
    impacted_symbols: list[str] | None = None,
    validation_steps: list[str] | None = None,
) -> CodePlan:
    return CodePlan(
        task=task,
        objective=objective,
        steps=steps or [],
        impacted_files=impacted_files or [],
        impacted_symbols=impacted_symbols or [],
        validation_steps=validation_steps or [],
    )


def step(
    step_id: str,
    description: str = "Implement change",
    category: str = "implementation",
    *,
    files: list[str] | None = None,
    symbols: list[str] | None = None,
    dependencies: list[str] | None = None,
) -> PlanStep:
    return PlanStep(
        id=step_id,
        description=description,
        category=category,
        affected_files=files or [],
        affected_symbols=symbols or [],
        dependencies=dependencies or [],
    )


def test_empty_plan_has_low_risk():
    analysis = analyze_risk(make_plan())

    assert isinstance(analysis, RiskAnalysis)
    assert analysis.level == "LOW"
    assert analysis.score == 0
    assert analysis.factors == []


def test_single_simple_step_has_low_risk():
    plan = make_plan(
        steps=[
            step("step-1"),
        ],
    )

    analysis = analyze_risk(plan)

    assert analysis.level == "LOW"
    assert analysis.score == 2
    assert any(
        factor.code == "no_validation"
        for factor in analysis.factors
    )


def test_validation_coverage_removes_no_validation_factor():
    plan = make_plan(
        steps=[
            step("step-1"),
            step(
                "step-2",
                description="Run tests",
                category="validation",
                dependencies=["step-1"],
            ),
        ],
    )

    analysis = analyze_risk(plan)

    assert not any(
        factor.code == "no_validation"
        for factor in analysis.factors
    )


def test_validation_steps_list_counts_as_coverage():
    plan = make_plan(
        steps=[step("step-1")],
        validation_steps=["Run the test suite."],
    )

    analysis = analyze_risk(plan)

    assert not any(
        factor.code == "no_validation"
        for factor in analysis.factors
    )


def test_multiple_files_add_risk():
    plan = make_plan(
        steps=[
            step(
                "step-1",
                files=[
                    "a.py",
                    "b.py",
                    "c.py",
                    "d.py",
                ],
            ),
        ],
        impacted_files=[
            "a.py",
            "b.py",
            "c.py",
            "d.py",
        ],
        validation_steps=["Run tests."],
    )

    analysis = analyze_risk(plan)

    factor = next(
        factor
        for factor in analysis.factors
        if factor.code == "multiple_files"
    )

    assert factor.score == 2


def test_many_files_add_higher_risk():
    files = [f"file_{index}.py" for index in range(8)]

    plan = make_plan(
        steps=[
            step("step-1", files=files),
        ],
        impacted_files=files,
        validation_steps=["Run tests."],
    )

    analysis = analyze_risk(plan)

    factor = next(
        factor
        for factor in analysis.factors
        if factor.code == "many_files"
    )

    assert factor.score == 3


def test_multiple_symbols_add_risk():
    symbols = [
        "A.one",
        "A.two",
        "A.three",
        "A.four",
        "A.five",
        "A.six",
    ]

    plan = make_plan(
        steps=[
            step("step-1", symbols=symbols),
        ],
        impacted_symbols=symbols,
        validation_steps=["Run tests."],
    )

    analysis = analyze_risk(plan)

    factor = next(
        factor
        for factor in analysis.factors
        if factor.code == "multiple_symbols"
    )

    assert factor.score == 2


def test_deep_dependency_chain_adds_risk():
    steps = [
        step("step-1"),
        step("step-2", dependencies=["step-1"]),
        step("step-3", dependencies=["step-2"]),
        step("step-4", dependencies=["step-3"]),
    ]

    plan = make_plan(
        steps=steps,
        validation_steps=["Run tests."],
    )

    analysis = analyze_risk(plan)

    factor = next(
        factor
        for factor in analysis.factors
        if factor.code == "deep_dependencies"
    )

    assert factor.score == 1


def test_very_deep_dependency_chain_adds_higher_risk():
    steps = [
        step("step-1"),
        step("step-2", dependencies=["step-1"]),
        step("step-3", dependencies=["step-2"]),
        step("step-4", dependencies=["step-3"]),
        step("step-5", dependencies=["step-4"]),
        step("step-6", dependencies=["step-5"]),
    ]

    plan = make_plan(
        steps=steps,
        validation_steps=["Run tests."],
    )

    analysis = analyze_risk(plan)

    factor = next(
        factor
        for factor in analysis.factors
        if factor.code == "deep_dependencies"
    )

    assert factor.score == 2


def test_authentication_is_security_sensitive():
    plan = make_plan(
        task="Add authentication",
        steps=[
            step(
                "step-1",
                description="Implement login authentication",
            ),
        ],
        validation_steps=["Run authentication tests."],
    )

    analysis = analyze_risk(plan)

    factor = next(
        factor
        for factor in analysis.factors
        if factor.code == "security_sensitive"
    )

    assert factor.score == 3


def test_authorization_is_security_sensitive():
    plan = make_plan(
        task="Update authorization permissions",
        steps=[
            step(
                "step-1",
                description="Update role permissions",
            ),
        ],
        validation_steps=["Run authorization tests."],
    )

    analysis = analyze_risk(plan)

    assert any(
        factor.code == "security_sensitive"
        for factor in analysis.factors
    )


def test_database_change_is_data_sensitive():
    plan = make_plan(
        task="Add database migration",
        steps=[
            step(
                "step-1",
                description="Create database schema migration",
            ),
        ],
        validation_steps=["Verify migration."],
    )

    analysis = analyze_risk(plan)

    factor = next(
        factor
        for factor in analysis.factors
        if factor.code == "data_sensitive"
    )

    assert factor.score == 2


def test_broad_refactor_adds_risk():
    plan = make_plan(
        task="Refactor the entire project",
        steps=[
            step(
                "step-1",
                description="Rewrite the architecture",
            ),
        ],
        validation_steps=["Run the full test suite."],
    )

    analysis = analyze_risk(plan)

    assert any(
        factor.code == "broad_change"
        for factor in analysis.factors
    )


def test_invalid_plan_adds_high_risk_factor():
    plan = make_plan(
        steps=[
            step(
                "step-1",
                description="",
            ),
        ],
        validation_steps=["Run tests."],
    )

    analysis = analyze_risk(plan)

    factor = next(
        factor
        for factor in analysis.factors
        if factor.code == "invalid_plan"
    )

    assert factor.score == 5


def test_missing_dependency_adds_unorderable_risk():
    plan = make_plan(
        steps=[
            step(
                "step-2",
                dependencies=["missing"],
            ),
        ],
        validation_steps=["Run tests."],
    )

    analysis = analyze_risk(plan)

    assert any(
        factor.code == "unorderable_plan"
        for factor in analysis.factors
    )


def test_cycle_adds_unorderable_risk():
    plan = make_plan(
        steps=[
            step("step-1", dependencies=["step-2"]),
            step("step-2", dependencies=["step-1"]),
        ],
        validation_steps=["Run tests."],
    )

    analysis = analyze_risk(plan)

    assert any(
        factor.code == "unorderable_plan"
        for factor in analysis.factors
    )


def test_high_risk_level_for_combined_sensitive_plan():
    files = [f"service_{index}.py" for index in range(8)]

    plan = make_plan(
        task="Refactor authentication and database migration",
        steps=[
            step(
                "analyze",
                description="Analyze authentication and database",
                files=files,
                symbols=[
                    f"AuthService.method_{index}"
                    for index in range(12)
                ],
            ),
        ],
        impacted_files=files,
        impacted_symbols=[
            f"AuthService.method_{index}"
            for index in range(12)
        ],
        validation_steps=["Run the full test suite."],
    )

    analysis = analyze_risk(plan)

    assert analysis.level == "HIGH"
    assert analysis.score >= 10


def test_recommendation_for_security_sensitive_plan():
    plan = make_plan(
        task="Modify authentication",
        steps=[
            step("step-1", description="Change login authentication"),
        ],
        validation_steps=["Run tests."],
    )

    analysis = analyze_risk(plan)

    assert any(
        "authentication" in recommendation.lower()
        for recommendation in analysis.recommendations
    )


def test_recommendation_for_database_plan():
    plan = make_plan(
        task="Change database schema",
        steps=[
            step("step-1", description="Update database migration"),
        ],
        validation_steps=["Verify migration."],
    )

    analysis = analyze_risk(plan)

    assert any(
        "database" in recommendation.lower()
        for recommendation in analysis.recommendations
    )


def test_recommendation_for_broad_change():
    plan = make_plan(
        task="Refactor the entire project",
        steps=[
            step("step-1", description="Rewrite everything"),
        ],
        validation_steps=["Run tests."],
    )

    analysis = analyze_risk(plan)

    assert any(
        "broad" in recommendation.lower()
        or "smaller" in recommendation.lower()
        for recommendation in analysis.recommendations
    )


def test_recommendation_for_missing_validation():
    plan = make_plan(
        steps=[
            step("step-1"),
        ],
    )

    analysis = analyze_risk(plan)

    assert any(
        "validation" in recommendation.lower()
        or "tests" in recommendation.lower()
        for recommendation in analysis.recommendations
    )


def test_risk_analysis_is_deterministic():
    plan = make_plan(
        task="Modify authentication",
        steps=[
            step(
                "step-1",
                description="Update login",
                files=["auth.py", "user.py"],
                symbols=["Auth.login", "User.save"],
            ),
            step(
                "step-2",
                description="Run tests",
                category="validation",
                dependencies=["step-1"],
            ),
        ],
        impacted_files=["auth.py", "user.py"],
        impacted_symbols=["Auth.login", "User.save"],
    )

    first = analyze_risk(plan)
    second = analyze_risk(plan)

    assert first == second


def test_report_contains_risk_level_and_score():
    plan = make_plan(
        steps=[
            step("step-1"),
        ],
    )

    analysis = analyze_risk(plan)
    report = build_risk_report(plan, analysis)

    assert "FLY-CODER RISK ANALYSIS" in report
    assert f"Risk level: {analysis.level}" in report
    assert f"Risk score: {analysis.score}" in report


def test_report_contains_factors():
    plan = make_plan(
        task="Add authentication",
        steps=[
            step(
                "step-1",
                description="Implement authentication",
            ),
        ],
        validation_steps=["Run tests."],
    )

    analysis = analyze_risk(plan)
    report = build_risk_report(plan, analysis)

    assert "[security_sensitive]" in report
    assert "authentication" in report.lower()


def test_report_contains_recommendations():
    plan = make_plan(
        task="Add authentication",
        steps=[
            step(
                "step-1",
                description="Implement authentication",
            ),
        ],
    )

    analysis = analyze_risk(plan)
    report = build_risk_report(plan, analysis)

    assert "Recommendations:" in report
    assert "authentication" in report.lower()


def test_risk_factor_dataclass():
    factor = RiskFactor(
        code="test",
        description="Test factor",
        score=2,
    )

    assert factor.code == "test"
    assert factor.description == "Test factor"
    assert factor.score == 2


def test_windows_paths_are_normalized_for_file_count():
    plan = make_plan(
        steps=[
            step(
                "step-1",
                files=[
                    r"src\auth.py",
                    "src/auth.py",
                ],
            ),
        ],
        validation_steps=["Run tests."],
    )

    analysis = analyze_risk(plan)

    assert not any(
        factor.code == "multiple_files"
        for factor in analysis.factors
    )


def test_recommendation_for_invalid_plan():
    plan = make_plan(
        steps=[
            step("step-1", description=""),
        ],
        validation_steps=["Run tests."],
    )

    analysis = analyze_risk(plan)

    assert any(
        "validation errors" in recommendation.lower()
        for recommendation in analysis.recommendations
    )
