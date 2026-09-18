"""Tests for FLY-CODER plan validation."""

from flycoder.tools.code_plan import CodePlan, PlanStep
from flycoder.tools.plan_validation import (
    PlanValidationResult,
    build_validation_report,
    validate_code_plan,
)


def make_plan(
    *,
    task: str = "Add authentication",
    steps: list[PlanStep] | None = None,
    impacted_files: list[str] | None = None,
    impacted_symbols: list[str] | None = None,
    validation_steps: list[str] | None = None,
) -> CodePlan:
    return CodePlan(
        task=task,
        objective="Implement the requested change.",
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


def test_empty_plan_is_valid():
    plan = make_plan()

    result = validate_code_plan(plan)

    assert result.valid
    assert result.errors == []


def test_empty_task_is_invalid():
    plan = make_plan(task="")

    result = validate_code_plan(plan)

    assert not result.valid
    assert any(
        issue.code == "invalid_structure"
        for issue in result.errors
    )


def test_empty_step_description_is_invalid():
    plan = make_plan(
        steps=[
            step("step-1", description=""),
        ],
    )

    result = validate_code_plan(plan)

    assert not result.valid
    assert any(
        issue.code == "empty_description"
        for issue in result.errors
    )


def test_empty_step_category_is_invalid():
    plan = make_plan(
        steps=[
            step("step-1", category=""),
        ],
    )

    result = validate_code_plan(plan)

    assert not result.valid
    assert any(
        issue.code == "empty_category"
        for issue in result.errors
    )


def test_empty_affected_file_is_invalid():
    plan = make_plan(
        steps=[
            step("step-1", files=[""]),
        ],
    )

    result = validate_code_plan(plan)

    assert not result.valid
    assert any(
        issue.code == "invalid_affected_file"
        for issue in result.errors
    )


def test_empty_affected_symbol_is_invalid():
    plan = make_plan(
        steps=[
            step("step-1", symbols=[""]),
        ],
    )

    result = validate_code_plan(plan)

    assert not result.valid
    assert any(
        issue.code == "invalid_affected_symbol"
        for issue in result.errors
    )


def test_empty_impacted_file_is_invalid():
    plan = make_plan(
        impacted_files=[""],
    )

    result = validate_code_plan(plan)

    assert not result.valid
    assert any(
        issue.code == "invalid_impacted_file"
        for issue in result.errors
    )


def test_empty_impacted_symbol_is_invalid():
    plan = make_plan(
        impacted_symbols=[""],
    )

    result = validate_code_plan(plan)

    assert not result.valid
    assert any(
        issue.code == "invalid_impacted_symbol"
        for issue in result.errors
    )


def test_step_file_must_exist_in_plan_impact():
    plan = make_plan(
        steps=[
            step(
                "step-1",
                files=["flycoder/main.py"],
            ),
        ],
        impacted_files=[],
    )

    result = validate_code_plan(plan)

    assert not result.valid
    assert any(
        issue.code == "missing_impacted_file"
        for issue in result.errors
    )


def test_step_symbol_must_exist_in_plan_impact():
    plan = make_plan(
        steps=[
            step(
                "step-1",
                symbols=["AuthController.login"],
            ),
        ],
        impacted_symbols=[],
    )

    result = validate_code_plan(plan)

    assert not result.valid
    assert any(
        issue.code == "missing_impacted_symbol"
        for issue in result.errors
    )


def test_matching_file_impact_is_valid():
    plan = make_plan(
        steps=[
            step(
                "step-1",
                files=["flycoder/main.py"],
            ),
        ],
        impacted_files=["flycoder/main.py"],
    )

    result = validate_code_plan(plan)

    assert result.valid


def test_windows_file_path_matches_posix_plan_path():
    plan = make_plan(
        steps=[
            step(
                "step-1",
                files=[r"flycoder\main.py"],
            ),
        ],
        impacted_files=["flycoder/main.py"],
    )

    result = validate_code_plan(plan)

    assert result.valid


def test_matching_symbol_impact_is_valid():
    plan = make_plan(
        steps=[
            step(
                "step-1",
                symbols=["AuthController.login"],
            ),
        ],
        impacted_symbols=["AuthController.login"],
    )

    result = validate_code_plan(plan)

    assert result.valid


def test_missing_dependency_is_invalid():
    plan = make_plan(
        steps=[
            step(
                "step-2",
                dependencies=["step-1"],
            ),
        ],
    )

    result = validate_code_plan(plan)

    assert not result.valid
    assert any(
        issue.code == "missing_dependency"
        for issue in result.errors
    )


def test_dependency_cycle_is_invalid():
    plan = make_plan(
        steps=[
            step("step-1", dependencies=["step-2"]),
            step("step-2", dependencies=["step-1"]),
        ],
    )

    result = validate_code_plan(plan)

    assert not result.valid
    assert any(
        issue.code == "dependency_cycle"
        for issue in result.errors
    )


def test_self_dependency_is_invalid():
    plan = make_plan(
        steps=[
            step("step-1", dependencies=["step-1"]),
        ],
    )

    result = validate_code_plan(plan)

    assert not result.valid


def test_validation_category_counts_as_validation_coverage():
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

    result = validate_code_plan(plan)

    assert result.valid
    assert result.warnings == []


def test_testing_category_counts_as_validation_coverage():
    plan = make_plan(
        steps=[
            step(
                "step-1",
                description="Run tests",
                category="testing",
            ),
        ],
    )

    result = validate_code_plan(plan)

    assert result.valid
    assert result.warnings == []


def test_validation_steps_list_counts_as_validation_coverage():
    plan = make_plan(
        steps=[
            step("step-1"),
        ],
        validation_steps=["Run the test suite."],
    )

    result = validate_code_plan(plan)

    assert result.valid
    assert result.warnings == []


def test_missing_validation_produces_warning():
    plan = make_plan(
        steps=[
            step("step-1"),
        ],
    )

    result = validate_code_plan(plan)

    assert result.valid
    assert any(
        issue.code == "missing_validation"
        for issue in result.warnings
    )


def test_empty_validation_step_is_invalid():
    plan = make_plan(
        validation_steps=[""],
    )

    result = validate_code_plan(plan)

    assert not result.valid
    assert any(
        issue.code == "empty_validation_step"
        for issue in result.errors
    )


def test_validation_result_separates_errors_and_warnings():
    plan = make_plan(
        steps=[
            step("step-1"),
        ],
    )

    result = validate_code_plan(plan)

    assert isinstance(result, PlanValidationResult)
    assert result.valid
    assert result.errors == []
    assert len(result.warnings) == 1


def test_multiple_errors_are_preserved():
    plan = make_plan(
        steps=[
            step(
                "step-1",
                description="",
                category="",
                files=[""],
                symbols=[""],
            ),
        ],
        impacted_files=[""],
        impacted_symbols=[""],
        validation_steps=[""],
    )

    result = validate_code_plan(plan)

    assert not result.valid
    assert len(result.errors) >= 5


def test_report_for_valid_plan():
    plan = make_plan(
        steps=[
            step(
                "step-1",
                description="Implement authentication",
                category="implementation",
                files=["auth.py"],
            ),
            step(
                "step-2",
                description="Run tests",
                category="validation",
                dependencies=["step-1"],
            ),
        ],
        impacted_files=["auth.py"],
    )

    result = validate_code_plan(plan)
    report = build_validation_report(plan, result)

    assert "FLY-CODER PLAN VALIDATION" in report
    assert "Status: VALID" in report
    assert "No validation issues found." in report


def test_report_for_invalid_plan():
    plan = make_plan(
        steps=[
            step(
                "step-2",
                dependencies=["missing"],
            ),
        ],
    )

    result = validate_code_plan(plan)
    report = build_validation_report(plan, result)

    assert "Status: INVALID" in report
    assert "missing_dependency" in report
    assert "step-2" in report


def test_report_includes_warning():
    plan = make_plan(
        steps=[
            step("step-1"),
        ],
    )

    result = validate_code_plan(plan)
    report = build_validation_report(plan, result)

    assert "Status: VALID" in report
    assert "[WARNING]" in report
    assert "missing_validation" in report


def test_valid_complete_plan_has_no_issues():
    plan = make_plan(
        steps=[
            step(
                "analyze",
                description="Analyze authentication",
                category="analysis",
                files=["auth.py"],
                symbols=["AuthController.login"],
            ),
            step(
                "implement",
                description="Implement authentication",
                category="implementation",
                files=["auth.py"],
                symbols=["AuthController.login"],
                dependencies=["analyze"],
            ),
            step(
                "validate",
                description="Run tests",
                category="validation",
                files=["tests/test_auth.py"],
                dependencies=["implement"],
            ),
        ],
        impacted_files=[
            "auth.py",
            "tests/test_auth.py",
        ],
        impacted_symbols=[
            "AuthController.login",
        ],
        validation_steps=[
            "Run the authentication test suite.",
        ],
    )

    result = validate_code_plan(plan)

    assert result.valid
    assert result.errors == []
    assert result.warnings == []
