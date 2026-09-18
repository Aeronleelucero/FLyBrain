from flycoder.tools.code_plan import (
    CodePlan,
    PlanStep,
    build_plan_report,
    create_plan,
    find_plan_step,
    find_plan_steps_by_category,
    validate_plan_structure,
)


def test_create_plan_normalizes_task():
    plan = create_plan("  Fix   the   login   bug  ")

    assert isinstance(plan, CodePlan)
    assert plan.task == "Fix the login bug"


def test_create_plan_normalizes_objective():
    plan = create_plan(
        "Fix login",
        "  Reject   expired   tokens  ",
    )

    assert plan.objective == "Reject expired tokens"


def test_new_plan_has_empty_collections():
    plan = create_plan("Fix login")

    assert plan.steps == []
    assert plan.impacted_files == []
    assert plan.impacted_symbols == []
    assert plan.validation_steps == []


def test_add_step():
    plan = create_plan("Fix login")
    step = PlanStep(
        id="step-1",
        description="Inspect authentication",
        category="inspect",
    )

    plan.add_step(step)

    assert plan.steps == [step]


def test_find_plan_step():
    plan = create_plan("Fix login")
    plan.add_step(
        PlanStep(
            id="step-1",
            description="Inspect authentication",
            category="inspect",
        )
    )

    result = find_plan_step(plan, "step-1")

    assert result is not None
    assert result.id == "step-1"


def test_find_plan_step_returns_none_for_missing_id():
    plan = create_plan("Fix login")

    assert find_plan_step(plan, "missing") is None


def test_find_plan_steps_by_category():
    plan = create_plan("Fix login")

    plan.add_step(
        PlanStep(
            id="step-1",
            description="Inspect authentication",
            category="inspect",
        )
    )
    plan.add_step(
        PlanStep(
            id="step-2",
            description="Modify authentication",
            category="modify",
        )
    )
    plan.add_step(
        PlanStep(
            id="step-3",
            description="Test authentication",
            category="test",
        )
    )

    result = find_plan_steps_by_category(plan, "modify")

    assert len(result) == 1
    assert result[0].id == "step-2"


def test_category_lookup_is_case_insensitive():
    plan = create_plan("Fix login")

    plan.add_step(
        PlanStep(
            id="step-1",
            description="Inspect authentication",
            category="Inspect",
        )
    )

    result = find_plan_steps_by_category(plan, "INSPECT")

    assert len(result) == 1


def test_add_impacted_file_deduplicates():
    plan = create_plan("Fix login")

    plan.add_impacted_file("auth.py")
    plan.add_impacted_file("auth.py")

    assert plan.impacted_files == ["auth.py"]


def test_add_impacted_symbol_deduplicates():
    plan = create_plan("Fix login")

    plan.add_impacted_symbol("login")
    plan.add_impacted_symbol("login")

    assert plan.impacted_symbols == ["login"]


def test_add_validation_step_deduplicates():
    plan = create_plan("Fix login")

    plan.add_validation_step("Run authentication tests")
    plan.add_validation_step("Run authentication tests")

    assert plan.validation_steps == [
        "Run authentication tests"
    ]


def test_step_stores_affected_files():
    step = PlanStep(
        id="step-1",
        description="Modify authentication",
        category="modify",
        affected_files=["auth.py"],
    )

    assert step.affected_files == ["auth.py"]


def test_step_stores_affected_symbols():
    step = PlanStep(
        id="step-1",
        description="Modify login",
        category="modify",
        affected_symbols=["login"],
    )

    assert step.affected_symbols == ["login"]


def test_step_stores_dependencies():
    step = PlanStep(
        id="step-2",
        description="Modify login",
        category="modify",
        dependencies=["step-1"],
    )

    assert step.dependencies == ["step-1"]


def test_valid_plan_has_no_structure_errors():
    plan = create_plan(
        "Fix login",
        "Reject expired tokens",
    )

    plan.add_step(
        PlanStep(
            id="step-1",
            description="Inspect authentication",
            category="inspect",
        )
    )
    plan.add_step(
        PlanStep(
            id="step-2",
            description="Modify token validation",
            category="modify",
            dependencies=["step-1"],
        )
    )

    assert validate_plan_structure(plan) == []


def test_empty_task_is_invalid():
    plan = create_plan("")

    errors = validate_plan_structure(plan)

    assert "Plan task must not be empty." in errors


def test_missing_step_id_is_invalid():
    plan = create_plan("Fix login")
    plan.add_step(
        PlanStep(
            id="",
            description="Inspect login",
            category="inspect",
        )
    )

    errors = validate_plan_structure(plan)

    assert "Step 1 must have an id." in errors


def test_duplicate_step_ids_are_invalid():
    plan = create_plan("Fix login")

    plan.add_step(
        PlanStep(
            id="step-1",
            description="Inspect login",
            category="inspect",
        )
    )
    plan.add_step(
        PlanStep(
            id="step-1",
            description="Modify login",
            category="modify",
        )
    )

    errors = validate_plan_structure(plan)

    assert "Duplicate step id: step-1" in errors


def test_missing_description_is_invalid():
    plan = create_plan("Fix login")
    plan.add_step(
        PlanStep(
            id="step-1",
            description="",
            category="inspect",
        )
    )

    errors = validate_plan_structure(plan)

    assert "Step 1 must have a description." in errors


def test_missing_category_is_invalid():
    plan = create_plan("Fix login")
    plan.add_step(
        PlanStep(
            id="step-1",
            description="Inspect login",
            category="",
        )
    )

    errors = validate_plan_structure(plan)

    assert "Step 1 must have a category." in errors


def test_self_dependency_is_invalid():
    plan = create_plan("Fix login")
    plan.add_step(
        PlanStep(
            id="step-1",
            description="Inspect login",
            category="inspect",
            dependencies=["step-1"],
        )
    )

    errors = validate_plan_structure(plan)

    assert (
        "Step step-1 cannot depend on itself."
        in errors
    )


def test_unknown_dependency_is_invalid():
    plan = create_plan("Fix login")
    plan.add_step(
        PlanStep(
            id="step-1",
            description="Modify login",
            category="modify",
            dependencies=["missing"],
        )
    )

    errors = validate_plan_structure(plan)

    assert (
        "Step step-1 references unknown dependency: missing"
        in errors
    )


def test_report_contains_task():
    plan = create_plan("Fix login")

    report = build_plan_report(plan)

    assert "Fix login" in report


def test_report_contains_objective():
    plan = create_plan(
        "Fix login",
        "Reject expired tokens",
    )

    report = build_plan_report(plan)

    assert "Reject expired tokens" in report


def test_report_contains_step_details():
    plan = create_plan("Fix login")

    plan.add_step(
        PlanStep(
            id="step-1",
            description="Inspect authentication",
            category="inspect",
            affected_files=["auth.py"],
            affected_symbols=["login"],
            rationale="Locate token validation.",
        )
    )

    report = build_plan_report(plan)

    assert "[step-1]" in report
    assert "[inspect]" in report
    assert "Inspect authentication" in report
    assert "auth.py" in report
    assert "login" in report
    assert "Locate token validation." in report


def test_report_contains_dependencies():
    plan = create_plan("Fix login")

    plan.add_step(
        PlanStep(
            id="step-2",
            description="Modify authentication",
            category="modify",
            dependencies=["step-1"],
        )
    )

    report = build_plan_report(plan)

    assert "Depends on: step-1" in report


def test_report_contains_impacted_files():
    plan = create_plan("Fix login")
    plan.add_impacted_file("auth.py")

    report = build_plan_report(plan)

    assert "auth.py" in report


def test_report_contains_impacted_symbols():
    plan = create_plan("Fix login")
    plan.add_impacted_symbol("login")

    report = build_plan_report(plan)

    assert "login" in report


def test_report_contains_validation_steps():
    plan = create_plan("Fix login")
    plan.add_validation_step("Run authentication tests")

    report = build_plan_report(plan)

    assert "Run authentication tests" in report


def test_report_handles_empty_sections():
    plan = create_plan("Fix login")

    report = build_plan_report(plan)

    assert "No plan steps defined." in report
    assert "None identified." in report
    assert "None defined." in report
