from flycoder.tools.code_plan import CodePlan, PlanStep
from flycoder.tools.ordering import (
    OrderingResult,
    build_ordering_report,
    find_dependency_cycles,
    find_missing_dependencies,
    order_plan_steps,
)


def make_step(
    step_id: str,
    description: str,
    dependencies: list[str] | None = None,
) -> PlanStep:
    return PlanStep(
        id=step_id,
        description=description,
        category="modify",
        dependencies=dependencies or [],
    )


def test_empty_plan_is_valid():
    plan = CodePlan(task="Fix login")

    result = order_plan_steps(plan)

    assert isinstance(result, OrderingResult)
    assert result.valid
    assert result.ordered_steps == []


def test_single_step_is_preserved():
    plan = CodePlan(task="Fix login")
    step = make_step("step-1", "Inspect login")
    plan.add_step(step)

    result = order_plan_steps(plan)

    assert result.valid
    assert result.ordered_steps == [step]


def test_independent_steps_preserve_original_order():
    plan = CodePlan(task="Fix login")

    first = make_step("step-1", "Inspect login")
    second = make_step("step-2", "Inspect users")
    third = make_step("step-3", "Inspect config")

    plan.add_step(first)
    plan.add_step(second)
    plan.add_step(third)

    result = order_plan_steps(plan)

    assert [step.id for step in result.ordered_steps] == [
        "step-1",
        "step-2",
        "step-3",
    ]


def test_dependency_is_ordered_before_dependent():
    plan = CodePlan(task="Fix login")

    first = make_step("step-1", "Inspect authentication")
    second = make_step(
        "step-2",
        "Modify authentication",
        ["step-1"],
    )

    plan.add_step(second)
    plan.add_step(first)

    result = order_plan_steps(plan)

    assert result.valid
    assert [step.id for step in result.ordered_steps] == [
        "step-1",
        "step-2",
    ]


def test_three_step_chain_is_ordered_correctly():
    plan = CodePlan(task="Fix login")

    plan.add_step(
        make_step(
            "step-3",
            "Test authentication",
            ["step-2"],
        )
    )
    plan.add_step(
        make_step(
            "step-2",
            "Modify authentication",
            ["step-1"],
        )
    )
    plan.add_step(
        make_step(
            "step-1",
            "Inspect authentication",
        )
    )

    result = order_plan_steps(plan)

    assert [step.id for step in result.ordered_steps] == [
        "step-1",
        "step-2",
        "step-3",
    ]


def test_branching_dependencies_are_respected():
    plan = CodePlan(task="Fix login")

    plan.add_step(
        make_step("step-4", "Integrate", ["step-2", "step-3"])
    )
    plan.add_step(
        make_step("step-2", "Modify auth", ["step-1"])
    )
    plan.add_step(
        make_step("step-3", "Modify tests", ["step-1"])
    )
    plan.add_step(
        make_step("step-1", "Inspect auth")
    )

    result = order_plan_steps(plan)
    order = [step.id for step in result.ordered_steps]

    assert order.index("step-1") < order.index("step-2")
    assert order.index("step-1") < order.index("step-3")
    assert order.index("step-2") < order.index("step-4")
    assert order.index("step-3") < order.index("step-4")


def test_missing_dependency_is_detected():
    plan = CodePlan(task="Fix login")

    plan.add_step(
        make_step(
            "step-2",
            "Modify authentication",
            ["step-1"],
        )
    )

    missing = find_missing_dependencies(plan)

    assert missing == {"step-2": ["step-1"]}


def test_missing_dependency_makes_order_invalid():
    plan = CodePlan(task="Fix login")

    plan.add_step(
        make_step(
            "step-2",
            "Modify authentication",
            ["missing"],
        )
    )

    result = order_plan_steps(plan)

    assert not result.valid
    assert result.ordered_steps == []
    assert result.missing_dependencies == {
        "step-2": ["missing"]
    }


def test_multiple_missing_dependencies_are_preserved():
    plan = CodePlan(task="Fix login")

    plan.add_step(
        make_step(
            "step-1",
            "Modify authentication",
            ["missing-a", "missing-b"],
        )
    )

    missing = find_missing_dependencies(plan)

    assert missing["step-1"] == [
        "missing-a",
        "missing-b",
    ]


def test_self_dependency_is_cycle():
    plan = CodePlan(task="Fix login")

    plan.add_step(
        make_step(
            "step-1",
            "Modify authentication",
            ["step-1"],
        )
    )

    cycles = find_dependency_cycles(plan)

    assert cycles == [["step-1", "step-1"]]


def test_two_step_cycle_is_detected():
    plan = CodePlan(task="Fix login")

    plan.add_step(
        make_step(
            "step-1",
            "Inspect authentication",
            ["step-2"],
        )
    )
    plan.add_step(
        make_step(
            "step-2",
            "Modify authentication",
            ["step-1"],
        )
    )

    cycles = find_dependency_cycles(plan)

    assert cycles == [
        ["step-1", "step-2", "step-1"]
    ]


def test_three_step_cycle_is_detected():
    plan = CodePlan(task="Fix login")

    plan.add_step(
        make_step("step-1", "One", ["step-3"])
    )
    plan.add_step(
        make_step("step-2", "Two", ["step-1"])
    )
    plan.add_step(
        make_step("step-3", "Three", ["step-2"])
    )

    cycles = find_dependency_cycles(plan)

    assert cycles == [
        ["step-1", "step-3", "step-2", "step-1"]
    ]


def test_cycle_makes_order_invalid():
    plan = CodePlan(task="Fix login")

    plan.add_step(
        make_step("step-1", "One", ["step-2"])
    )
    plan.add_step(
        make_step("step-2", "Two", ["step-1"])
    )

    result = order_plan_steps(plan)

    assert not result.valid
    assert result.ordered_steps == []
    assert result.cycles


def test_dependency_order_is_deterministic():
    plan = CodePlan(task="Fix login")

    plan.add_step(
        make_step("step-3", "Three", ["step-1"])
    )
    plan.add_step(
        make_step("step-2", "Two", ["step-1"])
    )
    plan.add_step(
        make_step("step-1", "One")
    )

    first = order_plan_steps(plan)
    second = order_plan_steps(plan)

    assert [step.id for step in first.ordered_steps] == [
        step.id for step in second.ordered_steps
    ]


def test_ready_steps_use_original_plan_order():
    plan = CodePlan(task="Fix login")

    plan.add_step(make_step("step-3", "Three"))
    plan.add_step(make_step("step-1", "One"))
    plan.add_step(make_step("step-2", "Two"))

    result = order_plan_steps(plan)

    assert [step.id for step in result.ordered_steps] == [
        "step-3",
        "step-1",
        "step-2",
    ]


def test_duplicate_dependency_does_not_break_ordering():
    plan = CodePlan(task="Fix login")

    plan.add_step(make_step("step-1", "Inspect"))
    plan.add_step(
        make_step(
            "step-2",
            "Modify",
            ["step-1", "step-1"],
        )
    )

    result = order_plan_steps(plan)

    assert result.valid
    assert [step.id for step in result.ordered_steps] == [
        "step-1",
        "step-2",
    ]


def test_find_missing_dependencies_returns_empty_for_valid_plan():
    plan = CodePlan(task="Fix login")
    plan.add_step(make_step("step-1", "Inspect"))
    plan.add_step(
        make_step(
            "step-2",
            "Modify",
            ["step-1"],
        )
    )

    assert find_missing_dependencies(plan) == {}


def test_find_dependency_cycles_returns_empty_for_valid_plan():
    plan = CodePlan(task="Fix login")
    plan.add_step(make_step("step-1", "Inspect"))
    plan.add_step(
        make_step(
            "step-2",
            "Modify",
            ["step-1"],
        )
    )

    assert find_dependency_cycles(plan) == []


def test_valid_result_has_empty_errors():
    plan = CodePlan(task="Fix login")
    plan.add_step(make_step("step-1", "Inspect"))

    result = order_plan_steps(plan)

    assert result.missing_dependencies == {}
    assert result.cycles == []


def test_report_contains_task():
    plan = CodePlan(task="Fix login")
    result = order_plan_steps(plan)

    report = build_ordering_report(plan, result)

    assert "Fix login" in report


def test_report_contains_execution_order():
    plan = CodePlan(task="Fix login")
    plan.add_step(make_step("step-1", "Inspect login"))

    result = order_plan_steps(plan)
    report = build_ordering_report(plan, result)

    assert "Execution order:" in report
    assert "[step-1]" in report


def test_report_contains_missing_dependencies():
    plan = CodePlan(task="Fix login")
    plan.add_step(
        make_step(
            "step-2",
            "Modify login",
            ["missing"],
        )
    )

    result = order_plan_steps(plan)
    report = build_ordering_report(plan, result)

    assert "Missing dependencies:" in report
    assert "step-2: missing" in report
    assert "Plan cannot be ordered." in report


def test_report_contains_cycles():
    plan = CodePlan(task="Fix login")
    plan.add_step(
        make_step("step-1", "One", ["step-2"])
    )
    plan.add_step(
        make_step("step-2", "Two", ["step-1"])
    )

    result = order_plan_steps(plan)
    report = build_ordering_report(plan, result)

    assert "Dependency cycles:" in report
    assert "step-1 -> step-2 -> step-1" in report


def test_report_handles_empty_plan():
    plan = CodePlan(task="Fix login")
    result = order_plan_steps(plan)

    report = build_ordering_report(plan, result)

    assert "No plan steps defined." in report
