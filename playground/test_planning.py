from flycoder.tools.planning import (
    TaskItem,
    TaskPlan,
    build_plan_report,
    decompose_task,
    find_plan_items_by_category,
)


def test_empty_task_returns_empty_plan():
    plan = decompose_task("")

    assert isinstance(plan, TaskPlan)
    assert plan.task == ""
    assert plan.items == []


def test_task_is_normalized():
    plan = decompose_task(
        "  Add   authentication   to the API  "
    )

    assert plan.task == "Add authentication to the API"


def test_every_non_empty_task_starts_with_understand():
    plan = decompose_task("Add authentication")

    assert plan.items[0].category == "understand"


def test_add_task_contains_implement():
    plan = decompose_task("Add authentication")

    categories = [
        item.category
        for item in plan.items
    ]

    assert "implement" in categories


def test_fix_task_contains_repair():
    plan = decompose_task(
        "Fix the login validation bug"
    )

    categories = [
        item.category
        for item in plan.items
    ]

    assert "repair" in categories


def test_explicit_test_task_contains_test():
    plan = decompose_task(
        "Add tests for authentication"
    )

    categories = [
        item.category
        for item in plan.items
    ]

    assert "test" in categories


def test_non_test_task_contains_verify():
    plan = decompose_task(
        "Add authentication"
    )

    categories = [
        item.category
        for item in plan.items
    ]

    assert "verify" in categories
    assert "test" not in categories


def test_modify_task_contains_modify():
    plan = decompose_task(
        "Update the configuration"
    )

    categories = [
        item.category
        for item in plan.items
    ]

    assert "modify" in categories


def test_refactor_task_contains_refactor():
    plan = decompose_task(
        "Refactor the authentication service"
    )

    categories = [
        item.category
        for item in plan.items
    ]

    assert "refactor" in categories


def test_find_items_by_category():
    plan = decompose_task(
        "Fix the bug and add tests"
    )

    repair_items = find_plan_items_by_category(
        plan,
        "repair",
    )

    assert len(repair_items) == 1
    assert repair_items[0].category == "repair"


def test_category_lookup_is_case_insensitive():
    plan = decompose_task("Add authentication")

    items = find_plan_items_by_category(
        plan,
        "IMPLEMENT",
    )

    assert len(items) == 1


def test_plan_items_are_task_items():
    plan = decompose_task("Add authentication")

    assert all(
        isinstance(item, TaskItem)
        for item in plan.items
    )


def test_report_contains_task():
    plan = decompose_task("Add authentication")

    report = build_plan_report(plan)

    assert "Add authentication" in report


def test_report_contains_categories():
    plan = decompose_task("Add authentication")

    report = build_plan_report(plan)

    assert "[understand]" in report
    assert "[implement]" in report
    assert "[verify]" in report


def test_report_numbers_items():
    plan = decompose_task("Add authentication")

    report = build_plan_report(plan)

    assert "1." in report
    assert "2." in report


def test_report_handles_empty_plan():
    plan = TaskPlan(
        task="",
        items=[],
    )

    report = build_plan_report(plan)

    assert "No work items identified." in report


def test_multiple_intents_are_preserved():
    plan = decompose_task(
        "Fix the login bug and add tests"
    )

    categories = [
        item.category
        for item in plan.items
    ]

    assert categories == [
        "understand",
        "repair",
        "implement",
        "test",
    ]
