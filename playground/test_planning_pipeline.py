"""Tests for the FLY-CODER Phase 5.7 planning pipeline."""

from pathlib import Path

from flycoder.state import CodingState
from flycoder.tools.code_plan import CodePlan
from flycoder.tools.filesystem import Workspace
from flycoder.tools.planning_pipeline import (
    IntegratedPlanningResult,
    build_planning_report,
    create_integrated_plan,
)


def make_workspace(tmp_path: Path) -> Workspace:
    """Build a small workspace for pipeline tests."""

    files = {
        "app.py": (
            "def authenticate(user):\n"
            "    return True\n"
        ),
        "auth.py": (
            "def validate_credentials(user):\n"
            "    return True\n"
        ),
        "database.py": (
            "def connect_database():\n"
            "    return None\n"
        ),
    }

    for relative_path, content in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    return Workspace(tmp_path)


def make_state(
    task: str = "Add authentication validation",
) -> CodingState:
    """Build a minimal state for pipeline tests."""

    return CodingState(
        task=task,
        files=[
            "app.py",
            "auth.py",
            "database.py",
        ],
        relevant_files=[
            "app.py",
            "auth.py",
            "database.py",
        ],
    )


def test_create_integrated_plan_returns_complete_result(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
    )

    assert isinstance(result, IntegratedPlanningResult)
    assert result.task == "Add authentication validation"
    assert result.task_plan is not None
    assert result.impact is not None
    assert result.plan is not None
    assert result.ordering is not None
    assert result.validation is not None
    assert result.risk is not None


def test_pipeline_creates_code_plan(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
    )

    assert isinstance(result.plan, CodePlan)
    assert result.plan.task == result.task


def test_pipeline_creates_plan_steps_from_task_decomposition(
    tmp_path,
):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
    )

    assert len(result.task_plan.items) > 0
    assert len(result.plan.steps) == len(
        result.task_plan.items
    )


def test_plan_step_ids_are_unique(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
    )

    step_ids = [step.id for step in result.plan.steps]

    assert len(step_ids) == len(set(step_ids))


def test_pipeline_orders_plan_steps(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
    )

    assert result.ordering.valid
    assert len(result.ordering.ordered_steps) == len(
        result.plan.steps
    )


def test_pipeline_validates_plan(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
    )

    assert result.validation.valid
    assert not result.validation.errors


def test_pipeline_adds_validation_steps(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
    )

    assert result.plan.validation_steps
    assert any(
        "test" in step.lower()
        for step in result.plan.validation_steps
    )


def test_pipeline_performs_risk_analysis(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
    )

    assert result.risk is not None
    assert result.risk.level in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }
    assert result.risk.score >= 0


def test_integrated_result_valid_property(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
    )

    assert result.valid is True


def test_integrated_result_safe_property(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
    )

    assert result.safe is True


def test_task_argument_overrides_state_task(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state("Old task")

    result = create_integrated_plan(
        workspace,
        state,
        task="Add database migration",
    )

    assert result.task == "Add database migration"
    assert result.plan.task == "Add database migration"


def test_task_is_normalized(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
        task="   Add    authentication    validation   ",
    )

    assert result.task == "Add authentication validation"


def test_pipeline_does_not_modify_state_analysis(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    original_files = list(state.files)
    original_relevant_files = list(state.relevant_files)

    create_integrated_plan(
        workspace,
        state,
    )

    assert state.files == original_files
    assert state.relevant_files == original_relevant_files


def test_pipeline_does_not_modify_source_content(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    state.current_file = "auth.py"
    state.current_file_content = (
        "def authenticate(user):\n"
        "    return True\n"
    )

    original_content = state.current_file_content

    create_integrated_plan(
        workspace,
        state,
    )

    assert state.current_file_content == original_content


def test_pipeline_discovers_workspace_files(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
    )

    assert result.impact is not None

    workspace_files = set(workspace.list_files())

    for item in result.impact.files:
        assert item in workspace_files


def test_build_planning_report_contains_summary(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
    )

    report = build_planning_report(result)

    assert "FLY-CODER INTEGRATED PLANNING" in report
    assert "Task:" in report
    assert "Task items:" in report
    assert "Impact items:" in report
    assert "Plan steps:" in report
    assert "Validation:" in report
    assert "Risk:" in report


def test_build_planning_report_contains_risk_information(
    tmp_path,
):
    workspace = make_workspace(tmp_path)
    state = make_state(
        "Add authentication to the database",
    )

    result = create_integrated_plan(
        workspace,
        state,
    )

    report = build_planning_report(result)

    assert "Risk:" in report
    assert (
        "Risk factors:" in report
        or result.risk.score == 0
    )


def test_pipeline_is_deterministic(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state(
        "Add authentication validation",
    )

    first = create_integrated_plan(
        workspace,
        state,
    )
    second = create_integrated_plan(
        workspace,
        state,
    )

    assert first.task == second.task

    assert [
        item.description
        for item in first.task_plan.items
    ] == [
        item.description
        for item in second.task_plan.items
    ]

    assert [
        item.path
        for item in first.impact.items
    ] == [
        item.path
        for item in second.impact.items
    ]

    assert [
        step.id
        for step in first.plan.steps
    ] == [
        step.id
        for step in second.plan.steps
    ]

    assert [
        step.id
        for step in first.ordering.ordered_steps
    ] == [
        step.id
        for step in second.ordering.ordered_steps
    ]

    assert first.risk.level == second.risk.level
    assert first.risk.score == second.risk.score


def test_pipeline_is_planning_only(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    before = {
        "files": list(state.files),
        "relevant_files": list(state.relevant_files),
        "current_file": state.current_file,
        "current_file_content": state.current_file_content,
        "repair_applied": state.repair_applied,
        "proposed_content": state.proposed_content,
    }

    create_integrated_plan(
        workspace,
        state,
    )

    after = {
        "files": list(state.files),
        "relevant_files": list(state.relevant_files),
        "current_file": state.current_file,
        "current_file_content": state.current_file_content,
        "repair_applied": state.repair_applied,
        "proposed_content": state.proposed_content,
    }

    assert after == before
