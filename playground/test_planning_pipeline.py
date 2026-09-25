"""Tests for the FLY-CODER Phase 9.1 planning pipeline."""

from datetime import datetime, timezone
from pathlib import Path

from flycoder.state import CodingState
from flycoder.tools.code_plan import CodePlan
from flycoder.tools.filesystem import Workspace
from flycoder.tools.memory import (
    Experience,
    MemoryStore,
)
from flycoder.tools.planning_pipeline import (
    IntegratedPlanningResult,
    build_planning_report,
    create_integrated_plan,
)
from flycoder.tools.strategy import StrategyDecision


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
    assert result.learning is not None
    assert result.strategy is not None
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

    assert first.strategy.strategy == second.strategy.strategy
    assert first.strategy.confidence == second.strategy.confidence


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


def test_pipeline_exposes_learning_context(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
    )

    assert result.learning is not None
    assert result.learning.task == result.task
    assert result.learning.memories == []


def test_pipeline_uses_supplied_memory_store(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    store = MemoryStore()

    experience = Experience(
        task="Add authentication validation",
        action="Update authentication middleware",
        outcome="Tests passed",
        success=True,
        verified=True,
        recorded_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    store.record_experience(experience)

    result = create_integrated_plan(
        workspace,
        state,
        memory_store=store,
    )

    assert len(result.learning.memories) == 1
    assert result.learning.memories[0].experience is experience


def test_pipeline_learning_does_not_change_plan_structure(
    tmp_path,
):
    workspace = make_workspace(tmp_path)
    state = make_state()

    without_memory = create_integrated_plan(
        workspace,
        state,
    )

    store = MemoryStore()

    store.record_experience(
        Experience(
            task="Add authentication validation",
            action="Update authentication middleware",
            outcome="Tests passed",
            success=True,
            verified=True,
        )
    )

    with_memory = create_integrated_plan(
        workspace,
        state,
        memory_store=store,
    )

    assert [
        item.description
        for item in without_memory.task_plan.items
    ] == [
        item.description
        for item in with_memory.task_plan.items
    ]

    assert [
        step.id
        for step in without_memory.plan.steps
    ] == [
        step.id
        for step in with_memory.plan.steps
    ]

    assert (
        without_memory.plan.validation_steps
        == with_memory.plan.validation_steps
    )


def test_pipeline_learning_does_not_modify_memory_store(
    tmp_path,
):
    workspace = make_workspace(tmp_path)
    state = make_state()

    store = MemoryStore()

    experience = Experience(
        task="Add authentication validation",
        action="Update middleware",
        outcome="Passed",
        success=True,
        verified=True,
    )

    store.record_experience(experience)
    before = list(store.experiences)

    create_integrated_plan(
        workspace,
        state,
        memory_store=store,
    )

    assert store.experiences == before
    assert store.experiences[0] is experience


def test_planning_report_contains_learning_summary(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    store = MemoryStore()

    store.record_experience(
        Experience(
            task="Add authentication validation",
            action="Update authentication middleware",
            outcome="Passed",
            success=True,
            verified=True,
        )
    )

    result = create_integrated_plan(
        workspace,
        state,
        memory_store=store,
    )

    report = build_planning_report(result)

    assert "Relevant memories: 1" in report
    assert "Verified memories: 1" in report
    assert "Previous failures: 0" in report


def test_pipeline_exposes_strategy_decision(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
    )

    assert isinstance(result.strategy, StrategyDecision)
    assert result.strategy.strategy == "create_new"
    assert 0.0 <= result.strategy.confidence <= 1.0


def test_pipeline_strategy_uses_learning_context(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    store = MemoryStore()

    store.record_experience(
        Experience(
            task="Add authentication validation",
            action="Implement authentication validation",
            outcome="Tests passed",
            success=True,
            verified=True,
        )
    )

    result = create_integrated_plan(
        workspace,
        state,
        memory_store=store,
    )

    assert result.strategy.strategy == "create_new"
    assert len(result.strategy.supporting_memories) == 1
    assert result.strategy.supporting_memories[0].experience is (
        store.experiences[0]
    )


def test_pipeline_strategy_does_not_change_plan_structure(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    without_memory = create_integrated_plan(
        workspace,
        state,
    )

    store = MemoryStore()

    store.record_experience(
        Experience(
            task="Add authentication validation",
            action="Implement authentication validation",
            outcome="Tests passed",
            success=True,
            verified=True,
        )
    )

    with_memory = create_integrated_plan(
        workspace,
        state,
        memory_store=store,
    )

    assert [
        step.id for step in with_memory.plan.steps
    ] == [
        step.id for step in without_memory.plan.steps
    ]

    assert [
        step.description for step in with_memory.plan.steps
    ] == [
        step.description for step in without_memory.plan.steps
    ]

    assert with_memory.plan.validation_steps == (
        without_memory.plan.validation_steps
    )


def test_pipeline_strategy_does_not_modify_memory_store(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    store = MemoryStore()

    experience = Experience(
        task="Add authentication validation",
        action="Implement authentication validation",
        outcome="Tests passed",
        success=True,
        verified=True,
    )

    store.record_experience(experience)

    before = list(store.experiences)

    create_integrated_plan(
        workspace,
        state,
        memory_store=store,
    )

    assert store.experiences == before
    assert store.experiences[0] is experience


def test_planning_report_contains_strategy_summary(tmp_path):
    workspace = make_workspace(tmp_path)
    state = make_state()

    result = create_integrated_plan(
        workspace,
        state,
    )

    report = build_planning_report(result)

    assert "Strategy:" in report
    assert "Selected: create_new" in report
    assert "Confidence:" in report