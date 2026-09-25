from datetime import datetime, timezone

from flycoder.tools.action_selection import ActionDecision
from flycoder.tools.code_plan import CodePlan, PlanStep
from flycoder.tools.impact import ImpactAnalysis
from flycoder.tools.learning import LearningContext, LearningMemory
from flycoder.tools.learning_validation import validate_learning_context
from flycoder.tools.memory import Experience
from flycoder.tools.ordering import OrderingResult
from flycoder.tools.plan_validation import PlanValidationResult
from flycoder.tools.planning import TaskPlan, TaskItem
from flycoder.tools.planning_pipeline import (
    IntegratedPlanningResult,
    build_planning_report,
)
from flycoder.tools.risk import RiskAnalysis
from flycoder.tools.strategy import StrategyDecision


BASE_TIME = datetime(
    2026,
    1,
    1,
    tzinfo=timezone.utc,
)


def test_planning_report_includes_validation_guidance_with_learning_guidance():
    experience = Experience(
        task="Fix authentication",
        action="repair authentication",
        outcome="Passed",
        success=True,
        verified=True,
        recorded_at=BASE_TIME,
    )

    memory = LearningMemory(
        experience=experience,
        similarity_score=1.0,
        outcome_score=1.0,
        ranking_score=1.0,
        decay_score=1.0,
    )

    learning = LearningContext(
        task="Fix authentication",
        memories=[memory],
    )

    learning_validation = validate_learning_context(
        learning
    )

    strategy = StrategyDecision(
        strategy="repair",
        reason="Previous verified evidence.",
        confidence=0.8,
        supporting_memories=[memory],
        learning_guidance=[
            "A similar verified solution previously succeeded: "
            "repair authentication."
        ],
        validation_guidance=[
            "High-trust learning evidence is available."
        ],
        trusted_memory_count=1,
        rejected_memory_count=0,
        risks=[],
    )

    action = ActionDecision(
        action="inspect_files",
        reason="Inspect before repair.",
        confidence=0.8,
        strategy="repair",
        risks=[],
        learning_guidance=[],
        validation_guidance=[],
        trusted_memory_count=1,
        rejected_memory_count=0,
    )

    task_plan = TaskPlan(
        task="Fix authentication",
        items=[
            TaskItem(
                description="Inspect authentication code.",
                category="repair",
            )
        ],
    )

    impact = ImpactAnalysis(
        task="Fix authentication",
        items=[],
    )

    plan = CodePlan(
        task="Fix authentication",
        objective="Repair authentication behavior.",
        steps=[
            PlanStep(
                id="step-1",
                description="Inspect authentication code.",
                category="repair",
            )
        ],
    )

    ordering = OrderingResult(
        ordered_steps=plan.steps,
        missing_dependencies={},
        cycles=[],
    )

    validation = PlanValidationResult(
        valid=True,
        issues=[],
    )

    risk = RiskAnalysis(
        level="LOW",
        score=0,
        factors=[],
        recommendations=[],
    )

    result = IntegratedPlanningResult(
        task="Fix authentication",
        learning=learning,
        learning_validation=learning_validation,
        strategy=strategy,
        action=action,
        task_plan=task_plan,
        impact=impact,
        plan=plan,
        ordering=ordering,
        validation=validation,
        risk=risk,
    )

    report = build_planning_report(
        result
    )

    assert "Learning guidance:" in report
    assert "Learning validation:" in report
    assert "High-trust learning evidence" in report
    assert "Supporting strategy memories:" in report
    assert "repair authentication" in report
    assert "Plan executable: YES" in report
    assert "Plan safe: YES" in report
