from flycoder.actions.registry import ActionResult
from flycoder.state import CodingState
from flycoder.tools.learning_feedback import (
    LearningOutcome,
    record_learning_outcome,
)
from flycoder.tools.memory import (
    Experience,
    MemoryStore,
)


def make_state() -> CodingState:
    return CodingState(
        task="fix authentication error",
        task_intent="repair",
    )


def test_successful_learning_outcome_is_recorded():
    store = MemoryStore()

    outcome = LearningOutcome(
        task="fix authentication error",
        task_intent="repair",
        action="run_tests",
        success=True,
        message="Tests passed.",
        tests_run=True,
        tests_passed=True,
        repair_applied=True,
    )

    experience = record_learning_outcome(
        store,
        outcome,
        files=["auth.py"],
    )

    assert isinstance(experience, Experience)
    assert len(store.experiences) == 1
    assert experience.task == "fix authentication error"
    assert experience.action == "run_tests"
    assert experience.success is True
    assert experience.verified is True
    assert experience.files == ["auth.py"]


def test_failed_learning_outcome_is_recorded_as_failure():
    store = MemoryStore()

    outcome = LearningOutcome(
        task="fix authentication error",
        task_intent="repair",
        action="run_tests",
        success=False,
        message="Tests failed.",
    )

    experience = record_learning_outcome(
        store,
        outcome,
    )

    assert isinstance(experience, Experience)
    assert experience.success is False
    assert experience.verified is False
    assert experience.errors == ["Tests failed."]


def test_success_without_verification_is_not_marked_verified():
    store = MemoryStore()

    outcome = LearningOutcome(
        task="review authentication code",
        task_intent="review",
        action="review_code",
        success=True,
        message="Review completed.",
        tests_run=False,
        tests_passed=False,
        repair_applied=False,
    )

    experience = record_learning_outcome(
        store,
        outcome,
    )

    assert experience.success is True
    assert experience.verified is False


def test_passing_tests_without_repair_application_is_not_verified():
    store = MemoryStore()

    outcome = LearningOutcome(
        task="run existing tests",
        task_intent="inspect",
        action="run_tests",
        success=True,
        message="Tests passed.",
        tests_run=True,
        tests_passed=True,
        repair_applied=False,
    )

    experience = record_learning_outcome(
        store,
        outcome,
    )

    assert experience.success is True
    assert experience.verified is False


def test_strategy_is_recorded_as_memory_note():
    store = MemoryStore()

    outcome = LearningOutcome(
        task="improve parser",
        task_intent="improve",
        action="improve_code",
        success=True,
        strategy="targeted_repair",
        message="Improvement completed.",
    )

    experience = record_learning_outcome(
        store,
        outcome,
    )

    assert any(
        "Strategy used: targeted_repair." in note
        for note in experience.notes
    )


def test_confidence_is_recorded_as_memory_note():
    store = MemoryStore()

    outcome = LearningOutcome(
        task="review parser",
        task_intent="review",
        action="review_code",
        success=True,
        confidence_score=0.85,
        message="Review completed.",
    )

    experience = record_learning_outcome(
        store,
        outcome,
    )

    assert any(
        "Confidence score: 0.850." in note
        for note in experience.notes
    )


def test_recovery_action_is_recorded_as_memory_note():
    store = MemoryStore()

    outcome = LearningOutcome(
        task="fix failing tests",
        task_intent="repair",
        action="fix_error",
        success=False,
        recovery_action="read_file",
        message="Diagnosis failed.",
    )

    experience = record_learning_outcome(
        store,
        outcome,
    )

    assert any(
        "Recovery action: read_file." in note
        for note in experience.notes
    )


def test_guardrail_block_is_recorded_without_becoming_success():
    store = MemoryStore()

    outcome = LearningOutcome(
        task="modify protected file",
        task_intent="repair",
        action="write_code",
        success=False,
        guardrail_blocked=True,
        human_input_required=True,
        message="Repair approval required.",
    )

    experience = record_learning_outcome(
        store,
        outcome,
    )

    assert experience.success is False
    assert experience.verified is False
    assert any(
        "Guardrail blocked execution." in note
        for note in experience.notes
    )


def test_evidence_and_lessons_are_preserved():
    store = MemoryStore()

    outcome = LearningOutcome(
        task="run tests",
        task_intent="inspect",
        action="run_tests",
        success=True,
        message="Tests passed.",
        evidence=["Evidence A."],
        lessons=["Lesson B."],
    )

    experience = record_learning_outcome(
        store,
        outcome,
    )

    assert "Evidence A." in experience.notes
    assert "Lesson B." in experience.notes


def test_recording_does_not_modify_learning_outcome():
    store = MemoryStore()

    outcome = LearningOutcome(
        task="review code",
        task_intent="review",
        action="review_code",
        success=True,
        strategy="review_first",
        confidence_score=0.7,
        message="Review completed.",
    )

    before = {
        "task": outcome.task,
        "action": outcome.action,
        "success": outcome.success,
        "strategy": outcome.strategy,
        "confidence_score": outcome.confidence_score,
        "evidence": list(outcome.evidence),
        "lessons": list(outcome.lessons),
    }

    record_learning_outcome(
        store,
        outcome,
    )

    after = {
        "task": outcome.task,
        "action": outcome.action,
        "success": outcome.success,
        "strategy": outcome.strategy,
        "confidence_score": outcome.confidence_score,
        "evidence": list(outcome.evidence),
        "lessons": list(outcome.lessons),
    }

    assert after == before
