from flycoder.tools.memory import (
    Experience,
    MemoryStore,
    Outcome,
    record_outcome,
)


def test_outcome_can_be_created():
    outcome = Outcome(
        status="success",
        summary="Authentication tests passed",
        tests_passed=12,
        tests_failed=0,
        verified=True,
    )

    assert outcome.status == "success"
    assert outcome.summary == "Authentication tests passed"
    assert outcome.tests_passed == 12
    assert outcome.tests_failed == 0
    assert outcome.errors == []
    assert outcome.verified is True


def test_outcome_defaults_are_safe():
    outcome = Outcome(
        status="failure",
        summary="Tests failed",
    )

    assert outcome.tests_passed == 0
    assert outcome.tests_failed == 0
    assert outcome.errors == []
    assert outcome.verified is False


def test_record_successful_outcome_creates_successful_experience():
    store = MemoryStore()

    outcome = Outcome(
        status="success",
        summary="Authentication tests passed",
        tests_passed=15,
        tests_failed=0,
        verified=True,
    )

    experience = record_outcome(
        store,
        task="Fix authentication",
        action="Update authentication middleware",
        outcome=outcome,
    )

    assert isinstance(experience, Experience)
    assert experience.success is True
    assert experience.outcome == "Authentication tests passed"


def test_record_successful_outcome_preserves_files():
    store = MemoryStore()

    outcome = Outcome(
        status="success",
        summary="Scanner verification passed",
        tests_passed=8,
        verified=True,
    )

    experience = record_outcome(
        store,
        task="Fix scanner",
        action="Update scanner service",
        outcome=outcome,
        files=["scanner.py", "camera.py"],
    )

    assert experience.files == [
        "scanner.py",
        "camera.py",
    ]


def test_record_successful_outcome_preserves_notes():
    store = MemoryStore()

    outcome = Outcome(
        status="success",
        summary="Tests passed",
        tests_passed=10,
        verified=True,
    )

    experience = record_outcome(
        store,
        task="Fix API",
        action="Update endpoint",
        outcome=outcome,
        notes=["Kept existing API contract"],
    )

    assert experience.notes == [
        "Kept existing API contract",
    ]


def test_record_failed_outcome_creates_failed_experience():
    store = MemoryStore()

    outcome = Outcome(
        status="failure",
        summary="Authentication tests failed",
        tests_passed=8,
        tests_failed=2,
        errors=["Unauthorized response"],
        verified=False,
    )

    experience = record_outcome(
        store,
        task="Fix authentication",
        action="Update middleware",
        outcome=outcome,
    )

    assert experience.success is False
    assert experience.outcome == "Authentication tests failed"


def test_record_failed_outcome_preserves_errors():
    store = MemoryStore()

    outcome = Outcome(
        status="failure",
        summary="Database tests failed",
        tests_passed=3,
        tests_failed=1,
        errors=[
            "Connection refused",
            "Database unavailable",
        ],
        verified=False,
    )

    experience = record_outcome(
        store,
        task="Fix database connection",
        action="Update database configuration",
        outcome=outcome,
    )

    assert experience.errors == [
        "Connection refused",
        "Database unavailable",
    ]


def test_unverified_success_status_is_not_successful():
    store = MemoryStore()

    outcome = Outcome(
        status="success",
        summary="Change completed",
        tests_passed=10,
        tests_failed=0,
        verified=False,
    )

    experience = record_outcome(
        store,
        task="Fix authentication",
        action="Update middleware",
        outcome=outcome,
    )

    assert experience.success is False


def test_failed_tests_prevent_success():
    store = MemoryStore()

    outcome = Outcome(
        status="success",
        summary="Some tests passed",
        tests_passed=9,
        tests_failed=1,
        verified=True,
    )

    experience = record_outcome(
        store,
        task="Fix authentication",
        action="Update middleware",
        outcome=outcome,
    )

    assert experience.success is False


def test_non_success_status_is_not_successful():
    store = MemoryStore()

    outcome = Outcome(
        status="partial",
        summary="Partially completed",
        tests_passed=5,
        tests_failed=0,
        verified=True,
    )

    experience = record_outcome(
        store,
        task="Improve API",
        action="Update endpoint",
        outcome=outcome,
    )

    assert experience.success is False


def test_record_outcome_adds_experience_to_store():
    store = MemoryStore()

    outcome = Outcome(
        status="success",
        summary="Tests passed",
        tests_passed=5,
        verified=True,
    )

    experience = record_outcome(
        store,
        task="Fix scanner",
        action="Update scanner",
        outcome=outcome,
    )

    assert store.experiences == [experience]


def test_record_outcome_does_not_modify_outcome():
    store = MemoryStore()

    outcome = Outcome(
        status="success",
        summary="Tests passed",
        tests_passed=5,
        tests_failed=0,
        errors=[],
        verified=True,
    )

    before = {
        "status": outcome.status,
        "summary": outcome.summary,
        "tests_passed": outcome.tests_passed,
        "tests_failed": outcome.tests_failed,
        "errors": list(outcome.errors),
        "verified": outcome.verified,
    }

    record_outcome(
        store,
        task="Fix scanner",
        action="Update scanner",
        outcome=outcome,
    )

    assert {
        "status": outcome.status,
        "summary": outcome.summary,
        "tests_passed": outcome.tests_passed,
        "tests_failed": outcome.tests_failed,
        "errors": list(outcome.errors),
        "verified": outcome.verified,
    } == before
