from flycoder.tools.memory import (
    Experience,
    MemoryStore,
    build_memory_report,
)


def test_experience_can_be_created():
    experience = Experience(
        task="Fix authentication bug",
        action="Updated auth middleware",
        outcome="Authentication succeeds",
        success=True,
    )

    assert experience.task == "Fix authentication bug"
    assert experience.action == "Updated auth middleware"
    assert experience.outcome == "Authentication succeeds"
    assert experience.success is True


def test_experience_defaults_are_empty_lists():
    experience = Experience(
        task="Run tests",
        action="Executed pytest",
        outcome="Tests passed",
        success=True,
    )

    assert experience.files == []
    assert experience.errors == []
    assert experience.notes == []


def test_failed_experience_can_record_errors():
    experience = Experience(
        task="Fix database connection",
        action="Changed database configuration",
        outcome="Connection still fails",
        success=False,
        errors=["Connection refused"],
    )

    assert experience.success is False
    assert experience.errors == ["Connection refused"]


def test_memory_store_starts_empty():
    store = MemoryStore()

    assert store.experiences == []


def test_record_experience():
    store = MemoryStore()
    experience = Experience(
        task="Fix login",
        action="Updated login validation",
        outcome="Login works",
        success=True,
    )

    recorded = store.record_experience(experience)

    assert recorded is experience
    assert store.experiences == [experience]


def test_multiple_experiences_are_preserved_in_order():
    store = MemoryStore()

    first = Experience(
        task="Fix login",
        action="Updated validation",
        outcome="Passed",
        success=True,
    )
    second = Experience(
        task="Fix database",
        action="Updated query",
        outcome="Failed",
        success=False,
    )

    store.record_experience(first)
    store.record_experience(second)

    assert store.experiences == [first, second]


def test_find_experiences_matches_task_case_insensitively():
    store = MemoryStore()

    login = Experience(
        task="Fix Login Authentication",
        action="Updated middleware",
        outcome="Passed",
        success=True,
    )
    database = Experience(
        task="Fix Database Connection",
        action="Updated configuration",
        outcome="Passed",
        success=True,
    )

    store.record_experience(login)
    store.record_experience(database)

    results = store.find_experiences("login")

    assert results == [login]


def test_find_experiences_supports_partial_task_matches():
    store = MemoryStore()

    experience = Experience(
        task="Fix authentication middleware",
        action="Updated middleware",
        outcome="Passed",
        success=True,
    )

    store.record_experience(experience)

    results = store.find_experiences("authentication")

    assert results == [experience]


def test_find_experiences_returns_empty_for_unknown_task():
    store = MemoryStore()

    store.record_experience(
        Experience(
            task="Fix authentication",
            action="Updated middleware",
            outcome="Passed",
            success=True,
        )
    )

    assert store.find_experiences("database") == []


def test_find_experiences_returns_empty_for_blank_query():
    store = MemoryStore()

    store.record_experience(
        Experience(
            task="Fix authentication",
            action="Updated middleware",
            outcome="Passed",
            success=True,
        )
    )

    assert store.find_experiences("   ") == []


def test_memory_report_counts_successes_and_failures():
    store = MemoryStore()

    store.record_experience(
        Experience(
            task="Fix login",
            action="Updated auth",
            outcome="Passed",
            success=True,
        )
    )
    store.record_experience(
        Experience(
            task="Fix database",
            action="Updated connection",
            outcome="Failed",
            success=False,
        )
    )

    report = build_memory_report(store)

    assert report["experience_count"] == 2
    assert report["successful_count"] == 1
    assert report["failed_count"] == 1


def test_memory_report_preserves_experience_details():
    store = MemoryStore()

    store.record_experience(
        Experience(
            task="Fix scanner",
            action="Updated scanner service",
            outcome="Tests passed",
            success=True,
            files=["scanner.py"],
            errors=[],
            notes=["Used existing service"],
        )
    )

    report = build_memory_report(store)

    assert report["experiences"] == [
        {
            "task": "Fix scanner",
            "action": "Updated scanner service",
            "outcome": "Tests passed",
            "success": True,
            "files": ["scanner.py"],
            "errors": [],
            "notes": ["Used existing service"],
        }
    ]


def test_empty_memory_report_is_deterministic():
    store = MemoryStore()

    assert build_memory_report(store) == {
        "experience_count": 0,
        "successful_count": 0,
        "failed_count": 0,
        "experiences": [],
    }
