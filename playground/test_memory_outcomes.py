from flycoder.tools.memory import (
    Experience,
    MemoryStore,
    record_failure,
    record_solution,
)


def test_record_solution_creates_successful_experience():
    store = MemoryStore()

    experience = record_solution(
        store,
        task="Fix authentication",
        action="Updated authentication middleware",
        outcome="Authentication tests passed",
    )

    assert isinstance(experience, Experience)
    assert experience.success is True
    assert experience.task == "Fix authentication"
    assert experience.action == "Updated authentication middleware"
    assert experience.outcome == "Authentication tests passed"


def test_record_solution_adds_experience_to_store():
    store = MemoryStore()

    experience = record_solution(
        store,
        task="Fix login",
        action="Updated validation",
        outcome="Login works",
    )

    assert store.experiences == [experience]


def test_record_solution_stores_files():
    store = MemoryStore()

    experience = record_solution(
        store,
        task="Fix API",
        action="Updated endpoint",
        outcome="API tests passed",
        files=["api.py", "auth.py"],
    )

    assert experience.files == ["api.py", "auth.py"]


def test_record_solution_stores_notes():
    store = MemoryStore()

    experience = record_solution(
        store,
        task="Fix scanner",
        action="Updated scanner",
        outcome="Scanner works",
        notes=["Kept existing scanner architecture"],
    )

    assert experience.notes == ["Kept existing scanner architecture"]


def test_record_solution_defaults_optional_fields():
    store = MemoryStore()

    experience = record_solution(
        store,
        task="Fix database",
        action="Updated query",
        outcome="Query passed",
    )

    assert experience.files == []
    assert experience.notes == []
    assert experience.errors == []


def test_record_failure_creates_failed_experience():
    store = MemoryStore()

    experience = record_failure(
        store,
        task="Fix database connection",
        action="Changed database host",
        outcome="Connection still failed",
        error="Connection refused",
    )

    assert isinstance(experience, Experience)
    assert experience.success is False
    assert experience.task == "Fix database connection"
    assert experience.action == "Changed database host"
    assert experience.outcome == "Connection still failed"


def test_record_failure_stores_error():
    store = MemoryStore()

    experience = record_failure(
        store,
        task="Fix database",
        action="Changed configuration",
        outcome="Connection failed",
        error="Connection refused",
    )

    assert experience.errors == ["Connection refused"]


def test_record_failure_stores_files_and_notes():
    store = MemoryStore()

    experience = record_failure(
        store,
        task="Fix authentication",
        action="Changed middleware",
        outcome="Tests failed",
        error="Unauthorized response",
        files=["auth.py"],
        notes=["Existing token handling was preserved"],
    )

    assert experience.files == ["auth.py"]
    assert experience.notes == ["Existing token handling was preserved"]


def test_record_failure_without_error_has_empty_errors():
    store = MemoryStore()

    experience = record_failure(
        store,
        task="Fix UI",
        action="Changed layout",
        outcome="Layout still incorrect",
    )

    assert experience.success is False
    assert experience.errors == []


def test_solution_and_failure_can_coexist():
    store = MemoryStore()

    solution = record_solution(
        store,
        task="Fix authentication",
        action="Updated middleware",
        outcome="Tests passed",
    )

    failure = record_failure(
        store,
        task="Fix database",
        action="Changed connection settings",
        outcome="Connection failed",
        error="Connection refused",
    )

    assert store.experiences == [solution, failure]
    assert store.experiences[0].success is True
    assert store.experiences[1].success is False


def test_solution_and_failure_are_retrievable_by_task():
    store = MemoryStore()

    solution = record_solution(
        store,
        task="Fix authentication middleware",
        action="Updated middleware",
        outcome="Tests passed",
    )

    failure = record_failure(
        store,
        task="Fix database connection",
        action="Changed configuration",
        outcome="Still failed",
        error="Connection refused",
    )

    assert store.find_experiences("authentication") == [solution]
    assert store.find_experiences("database") == [failure]
