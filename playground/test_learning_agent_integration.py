from pathlib import Path

from flycoder.agent import FlyCoderAgent
from flycoder.state import CodingState
from flycoder.tools.memory import MemoryStore


def test_agent_owns_memory_store(tmp_path: Path):
    agent = FlyCoderAgent(tmp_path)

    assert isinstance(agent.memory_store, MemoryStore)
    assert agent.memory_store.experiences == []


def test_agent_accepts_existing_memory_store(tmp_path: Path):
    store = MemoryStore()

    agent = FlyCoderAgent(
        tmp_path,
        memory_store=store,
    )

    assert agent.memory_store is store


def test_run_once_records_learning_feedback(tmp_path: Path):
    (tmp_path / "example.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )

    store = MemoryStore()

    agent = FlyCoderAgent(
        tmp_path,
        memory_store=store,
    )

    state = CodingState(
        task="Inspect the project",
    )

    result = agent.run_once(state)

    assert result.action == "inspect_files"
    assert result.success is True

    assert len(store.experiences) == 1

    experience = store.experiences[0]

    assert experience.task == "Inspect the project"
    assert experience.action == "inspect_files"
    assert experience.success is True
    assert "example.py" in experience.files


def test_learning_feedback_is_attached_to_result(
    tmp_path: Path,
):
    (tmp_path / "example.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )

    agent = FlyCoderAgent(tmp_path)

    state = CodingState(
        task="Inspect the project",
    )

    result = agent.run_once(state)

    assert isinstance(result.data, dict)

    feedback = result.data["learning_feedback"]

    assert feedback["task"] == "Inspect the project"
    assert feedback["action"] == "inspect_files"
    assert feedback["success"] is True
    assert feedback["experience_recorded"] is True
    assert feedback["memory_size"] == 1


def test_multiple_agent_steps_share_same_memory_store(
    tmp_path: Path,
):
    (tmp_path / "example.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )

    store = MemoryStore()

    agent = FlyCoderAgent(
        tmp_path,
        memory_store=store,
    )

    state = CodingState(
        task="run tests",
    )

    first = agent.run_once(state)

    assert first.action == "inspect_files"
    assert len(store.experiences) == 1

    second = agent.run_once(state)

    assert second.action == "run_tests"
    assert len(store.experiences) == 2

    assert store.experiences[0].action == "inspect_files"
    assert store.experiences[1].action == "run_tests"


def test_learning_does_not_modify_state(
    tmp_path: Path,
):
    (tmp_path / "example.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )

    store = MemoryStore()

    agent = FlyCoderAgent(
        tmp_path,
        memory_store=store,
    )

    state = CodingState(
        task="Inspect the project",
    )

    result = agent.run_once(state)

    assert result.success is True

    assert state.task == "Inspect the project"
    assert state.task_intent == "inspect"
    assert state.files == ["example.py"]
    assert state.finished is False

    # Learning records the actual action,
    # not the finish control state.
    assert len(store.experiences) == 1
    assert store.experiences[0].action == "inspect_files"
