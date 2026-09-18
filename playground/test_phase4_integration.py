from flycoder.actions.analysis_actions import review_code_action
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace


def test_review_code_action_populates_phase4_intelligence():
    source = """
from pathlib import Path


class Processor:
    def __init__(self, value):
        self.value = value

    def process(self, items):
        results = []

        for item in items:
            if item > self.value:
                results.append(item)

        return results


def helper(value):
    if value:
        return value + 1

    return 0
"""

    state = CodingState(
        task="Analyze the selected Python file.",
        current_file="example.py",
        current_file_content=source,
    )

    workspace = Workspace(".")

    result = review_code_action(
        workspace,
        state,
    )

    assert result.success is True

    # Phase 3
    assert state.symbols
    assert state.symbol_imports
    assert state.symbol_calls
    assert state.symbol_relationships
    assert state.symbol_graph is not None

    # Phase 4
    assert state.control_flow
    assert state.symbol_references
    assert state.data_flow
    assert state.complexity_metrics
    assert state.code_paths


def test_review_code_action_returns_phase4_counts():
    source = """
def calculate(value):
    if value:
        return helper(value)

    return 0
"""

    state = CodingState(
        task="Review this file.",
        current_file="calculator.py",
        current_file_content=source,
    )

    workspace = Workspace(".")

    result = review_code_action(
        workspace,
        state,
    )

    assert result.success is True

    counts = result.data["code_intelligence_counts"]

    assert counts["control_flow"] > 0
    assert counts["references"] > 0
    assert counts["data_flow"] > 0
    assert counts["complexity"] > 0
    assert counts["code_paths"] > 0


def test_review_code_action_returns_unified_analysis_counts():
    source = """
def calculate(value):
    if value:
        return helper(value)

    return 0
"""

    state = CodingState(
        task="Review this file.",
        current_file="calculator.py",
        current_file_content=source,
    )

    workspace = Workspace(".")

    result = review_code_action(
        workspace,
        state,
    )

    counts = result.data["analysis_counts"]

    expected_keys = {
        "symbols",
        "imports",
        "calls",
        "inheritance",
        "decorators",
        "relationships",
        "control_flow",
        "references",
        "data_flow",
        "complexity",
        "code_paths",
    }

    assert set(counts) == expected_keys

    for value in counts.values():
        assert isinstance(value, int)
        assert value >= 0


def test_reanalyzing_same_file_does_not_duplicate_phase4_data():
    source = """
def calculate(value):
    if value:
        return helper(value)

    return 0
"""

    state = CodingState(
        task="Review this file.",
        current_file="calculator.py",
        current_file_content=source,
    )

    workspace = Workspace(".")

    first = review_code_action(
        workspace,
        state,
    )

    assert first.success is True

    first_counts = {
        "control_flow": len(state.control_flow),
        "references": len(state.symbol_references),
        "data_flow": len(state.data_flow),
        "complexity": len(state.complexity_metrics),
        "code_paths": len(state.code_paths),
    }

    second = review_code_action(
        workspace,
        state,
    )

    assert second.success is True

    second_counts = {
        "control_flow": len(state.control_flow),
        "references": len(state.symbol_references),
        "data_flow": len(state.data_flow),
        "complexity": len(state.complexity_metrics),
        "code_paths": len(state.code_paths),
    }

    assert second_counts == first_counts


def test_analyzing_multiple_files_preserves_previous_file():
    first_source = """
def first(value):
    if value:
        return value
    return 0
"""

    second_source = """
def second(items):
    for item in items:
        print(item)
"""

    state = CodingState(
        task="Review multiple files.",
    )

    workspace = Workspace(".")

    state.current_file = "first.py"
    state.current_file_content = first_source

    first_result = review_code_action(
        workspace,
        state,
    )

    assert first_result.success is True

    first_control_flow = [
        item
        for item in state.control_flow
        if item.file == "first.py"
    ]

    first_complexity = [
        item
        for item in state.complexity_metrics
        if item.file == "first.py"
    ]

    assert first_control_flow
    assert first_complexity

    state.current_file = "second.py"
    state.current_file_content = second_source

    second_result = review_code_action(
        workspace,
        state,
    )

    assert second_result.success is True

    assert any(
        item.file == "first.py"
        for item in state.control_flow
    )

    assert any(
        item.file == "second.py"
        for item in state.control_flow
    )

    assert any(
        item.file == "first.py"
        for item in state.complexity_metrics
    )

    assert any(
        item.file == "second.py"
        for item in state.complexity_metrics
    )
