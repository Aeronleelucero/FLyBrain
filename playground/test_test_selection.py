from pathlib import Path

from flycoder.tools.testing import (
    TestDiscoveryResult,
    build_test_selection_report,
    select_tests,
)


def make_result(*files: str) -> TestDiscoveryResult:
    return TestDiscoveryResult(
        files=sorted(files),
        directories=[],
    )


def test_selects_matching_test_prefix():
    result = make_result(
        "playground/test_changes.py",
        "playground/test_agent.py",
    )

    selection = select_tests(
        result,
        changed_files=["flycoder/tools/changes.py"],
    )

    assert selection.selected == ["playground/test_changes.py"]
    assert selection.fallback is False


def test_selects_matching_test_suffix():
    result = make_result(
        "tests/changes_test.py",
        "tests/agent_test.py",
    )

    selection = select_tests(
        result,
        changed_files=["flycoder/tools/changes.py"],
    )

    assert selection.selected == ["tests/changes_test.py"]


def test_unrelated_file_is_not_selected():
    result = make_result(
        "playground/test_changes.py",
        "playground/test_agent.py",
    )

    selection = select_tests(
        result,
        changed_files=["flycoder/tools/filesystem.py"],
        fallback_to_all=False,
    )

    assert selection.selected == []
    assert selection.fallback is False


def test_explicit_test_name_is_selected():
    result = make_result(
        "playground/test_changes.py",
        "playground/test_agent.py",
    )

    selection = select_tests(
        result,
        test_names=["changes"],
        fallback_to_all=False,
    )

    assert selection.selected == ["playground/test_changes.py"]
    assert selection.reasons[0].score == 1000


def test_explicit_selection_has_priority():
    result = make_result(
        "playground/test_changes.py",
        "playground/test_agent.py",
    )

    selection = select_tests(
        result,
        changed_files=["flycoder/tools/agent.py"],
        test_names=["changes"],
        fallback_to_all=False,
    )

    assert selection.selected[0] == "playground/test_changes.py"


def test_multiple_changed_files_are_deduplicated():
    result = make_result(
        "playground/test_changes.py",
        "playground/test_agent.py",
    )

    selection = select_tests(
        result,
        changed_files=[
            "flycoder/tools/changes.py",
            "flycoder/tools/changes.py",
        ],
    )

    assert selection.selected.count("playground/test_changes.py") == 1


def test_selection_is_deterministic():
    result = make_result(
        "playground/test_zeta.py",
        "playground/test_alpha.py",
        "playground/test_changes.py",
    )

    first = select_tests(
        result,
        changed_files=[
            "flycoder/tools/changes.py",
            "flycoder/tools/agent.py",
        ],
    )

    second = select_tests(
        result,
        changed_files=[
            "flycoder/tools/changes.py",
            "flycoder/tools/agent.py",
        ],
    )

    assert first.selected == second.selected
    assert [
        reason.score for reason in first.reasons
    ] == [
        reason.score for reason in second.reasons
    ]


def test_falls_back_to_all_tests_when_no_match():
    result = make_result(
        "tests/test_alpha.py",
        "tests/test_beta.py",
    )

    selection = select_tests(
        result,
        changed_files=["flycoder/unrelated.py"],
    )

    assert selection.selected == [
        "tests/test_alpha.py",
        "tests/test_beta.py",
    ]
    assert selection.fallback is True


def test_can_disable_fallback():
    result = make_result(
        "tests/test_alpha.py",
        "tests/test_beta.py",
    )

    selection = select_tests(
        result,
        changed_files=["flycoder/unrelated.py"],
        fallback_to_all=False,
    )

    assert selection.selected == []
    assert selection.fallback is False


def test_empty_inputs_fall_back_to_all_tests():
    result = make_result(
        "tests/test_alpha.py",
        "tests/test_beta.py",
    )

    selection = select_tests(result)

    assert selection.selected == [
        "tests/test_alpha.py",
        "tests/test_beta.py",
    ]
    assert selection.fallback is True


def test_empty_inputs_without_fallback_select_nothing():
    result = make_result("tests/test_alpha.py")

    selection = select_tests(
        result,
        fallback_to_all=False,
    )

    assert selection.selected == []
    assert selection.fallback is False


def test_report_contains_selection_reasons():
    result = make_result("playground/test_changes.py")

    selection = select_tests(
        result,
        changed_files=["flycoder/tools/changes.py"],
    )

    report = build_test_selection_report(selection)

    assert report["selected_tests"] == [
        "playground/test_changes.py"
    ]
    assert report["test_count"] == 1
    assert report["fallback"] is False
    assert report["reasons"][0]["test_file"] == (
        "playground/test_changes.py"
    )
    assert report["reasons"][0]["score"] > 0


def test_selection_only_returns_discovered_files():
    result = make_result(
        "tests/test_changes.py",
    )

    selection = select_tests(
        result,
        changed_files=["flycoder/tools/changes.py"],
        test_names=["does_not_exist"],
        fallback_to_all=False,
    )

    assert all(
        path in result.files
        for path in selection.selected
    )
