from pathlib import Path

from flycoder.tools.testing import (
    TestExecutionResult,
    build_test_execution_report,
    execute_tests,
)


def write_test_file(root: Path, name: str, content: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_execute_selected_tests(tmp_path):
    write_test_file(
        tmp_path,
        "tests/test_pass.py",
        """
def test_pass():
    assert True
""",
    )

    write_test_file(
        tmp_path,
        "tests/test_other.py",
        """
def test_other():
    assert False
""",
    )

    result = execute_tests(
        tmp_path,
        test_files=["tests/test_pass.py"],
    )

    assert result.passed is True
    assert result.failed is False
    assert result.return_code == 0
    assert result.timed_out is False
    assert "1 passed" in result.output


def test_execute_full_suite_when_no_files_are_supplied(tmp_path):
    write_test_file(
        tmp_path,
        "tests/test_one.py",
        """
def test_one():
    assert True
""",
    )

    write_test_file(
        tmp_path,
        "tests/test_two.py",
        """
def test_two():
    assert True
""",
    )

    result = execute_tests(tmp_path)

    assert result.passed is True
    assert result.return_code == 0
    assert "2 passed" in result.output


def test_failed_tests_are_reported(tmp_path):
    write_test_file(
        tmp_path,
        "tests/test_failure.py",
        """
def test_failure():
    assert False
""",
    )

    result = execute_tests(
        tmp_path,
        test_files=["tests/test_failure.py"],
    )

    assert result.passed is False
    assert result.failed is True
    assert result.return_code != 0
    assert result.timed_out is False
    assert "1 failed" in result.output


def test_result_contains_pytest_command(tmp_path):
    write_test_file(
        tmp_path,
        "tests/test_command.py",
        """
def test_command():
    assert True
""",
    )

    result = execute_tests(
        tmp_path,
        test_files=["tests/test_command.py"],
    )

    assert result.command[:4] == [
        result.command[0],
        "-m",
        "pytest",
        "-q",
    ]
    assert result.command[-1] == "tests/test_command.py"


def test_execution_records_duration(tmp_path):
    write_test_file(
        tmp_path,
        "tests/test_duration.py",
        """
def test_duration():
    assert True
""",
    )

    result = execute_tests(
        tmp_path,
        test_files=["tests/test_duration.py"],
    )

    assert result.duration_seconds >= 0


def test_execution_does_not_modify_source_file(tmp_path):
    source = tmp_path / "source.py"
    original = "VALUE = 42\n"
    source.write_text(original, encoding="utf-8")

    write_test_file(
        tmp_path,
        "tests/test_source.py",
        """
def test_source():
    assert True
""",
    )

    result = execute_tests(
        tmp_path,
        test_files=["tests/test_source.py"],
    )

    assert result.passed is True
    assert source.read_text(encoding="utf-8") == original


def test_timeout_is_reported(tmp_path):
    write_test_file(
        tmp_path,
        "tests/test_timeout.py",
        """
import time

def test_timeout():
    time.sleep(2)
""",
    )

    result = execute_tests(
        tmp_path,
        test_files=["tests/test_timeout.py"],
        timeout=1,
    )

    assert result.passed is False
    assert result.failed is True
    assert result.timed_out is True
    assert result.return_code == -1
    assert "timed out after 1 seconds" in result.output


def test_execution_report_contains_expected_fields(tmp_path):
    write_test_file(
        tmp_path,
        "tests/test_report.py",
        """
def test_report():
    assert True
""",
    )

    result = execute_tests(
        tmp_path,
        test_files=["tests/test_report.py"],
    )

    report = build_test_execution_report(result)

    assert report["passed"] is True
    assert report["failed"] is False
    assert report["timed_out"] is False
    assert report["return_code"] == 0
    assert isinstance(report["command"], list)
    assert isinstance(report["output"], str)
    assert isinstance(report["duration_seconds"], float)


def test_failed_property_is_inverse_of_passed():
    passed = TestExecutionResult(
        command=[],
        return_code=0,
        output="",
        passed=True,
    )

    failed = TestExecutionResult(
        command=[],
        return_code=1,
        output="",
        passed=False,
    )

    assert passed.failed is False
    assert failed.failed is True
