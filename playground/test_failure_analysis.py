from flycoder.tools.testing import (
    FailureAnalysisResult,
    TestExecutionResult,
    TestFailure,
    analyze_test_failures,
    build_failure_analysis_report,
)


def execution(
    output: str,
    passed: bool = False,
    return_code: int = 1,
    timed_out: bool = False,
) -> TestExecutionResult:
    return TestExecutionResult(
        command=["python", "-m", "pytest", "-q"],
        return_code=return_code,
        output=output,
        passed=passed,
        timed_out=timed_out,
    )


def test_success_has_no_failures():
    result = analyze_test_failures(
        execution(
            "1 passed in 0.02s",
            passed=True,
            return_code=0,
        )
    )

    assert result.passed is True
    assert result.failed is False
    assert result.failure_count == 0
    assert result.collection_error is False
    assert result.timed_out is False


def test_assertion_failure_is_parsed():
    output = """
============================= FAILURES =============================
____________________________ test_example ___________________________

tests/test_example.py:10: in test_example
    assert 1 == 2
E   AssertionError: expected values to match

=========================== short test summary info ================
FAILED tests/test_example.py::test_example - AssertionError: expected values to match
========================= 1 failed in 0.03s =========================
"""

    result = analyze_test_failures(execution(output))

    assert result.failure_count == 1

    failure = result.failures[0]

    assert failure.test_file == "tests/test_example.py:10"
    assert failure.test_name == "test_example"
    assert failure.error_type == "AssertionError"
    assert failure.message == "expected values to match"


def test_exception_failure_is_parsed():
    output = """
============================= FAILURES =============================
__________________________ test_runtime ______________________________

tests/test_runtime.py:5: in test_runtime
    raise ValueError("bad value")
E   ValueError: bad value

=========================== short test summary info ================
FAILED tests/test_runtime.py::test_runtime - ValueError: bad value
========================= 1 failed in 0.03s =========================
"""

    result = analyze_test_failures(execution(output))

    assert result.failure_count == 1

    failure = result.failures[0]

    assert failure.test_name == "test_runtime"
    assert failure.error_type == "ValueError"
    assert failure.message == "bad value"


def test_multiple_failures_are_parsed():
    output = """
============================= FAILURES =============================
____________________________ test_one _______________________________

tests/test_one.py:3: in test_one
E   AssertionError: one failed

____________________________ test_two _______________________________

tests/test_two.py:4: in test_two
E   RuntimeError: two failed

=========================== short test summary info ================
FAILED tests/test_one.py::test_one - AssertionError: one failed
FAILED tests/test_two.py::test_two - RuntimeError: two failed
========================= 2 failed in 0.04s =========================
"""

    result = analyze_test_failures(execution(output))

    assert result.failure_count == 2
    assert result.failures[0].test_name == "test_one"
    assert result.failures[1].test_name == "test_two"


def test_collection_error_is_detected():
    output = """
============================= ERRORS =============================
_____ ERROR collecting tests/test_broken.py _____
ImportError while importing test module
E   ModuleNotFoundError: No module named 'missing_package'
=========================== short test summary info ================
ERROR tests/test_broken.py
========================= 1 error in 0.02s =========================
"""

    result = analyze_test_failures(execution(output))

    assert result.failed is True
    assert result.collection_error is True


def test_timeout_is_detected():
    result = analyze_test_failures(
        execution(
            "pytest timed out after 1 seconds",
            timed_out=True,
            return_code=-1,
        )
    )

    assert result.failed is True
    assert result.timed_out is True
    assert result.failure_count == 0


def test_short_summary_fallback_parses_failure():
    output = """
=========================== short test summary info ================
FAILED tests/test_alpha.py::test_alpha - AssertionError: failed alpha
========================= 1 failed in 0.01s =========================
"""

    result = analyze_test_failures(execution(output))

    assert result.failure_count == 1
    assert result.failures[0].test_file == "tests/test_alpha.py"
    assert result.failures[0].test_name == "test_alpha"
    assert result.failures[0].message == "AssertionError: failed alpha"


def test_qualified_name_contains_file_and_test():
    failure = TestFailure(
        test_file="tests/test_example.py",
        test_name="test_example",
        error_type="AssertionError",
        message="failed",
    )

    assert failure.qualified_name == (
        "tests/test_example.py::test_example"
    )


def test_qualified_name_handles_missing_file():
    failure = TestFailure(
        test_file=None,
        test_name="test_example",
        error_type="ValueError",
        message="failed",
    )

    assert failure.qualified_name == "test_example"


def test_failure_report_contains_structured_data():
    analysis = FailureAnalysisResult(
        failures=[
            TestFailure(
                test_file="tests/test_example.py",
                test_name="test_example",
                error_type="AssertionError",
                message="failed",
            )
        ],
        passed=False,
        raw_output="failed",
    )

    report = build_failure_analysis_report(analysis)

    assert report["failed"] is True
    assert report["passed"] is False
    assert report["failure_count"] == 1
    assert report["failures"][0]["qualified_name"] == (
        "tests/test_example.py::test_example"
    )


def test_raw_output_is_preserved():
    output = "some pytest output"

    result = analyze_test_failures(execution(output))

    assert result.raw_output == output
