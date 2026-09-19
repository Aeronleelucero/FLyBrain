"""Testing and verification tools for FLY-CODER."""

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestDiscoveryResult:
    """Represent tests discovered inside a workspace."""

    __test__ = False

    files: list[str] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Return the number of discovered test files."""
        return len(self.files)


@dataclass
class TestSelectionReason:
    """Explain why a test was selected."""

    __test__ = False

    test_file: str
    reason: str
    score: int = 0


@dataclass
class TestSelectionResult:
    """Represent tests selected for a code change."""

    __test__ = False

    selected: list[str] = field(default_factory=list)
    reasons: list[TestSelectionReason] = field(default_factory=list)
    fallback: bool = False

    @property
    def count(self) -> int:
        """Return the number of selected tests."""
        return len(self.selected)


@dataclass
class TestExecutionResult:
    """Represent the result of executing pytest."""

    __test__ = False

    command: list[str]
    return_code: int
    output: str
    passed: bool
    timed_out: bool = False
    duration_seconds: float = 0.0

    @property
    def failed(self) -> bool:
        """Return whether test execution failed."""
        return not self.passed


@dataclass
class TestFailure:
    """Represent one pytest test failure."""

    __test__ = False

    test_file: str | None
    test_name: str | None
    error_type: str | None
    message: str
    traceback: str = ""

    @property
    def qualified_name(self) -> str | None:
        """Return the test's file/name representation."""
        if self.test_file and self.test_name:
            return f"{self.test_file}::{self.test_name}"

        if self.test_name:
            return self.test_name

        return self.test_file


@dataclass
class FailureAnalysisResult:
    """Represent structured analysis of test execution failures."""

    __test__ = False

    failures: list[TestFailure] = field(default_factory=list)
    collection_error: bool = False
    timed_out: bool = False
    passed: bool = False
    raw_output: str = ""

    @property
    def failure_count(self) -> int:
        """Return the number of individual test failures."""
        return len(self.failures)

    @property
    def failed(self) -> bool:
        """Return whether the execution represents a failure."""
        return (
            self.timed_out
            or self.collection_error
            or bool(self.failures)
        )


_IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
}

_TEST_DIRECTORY_NAMES = {
    "test",
    "tests",
}

_TEST_FILE_PATTERNS = (
    "test_*.py",
    "*_test.py",
)

_PYTEST_FAILURE_HEADER = re.compile(
    r"^FAILED\s+"
    r"(?P<node>.+?)::"
    r"(?P<name>\S+)"
    r"(?:\s+-\s+(?P<message>.*))?$",
    re.MULTILINE,
)

_PYTEST_LOCATION_PATTERN = re.compile(
    r"^(?P<file>.+?):"
    r"(?P<line>\d+):\s+in\s+(?P<name>\S+)"
    r"\s*$"
)


def _is_ignored(path: Path) -> bool:
    """Return whether a path contains an ignored directory."""
    return any(
        directory in _IGNORED_DIRECTORIES
        for directory in path.parts
    )


def _is_test_file(path: Path) -> bool:
    """Return whether a path is a Python test file."""
    return path.is_file() and any(
        path.match(pattern)
        for pattern in _TEST_FILE_PATTERNS
    )


def _is_test_directory(path: Path) -> bool:
    """Return whether a path is a conventional test directory."""
    return (
        path.is_dir()
        and path.name.lower() in _TEST_DIRECTORY_NAMES
    )


def discover_tests(root: str | Path) -> TestDiscoveryResult:
    """Discover Python test files and conventional test directories."""
    workspace_root = Path(root).resolve()

    if not workspace_root.exists():
        return TestDiscoveryResult()

    files: set[str] = set()
    directories: set[str] = set()

    for path in workspace_root.rglob("*"):
        relative_path = path.relative_to(workspace_root)

        if _is_ignored(relative_path):
            continue

        if _is_test_directory(path):
            directories.add(relative_path.as_posix())

        if _is_test_file(path):
            files.add(relative_path.as_posix())

    return TestDiscoveryResult(
        files=sorted(files),
        directories=sorted(directories),
    )


def find_tests_by_name(
    result: TestDiscoveryResult,
    name: str,
) -> list[str]:
    """Find discovered tests whose filename contains a name."""
    normalized_name = name.strip().lower()

    if not normalized_name:
        return []

    return [
        path
        for path in result.files
        if normalized_name in Path(path).stem.lower()
    ]


def _normalize_path(path: str) -> str:
    """Normalize a path for deterministic matching."""
    return path.replace("\\", "/").strip("/").lower()


def _path_stems(path: str) -> set[str]:
    """Return useful normalized stems for a source/test path."""
    normalized = _normalize_path(path)
    filename = Path(normalized).name
    stem = Path(filename).stem

    stems = {
        stem.lower(),
    }

    if filename.endswith(".py"):
        stem = filename[:-3]
        stems.add(stem.lower())

    if stem.startswith("test_"):
        stems.add(stem[5:])

    if stem.endswith("_test"):
        stems.add(stem[:-5])

    return {
        value
        for value in stems
        if value
    }


def _test_match_score(
    test_file: str,
    changed_file: str,
) -> int:
    """Calculate a deterministic relevance score."""
    test_normalized = _normalize_path(test_file)
    changed_normalized = _normalize_path(changed_file)

    test_stems = _path_stems(test_normalized)
    changed_stems = _path_stems(changed_normalized)

    score = 0

    if test_normalized == changed_normalized:
        score += 100

    changed_filename = Path(changed_normalized).name
    test_filename = Path(test_normalized).name
    source_stem = Path(changed_filename).stem.lower()

    if (
        test_filename == f"test_{source_stem}.py"
        or test_filename == f"{source_stem}_test.py"
    ):
        score += 80

    if test_stems & changed_stems:
        score += 50

    changed_parts = set(
        Path(changed_normalized).parts[:-1]
    )

    test_parts = set(
        Path(test_normalized).parts[:-1]
    )

    common_parts = changed_parts & test_parts

    score += min(
        len(common_parts) * 5,
        20,
    )

    return score


def _reason_for_score(
    test_file: str,
    changed_file: str,
    score: int,
) -> str:
    """Build a human-readable selection reason."""
    source_stem = Path(
        _normalize_path(changed_file)
    ).stem.lower()

    test_filename = Path(
        _normalize_path(test_file)
    ).name

    if test_filename in {
        f"test_{source_stem}.py",
        f"{source_stem}_test.py",
    }:
        return (
            f"test filename matches changed file "
            f"'{changed_file}'"
        )

    if score >= 50:
        return (
            f"test name is related to changed file "
            f"'{changed_file}'"
        )

    return (
        f"test is located near changed file "
        f"'{changed_file}'"
    )


def select_tests(
    result: TestDiscoveryResult,
    changed_files: list[str] | None = None,
    test_names: list[str] | None = None,
    fallback_to_all: bool = True,
) -> TestSelectionResult:
    """
    Select relevant tests for changed files and explicit test names.

    Selection is deterministic and does not execute tests.
    """
    changed_files = changed_files or []
    test_names = test_names or []

    candidates: dict[str, TestSelectionReason] = {}

    # Explicit test names have the strongest priority.
    for name in test_names:
        for test_file in find_tests_by_name(
            result,
            name,
        ):
            candidates[test_file] = TestSelectionReason(
                test_file=test_file,
                reason=f"explicit test selection '{name}'",
                score=1000,
            )

    # Select tests related to changed files.
    for changed_file in changed_files:
        for test_file in result.files:
            score = _test_match_score(
                test_file,
                changed_file,
            )

            if score <= 0:
                continue

            reason = _reason_for_score(
                test_file,
                changed_file,
                score,
            )

            existing = candidates.get(test_file)

            if existing is None or score > existing.score:
                candidates[test_file] = TestSelectionReason(
                    test_file=test_file,
                    reason=reason,
                    score=score,
                )

    if not candidates and fallback_to_all and result.files:
        selected = list(result.files)

        reasons = [
            TestSelectionReason(
                test_file=test_file,
                reason=(
                    "no directly relevant tests found; "
                    "using full discovered test set"
                ),
                score=0,
            )
            for test_file in selected
        ]

        return TestSelectionResult(
            selected=selected,
            reasons=reasons,
            fallback=True,
        )

    ordered_reasons = sorted(
        candidates.values(),
        key=lambda item: (
            -item.score,
            item.test_file,
        ),
    )

    return TestSelectionResult(
        selected=[
            item.test_file
            for item in ordered_reasons
        ],
        reasons=ordered_reasons,
        fallback=False,
    )


def execute_tests(
    workspace: str | Path,
    test_files: list[str] | None = None,
    timeout: int = 60,
) -> TestExecutionResult:
    """
    Execute pytest inside a workspace.

    When test_files is supplied, only those test paths are executed.
    When it is omitted or empty, pytest executes the full discovered suite.

    This function only executes tests. It does not modify source files or
    attempt automatic repairs.
    """
    workspace_root = Path(workspace).resolve()

    environment = os.environ.copy()
    project_root = str(workspace_root.parent)

    existing_python_path = environment.get(
        "PYTHONPATH"
    )

    if existing_python_path:
        environment["PYTHONPATH"] = (
            f"{project_root}{os.pathsep}"
            f"{existing_python_path}"
        )
    else:
        environment["PYTHONPATH"] = project_root

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
    ]

    if test_files:
        command.extend(test_files)

    start_time = time.monotonic()

    try:
        completed = subprocess.run(
            command,
            cwd=workspace_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        duration_seconds = (
            time.monotonic() - start_time
        )

        output = "\n".join(
            part
            for part in (
                completed.stdout,
                completed.stderr,
            )
            if part
        ).strip()

        return TestExecutionResult(
            command=command,
            return_code=completed.returncode,
            output=output,
            passed=completed.returncode == 0,
            timed_out=False,
            duration_seconds=duration_seconds,
        )

    except subprocess.TimeoutExpired as exc:
        duration_seconds = (
            time.monotonic() - start_time
        )

        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

        if isinstance(stdout, bytes):
            stdout = stdout.decode(
                errors="replace"
            )

        if isinstance(stderr, bytes):
            stderr = stderr.decode(
                errors="replace"
            )

        output = "\n".join(
            part
            for part in (
                stdout,
                stderr,
                f"pytest timed out after {timeout} seconds",
            )
            if part
        ).strip()

        return TestExecutionResult(
            command=command,
            return_code=-1,
            output=output,
            passed=False,
            timed_out=True,
            duration_seconds=duration_seconds,
        )


def _extract_failure_blocks(
    output: str,
) -> list[str]:
    """
    Extract individual pytest failure blocks.

    Pytest normally reports a failure location like:

        tests/test_example.py:10: in test_example

    We use those locations as reliable boundaries between failures.
    """
    match = re.search(
        r"={5,}\s+FAILURES\s+={5,}",
        output,
    )

    if not match:
        return []

    section = output[match.end():]

    end_match = re.search(
        r"\n={5,}\s+"
        r"(?:short test summary info|short test summary)"
        r"\s+={5,}",
        section,
    )

    if end_match:
        section = section[:end_match.start()]

    lines = section.splitlines()

    blocks: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()

        location_match = _PYTEST_LOCATION_PATTERN.match(
            stripped
        )

        if location_match:
            if current:
                blocks.append(
                    "\n".join(current).strip()
                )

            current = [line]
            continue

        if current:
            current.append(line)

    if current:
        blocks.append(
            "\n".join(current).strip()
        )

    return [
        block
        for block in blocks
        if block
    ]


def _parse_failure_block(
    block: str,
) -> TestFailure | None:
    """Parse one pytest failure block."""
    lines = block.splitlines()

    if not lines:
        return None

    test_file = None
    test_name = None
    location_index = None

    # Standard pytest traceback location:
    #
    # tests/test_example.py:10: in test_example
    #
    for index, line in enumerate(lines):
        stripped = line.strip()

        location_match = _PYTEST_LOCATION_PATTERN.match(
            stripped
        )

        if location_match:
            test_file = (
                f"{location_match.group('file')}:"
                f"{location_match.group('line')}"
            )
            test_name = location_match.group("name")
            location_index = index
            break

    error_type = None
    message = ""
    error_line_index = None

    for index, line in enumerate(lines):
        stripped = line.strip()

        # Pytest prefixes traceback lines with "E   ".
        #
        # Convert:
        #
        #     E   AssertionError: expected values to match
        #
        # into:
        #
        #     AssertionError: expected values to match
        #
        normalized = re.sub(
            r"^E\s+",
            "",
            stripped,
        )

        assertion_match = re.match(
            r"^AssertionError:\s*(?P<message>.*)$",
            normalized,
        )

        if assertion_match:
            error_type = "AssertionError"
            message = assertion_match.group("message")
            error_line_index = index
            break

        error_match = re.match(
            r"^(?P<type>"
            r"[A-Za-z_][A-Za-z0-9_.]*"
            r"(?:Error|Exception|Failure)"
            r")"
            r":\s*(?P<message>.*)$",
            normalized,
        )

        if error_match:
            error_type = error_match.group("type")
            message = error_match.group("message")
            error_line_index = index
            break

    # If pytest did not provide a traceback location,
    # use the FAILED summary node as a fallback.
    if test_file is None or test_name is None:
        summary_match = _PYTEST_FAILURE_HEADER.search(
            block
        )

        if summary_match:
            test_file = (
                summary_match.group("node").strip()
            )
            test_name = (
                summary_match.group("name").strip()
            )

    # If no explicit exception message was found,
    # use the last meaningful line.
    if not message:
        for line in reversed(lines):
            stripped = line.strip()

            if not stripped:
                continue

            normalized = re.sub(
                r"^E\s+",
                "",
                stripped,
            )

            message = normalized
            break

    traceback_start = (
        location_index
        if location_index is not None
        else 0
    )

    if error_line_index is not None:
        traceback_end = error_line_index + 1
    else:
        traceback_end = len(lines)

    traceback_lines = lines[
        traceback_start:traceback_end
    ]

    traceback = "\n".join(
        traceback_lines
    ).strip()

    return TestFailure(
        test_file=test_file,
        test_name=test_name,
        error_type=error_type,
        message=message,
        traceback=traceback,
    )


def analyze_test_failures(
    result: TestExecutionResult,
) -> FailureAnalysisResult:
    """
    Analyze a pytest execution result.

    This function only interprets captured output. It does not execute tests.
    """
    if result.passed:
        return FailureAnalysisResult(
            failures=[],
            collection_error=False,
            timed_out=False,
            passed=True,
            raw_output=result.output,
        )

    if result.timed_out:
        return FailureAnalysisResult(
            failures=[],
            collection_error=False,
            timed_out=True,
            passed=False,
            raw_output=result.output,
        )

    output = result.output

    collection_error = (
        "ERROR collecting " in output
        or "ImportError while importing test module" in output
        or "ModuleNotFoundError" in output
    )

    failures: list[TestFailure] = []

    for block in _extract_failure_blocks(output):
        failure = _parse_failure_block(block)

        if failure is not None:
            failures.append(failure)

    # Pytest may provide failure information only in the
    # short summary. Preserve a useful fallback.
    if not failures:
        for match in _PYTEST_FAILURE_HEADER.finditer(
            output
        ):
            test_file = (
                match.group("node").strip()
            )

            test_name = (
                match.group("name").strip()
            )

            message = (
                match.group("message") or ""
            ).strip()

            failures.append(
                TestFailure(
                    test_file=test_file,
                    test_name=test_name,
                    error_type=None,
                    message=message,
                )
            )

    return FailureAnalysisResult(
        failures=failures,
        collection_error=collection_error,
        timed_out=False,
        passed=False,
        raw_output=output,
    )


def build_test_discovery_report(
    result: TestDiscoveryResult,
) -> dict:
    """Build a serializable discovery report."""
    return {
        "test_files": result.files,
        "test_directories": result.directories,
        "test_count": result.count,
    }


def build_test_selection_report(
    result: TestSelectionResult,
) -> dict:
    """Build a serializable test-selection report."""
    return {
        "selected_tests": result.selected,
        "test_count": result.count,
        "fallback": result.fallback,
        "reasons": [
            {
                "test_file": reason.test_file,
                "reason": reason.reason,
                "score": reason.score,
            }
            for reason in result.reasons
        ],
    }


def build_test_execution_report(
    result: TestExecutionResult,
) -> dict:
    """Build a serializable test-execution report."""
    return {
        "command": result.command,
        "return_code": result.return_code,
        "output": result.output,
        "passed": result.passed,
        "failed": result.failed,
        "timed_out": result.timed_out,
        "duration_seconds": result.duration_seconds,
    }


def build_failure_analysis_report(
    result: FailureAnalysisResult,
) -> dict:
    """Build a serializable failure-analysis report."""
    return {
        "failed": result.failed,
        "passed": result.passed,
        "failure_count": result.failure_count,
        "collection_error": result.collection_error,
        "timed_out": result.timed_out,
        "failures": [
            {
                "test_file": failure.test_file,
                "test_name": failure.test_name,
                "qualified_name": failure.qualified_name,
                "error_type": failure.error_type,
                "message": failure.message,
                "traceback": failure.traceback,
            }
            for failure in result.failures
        ],
    }