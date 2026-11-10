"""Test-result freshness for FLY-CODER Phase 10.5.

Test results are only meaningful for the code they ran against. FLY-CODER
tracks that with a code version counter on ``CodingState``:

* ``code_version`` increases on every successful code modification
  performed by an agent action (applied repair, rollback, write_code).
* ``tests_code_version`` records the code version a test run verified.

A recorded test result is current only when both versions match.

Scope: this covers modifications made through FLY-CODER actions. Edits
made outside the agent are not visible here; the verification pipeline's
change check (file content versus proposal) covers the repaired file.

All code-modifying actions must call ``record_code_change`` and all test
executions must call ``record_test_run``, so invalidation semantics live
in exactly one place.
"""

from __future__ import annotations

from flycoder.state import CodingState


def record_code_change(state: CodingState) -> None:
    """Record a successful code modification.

    Call this only after the workspace was actually modified. Failed or
    rejected operations must not call it, so their valid test results
    are preserved.
    """

    state.code_version += 1
    state.tests_run = False
    state.tests_passed = False


def record_test_run(
    state: CodingState,
    *,
    passed: bool,
    output: str | None,
) -> None:
    """Record a completed test run for the current code version."""

    state.tests_run = True
    state.tests_passed = passed
    state.tests_code_version = state.code_version
    state.last_error = None if passed else (output or "Tests failed.")

    # Allow a fresh error-inspection cycle on every test run.
    state.error_inspected = False


def tested_code_version(state: CodingState) -> int:
    """Return the code version recorded test results describe.

    Results recorded without a version predate any agent modification,
    which is version 0.
    """

    if state.tests_code_version is None:
        return 0

    return state.tests_code_version


def tests_current(state: CodingState) -> bool:
    """Return whether recorded test results describe the current code."""

    return state.tests_run and tested_code_version(state) == state.code_version


def tests_stale(state: CodingState) -> bool:
    """Return whether test results exist but describe older code."""

    return state.tests_run and not tests_current(state)


def freshness_violations(state: CodingState) -> list[str]:
    """Return contradictions in the freshness fields (read-only)."""

    violations: list[str] = []

    if state.code_version < 0:
        violations.append("code_version cannot be negative.")

    if state.tests_code_version is not None and (
        state.tests_code_version > state.code_version
    ):
        violations.append(
            "tests_code_version cannot be newer than code_version."
        )

    return violations
