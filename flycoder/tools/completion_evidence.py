"""Completion evidence for FLY-CODER Phase 10.4.4.

Progress is not proof of completion. This module decides whether the
*evidence* available for a task is sufficient to call it verified.

Evidence sources
----------------
1. ``CodingState`` test results written by ``run_tests_action`` or
   ``verify_changes_action`` (``tests_run``, ``tests_passed``,
   ``last_error``). Since Phase 10.5 each run records the code version
   it verified (``tests_code_version``); results for an older version
   are stale (see ``flycoder.tools.freshness``).
2. Phase 7 ``VerificationPipelineResult`` objects wrapped in a
   ``VerificationRecord``. A pipeline result carries no link to the code
   it verified, so the record stamps it with ``build_code_signature``
   at capture time. A signature mismatch at evaluation time makes the
   record stale, and stale evidence never counts. The latest record
   produced by ``verify_changes_action`` is kept on
   ``state.verification_record`` and used when none is passed in.

Safety decisions
----------------
* ``state.finished`` is never read. Neither is task progress, except to
  check the observational criteria of read-only tasks.
* Missing evidence is reported as MISSING, never as success.
* Any fresh failing evidence makes the result FAILED, even if another
  source passed. A failure stays failed until new evidence replaces it.
* A pending repair proposal that has not been explicitly approved and
  applied makes the result NEEDS_APPROVAL. Evidence never approves.
* Contradictory state (checked with the guardrails' read-only
  ``check_state_consistency``) makes the result INCONSISTENT.

Evaluation is deterministic and read-only. It never mutates state, runs
tests, executes actions, modifies files, approves repairs, or changes
guardrail decisions, and it never declares the autonomous loop done.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

from flycoder.state import CodingState
from flycoder.tools.freshness import (
    freshness_violations,
    tested_code_version,
    tests_current,
    tests_stale,
)
from flycoder.tools.guardrails import check_state_consistency
from flycoder.tools.task_progress import (
    INTENT_MILESTONES,
    build_progress_model,
)
from flycoder.tools.verification_pipeline import VerificationPipelineResult


EVIDENCE_VERIFIED = "VERIFIED"
EVIDENCE_FAILED = "FAILED"
EVIDENCE_MISSING = "MISSING"
EVIDENCE_STALE = "STALE"
EVIDENCE_NEEDS_APPROVAL = "NEEDS_APPROVAL"
EVIDENCE_INCOMPLETE = "INCOMPLETE"
EVIDENCE_INCONSISTENT = "INCONSISTENT"

EVIDENCE_STATUSES = {
    EVIDENCE_VERIFIED,
    EVIDENCE_FAILED,
    EVIDENCE_MISSING,
    EVIDENCE_STALE,
    EVIDENCE_NEEDS_APPROVAL,
    EVIDENCE_INCOMPLETE,
    EVIDENCE_INCONSISTENT,
}

METHOD_STATE_TESTS = "state_tests"
METHOD_VERIFICATION_PIPELINE = "verification_pipeline"
METHOD_TASK_OBSERVATION = "task_observation"

VERIFICATION_METHODS = {
    METHOD_STATE_TESTS,
    METHOD_VERIFICATION_PIPELINE,
    METHOD_TASK_OBSERVATION,
}

# Read-only intents never modify code. Their completion criteria are
# observable facts (files discovered/analyzed), not test results.
READ_ONLY_INTENTS = frozenset({"inspect", "explain", "review", "improve"})

# Intents whose completion is proven by passing tests on current code.
TEST_VERIFIED_INTENTS = frozenset({"test", "repair"})

_ERROR_SUMMARY_LIMIT = 300


# ==================================================================
# CODE SIGNATURE AND VERIFICATION RECORDS
# ==================================================================


def build_code_signature(state: CodingState) -> str:
    """Return a signature of the code the agent currently holds.

    The signature covers the selected file and its known content.
    Applying or rolling back a repair, or writing code, changes the
    content and therefore the signature.

    This is conservative: switching ``current_file`` also changes the
    signature and makes earlier pipeline records stale. That can only
    cause re-verification, never a false pass. Edits made outside the
    agent that it has not re-read are not visible in state.
    """

    payload = repr((state.current_file, state.current_file_content))

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VerificationRecord:
    """A Phase 7 pipeline result bound to the code it verified."""

    result: VerificationPipelineResult
    code_signature: str

    def __post_init__(self) -> None:
        if not isinstance(self.result, VerificationPipelineResult):
            raise TypeError("result must be a VerificationPipelineResult")

        if not isinstance(self.code_signature, str) or not self.code_signature:
            raise ValueError("code_signature must be a non-empty string")


def record_verification(
    result: VerificationPipelineResult,
    state: CodingState,
) -> VerificationRecord:
    """Bind a pipeline result to the current code signature.

    Call this immediately after the pipeline runs. The result is copied
    so later changes to the caller's object cannot alter the evidence.
    """

    if not isinstance(result, VerificationPipelineResult):
        raise TypeError("result must be a VerificationPipelineResult")

    return VerificationRecord(
        result=copy.deepcopy(result),
        code_signature=build_code_signature(state),
    )


def verification_is_current(state: CodingState) -> bool:
    """Return whether the state's verification record is fresh.

    A record is current when it was produced for the code signature the
    state holds now. This only says the record applies to the current
    code, not that it passed.
    """

    record = state.verification_record

    return (
        isinstance(record, VerificationRecord)
        and record.code_signature == build_code_signature(state)
    )


# ==================================================================
# COMPLETION EVIDENCE
# ==================================================================


@dataclass(frozen=True)
class CompletionEvidence:
    """Immutable, explainable verdict on a task's verification evidence.

    ``verified`` is derived from ``status`` and cannot disagree with it.
    Construction rejects a VERIFIED status that contradicts the other
    fields, so a verified verdict cannot be hand-built from bad inputs.
    """

    intent: str
    status: str
    verification_method: str | None
    code_signature: str
    verification_signature: str | None
    tests_run: bool
    tests_passed: bool
    unresolved_errors: tuple[str, ...]
    human_approval_required: bool
    human_approval_granted: bool
    criteria_satisfied: bool
    reasons: tuple[str, ...]
    verification: VerificationRecord | None = None

    def __post_init__(self) -> None:
        if self.intent not in INTENT_MILESTONES:
            raise ValueError(f"Unknown task intent: {self.intent!r}")

        if self.status not in EVIDENCE_STATUSES:
            raise ValueError(f"Unknown evidence status: {self.status!r}")

        if (
            self.verification_method is not None
            and self.verification_method not in VERIFICATION_METHODS
        ):
            raise ValueError(
                f"Unknown verification method: {self.verification_method!r}"
            )

        for name in ("unresolved_errors", "reasons"):
            value = getattr(self, name)

            if isinstance(value, str) or not isinstance(value, (tuple, list)):
                raise TypeError(f"{name} must be a tuple of strings")

            object.__setattr__(self, name, tuple(value))

        if self.tests_passed and not self.tests_run:
            raise ValueError("tests_passed requires tests_run")

        if self.human_approval_granted and not self.human_approval_required:
            raise ValueError(
                "human_approval_granted requires human_approval_required"
            )

        if self.status == EVIDENCE_VERIFIED:
            problems = []

            if not self.criteria_satisfied:
                problems.append("criteria are not satisfied")

            if self.verification_method is None:
                problems.append("no verification method")

            if self.unresolved_errors:
                problems.append("unresolved errors remain")

            if (
                self.human_approval_required
                and not self.human_approval_granted
            ):
                problems.append("required human approval is missing")

            if self.intent in TEST_VERIFIED_INTENTS and not self.tests_passed:
                problems.append("tests have not passed")

            if problems:
                raise ValueError(
                    "VERIFIED evidence is contradictory: "
                    + "; ".join(problems)
                )

    @property
    def verified(self) -> bool:
        """Return whether the evidence verifies the task."""

        return self.status == EVIDENCE_VERIFIED

    @property
    def failed(self) -> bool:
        """Return whether fresh evidence shows verification failed."""

        return self.status == EVIDENCE_FAILED

    @property
    def missing(self) -> bool:
        """Return whether no usable evidence exists yet."""

        return self.status in {EVIDENCE_MISSING, EVIDENCE_STALE}

    @property
    def stale(self) -> bool:
        return self.status == EVIDENCE_STALE

    @property
    def needs_approval(self) -> bool:
        return self.status == EVIDENCE_NEEDS_APPROVAL


def _summarize_error(text: str) -> str:
    """Return the last meaningful line of an error, truncated."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = lines[-1] if lines else text.strip()

    if len(summary) > _ERROR_SUMMARY_LIMIT:
        summary = summary[: _ERROR_SUMMARY_LIMIT - 3] + "..."

    return summary


def _pipeline_failures(result: VerificationPipelineResult) -> list[str]:
    """Describe a failed pipeline result using its own analysis."""

    failures = [f"Verification pipeline failed: {result.reason}"]

    analysis = result.targeted_analysis

    if analysis is not None:
        failures.extend(
            f"Failing test: {failure.qualified_name}"
            for failure in analysis.failures
            if failure.qualified_name
        )

    return failures


def _record_inconsistencies(
    record: VerificationRecord,
    state: CodingState,
) -> list[str]:
    """Return contradictions between a fresh pipeline record and state."""

    change = record.result.change
    problems: list[str] = []

    if state.repair_applied:
        if change.file is not None and change.file != state.current_file:
            problems.append(
                f"Verification record targets {change.file!r} but the "
                f"applied repair targets {state.current_file!r}."
            )

        if (
            change.expected_content is not None
            and change.expected_content != state.proposed_content
        ):
            problems.append(
                "Verification record expects content that differs from "
                "the applied repair."
            )

    # The pipeline only runs regression after targeted tests run.
    if record.result.targeted is None and record.result.regression is not None:
        problems.append(
            "Verification record contains a regression result without "
            "targeted tests."
        )

    return problems


def evaluate_completion_evidence(
    state: CodingState,
    verification: VerificationRecord | None = None,
    intent: str | None = None,
) -> CompletionEvidence:
    """Evaluate whether available evidence verifies the task.

    Args:
        state: Current task state. It is read, never modified.
        verification: Optional pipeline result bound to a code signature
            by ``record_verification``. Defaults to
            ``state.verification_record``.
        intent: Task intent; defaults to ``state.task_intent``.

    Raises:
        ValueError: if the intent is missing or unknown.
        TypeError: if ``verification`` is not a VerificationRecord.
    """

    resolved_intent = build_progress_model(intent or state.task_intent).intent

    if verification is None:
        verification = state.verification_record

    if verification is not None and not isinstance(
        verification,
        VerificationRecord,
    ):
        raise TypeError("verification must be a VerificationRecord")

    code_signature = build_code_signature(state)
    reasons: list[str] = []

    # --------------------------------------------------------------
    # Human approval: a proposal only takes effect once a human has
    # explicitly approved it and it has actually been applied.
    # --------------------------------------------------------------

    approval_required = state.repair_proposed
    approval_granted = (
        approval_required and state.repair_approved and state.repair_applied
    )

    # --------------------------------------------------------------
    # Pipeline record freshness
    # --------------------------------------------------------------

    fresh_record: VerificationRecord | None = None

    if verification is not None:
        if verification.code_signature == code_signature:
            fresh_record = verification
        else:
            reasons.append(
                "Verification record is stale: the code changed after it "
                "was produced, so it is ignored."
            )

    def finish(
        status: str,
        *,
        method: str | None = None,
        tests_run: bool = False,
        tests_passed: bool = False,
        errors: list[str] | None = None,
        criteria: bool = False,
    ) -> CompletionEvidence:
        return CompletionEvidence(
            intent=resolved_intent,
            status=status,
            verification_method=method,
            code_signature=code_signature,
            verification_signature=(
                verification.code_signature if verification else None
            ),
            tests_run=tests_run,
            tests_passed=tests_passed,
            unresolved_errors=tuple(errors or ()),
            human_approval_required=approval_required,
            human_approval_granted=approval_granted,
            criteria_satisfied=criteria,
            reasons=tuple(reasons),
            verification=verification,
        )

    # --------------------------------------------------------------
    # 1. Contradictory state or evidence cannot verify anything.
    # --------------------------------------------------------------

    violations, _ = check_state_consistency(state)
    violations = violations + freshness_violations(state)

    if fresh_record is not None:
        violations = violations + _record_inconsistencies(fresh_record, state)

    if violations:
        reasons.extend(f"Inconsistent evidence: {v}" for v in violations)

        return finish(EVIDENCE_INCONSISTENT)

    # --------------------------------------------------------------
    # 2. Pending approval blocks verification for every intent.
    # --------------------------------------------------------------

    if approval_required and not approval_granted:
        reasons.append(
            "A repair proposal is waiting for explicit human approval; "
            "its effect cannot be verified until it is approved and "
            "applied."
        )

        return finish(
            EVIDENCE_NEEDS_APPROVAL,
            tests_run=tests_current(state),
            tests_passed=tests_current(state) and state.tests_passed,
        )

    if approval_granted:
        reasons.append("The repair was explicitly approved and applied.")

    # --------------------------------------------------------------
    # 3. Read-only intents: observational completion criteria.
    # --------------------------------------------------------------

    if resolved_intent in READ_ONLY_INTENTS:
        if verification is not None:
            reasons.append(
                "Verification records are not required for read-only "
                f"'{resolved_intent}' tasks."
            )

        if state.last_error:
            reasons.append(
                "A recorded error exists but does not block a read-only "
                "task."
            )

        model = build_progress_model(resolved_intent)
        unmet = [m for m in model.evaluate(state) if not m.achieved]

        if unmet:
            reasons.extend(f"Criterion not met: {m.detail}" for m in unmet)

            return finish(EVIDENCE_INCOMPLETE)

        reasons.append(
            f"All read-only '{resolved_intent}' criteria were observed."
        )

        return finish(
            EVIDENCE_VERIFIED,
            method=METHOD_TASK_OBSERVATION,
            criteria=True,
        )

    # --------------------------------------------------------------
    # 4. Test-verified intents: passing tests on current code.
    # --------------------------------------------------------------

    # Only test results recorded for the current code version count.
    state_tests_run = tests_current(state)
    state_tests_passed = state_tests_run and state.tests_passed
    stale_tests = tests_stale(state)

    if stale_tests:
        reasons.append(
            "Recorded test results are stale: they verified code version "
            f"{tested_code_version(state)}, but the current code version "
            f"is {state.code_version}. They are ignored."
        )

    errors: list[str] = []

    if state_tests_run and not state_tests_passed:
        errors.append("Test run on the current code failed.")

    if state.last_error:
        errors.append(_summarize_error(state.last_error))

    if fresh_record is not None and not fresh_record.result.verified:
        errors.extend(_pipeline_failures(fresh_record.result))

    if fresh_record is not None:
        record_tests_run = fresh_record.result.targeted is not None
        record_passed = fresh_record.result.verified
    else:
        record_tests_run = False
        record_passed = False

    tests_run = state_tests_run or record_tests_run

    if errors:
        reasons.append(
            "Verification failed; it stays failed until new evidence is "
            "produced on the current code."
        )

        return finish(
            EVIDENCE_FAILED,
            tests_run=tests_run,
            errors=errors,
        )

    if record_passed:
        reasons.append(
            "The verification pipeline passed on the current code "
            "(change, targeted tests, and regression suite)."
        )

        return finish(
            EVIDENCE_VERIFIED,
            method=METHOD_VERIFICATION_PIPELINE,
            tests_run=True,
            tests_passed=True,
            criteria=True,
        )

    if state_tests_passed:
        reasons.append("The test suite passed on the current code.")

        return finish(
            EVIDENCE_VERIFIED,
            method=METHOD_STATE_TESTS,
            tests_run=True,
            tests_passed=True,
            criteria=True,
        )

    if verification is not None or stale_tests:
        return finish(EVIDENCE_STALE)

    if state.repair_applied:
        reasons.append(
            "Tests must be re-run after the applied repair; no evidence "
            "exists for the current code."
        )
    else:
        reasons.append("Tests have not been run on the current code.")

    return finish(EVIDENCE_MISSING)
