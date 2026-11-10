"""Immutable orchestration snapshots for FLY-CODER Phase 10.9.

``TaskSession`` is the orchestration controller. This module only
*describes* a session: ``build_snapshot`` reads every authoritative
source once and returns a deeply immutable ``OrchestrationSnapshot``.
Nothing here executes actions, changes state, or decides anything; the
snapshot stores no authority of its own.

Immutability strategy
---------------------
* Every snapshot type is a frozen dataclass.
* Values that are already deeply immutable (strings, numbers, datetimes,
  tuples and frozen dataclasses of those, such as ``CompletionOutcome``,
  ``InterventionRequest``, ``DecisionRecord``, ``RecoveryAttempt``, task
  progress snapshots) are kept as they are; sharing them is safe.
* Everything else is converted by ``freeze``: dicts become
  ``FrozenMapping``, lists and tuples become tuples, sets become
  frozensets, mutable dataclasses become a ``FrozenMapping`` of their
  fields (with a ``"__type__"`` entry), recursively. Anything else
  (objects without a frozen representation) becomes its ``repr``.
* Lifecycles, histories, steps, evidence, and reports are copied into
  dedicated snapshot types; no reference to the session, its state, or
  any mutable internal object is reachable from a snapshot.

The snapshot is a point-in-time view. During a run (``running``) the
outcome, evidence, and stop fields describe the last completed run.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import enum
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from flycoder.tools.autonomous import (
    COMPONENT_ERROR_ACTION,
    AutonomousStep,
    ComponentError,
    _verification_attempts,
    guardrail_blocked,
)
from flycoder.tools.completion_evidence import CompletionEvidence
from flycoder.tools.completion_outcome import (
    CompletionOutcome,
    VerificationAttempt,
)
from flycoder.tools.human_intervention import (
    DecisionRecord,
    InterventionRequest,
)
from flycoder.tools.recovery import RecoveryAttempt
from flycoder.tools.task_lifecycle import TASK_COMPLETED, TaskLifecycle

if TYPE_CHECKING:
    from flycoder.tools.task_session import TaskSession


_MAX_FREEZE_DEPTH = 40

_IMMUTABLE_SCALARS = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    _datetime.date,
    _datetime.time,
    _datetime.timedelta,
    _datetime.timezone,
    enum.Enum,
)


# ==================================================================
# FROZEN VALUES
# ==================================================================


class FrozenMapping(Mapping):
    """A read-only, hashable mapping.

    The items live in a ``MappingProxyType`` over a private dict that
    nothing else references, so the mapping cannot be changed through
    its public interface, and attributes cannot be set.
    """

    __slots__ = ("_proxy",)

    def __init__(self, items: Mapping | tuple = ()) -> None:
        object.__setattr__(self, "_proxy", MappingProxyType(dict(items)))

    def __getitem__(self, key: Any) -> Any:
        return self._proxy[key]

    def __iter__(self):
        return iter(self._proxy)

    def __len__(self) -> int:
        return len(self._proxy)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("FrozenMapping is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("FrozenMapping is immutable")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self._proxy) == dict(other.items())

        return NotImplemented

    def __hash__(self) -> int:
        return hash(frozenset(self._proxy.items()))

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._proxy)!r})"


def is_deeply_immutable(value: Any, _depth: int = 0) -> bool:
    """Return whether nothing reachable from ``value`` can be mutated."""

    if _depth > _MAX_FREEZE_DEPTH:
        return False

    if isinstance(value, _IMMUTABLE_SCALARS):
        return True

    if isinstance(value, (tuple, frozenset)):
        return all(is_deeply_immutable(item, _depth + 1) for item in value)

    if isinstance(value, FrozenMapping):
        return all(
            is_deeply_immutable(key, _depth + 1)
            and is_deeply_immutable(item, _depth + 1)
            for key, item in value.items()
        )

    if (
        dataclasses.is_dataclass(value)
        and not isinstance(value, type)
        and value.__dataclass_params__.frozen
    ):
        return all(
            is_deeply_immutable(getattr(value, field.name), _depth + 1)
            for field in dataclasses.fields(value)
        )

    return False


def freeze(value: Any, _depth: int = 0, _seen: frozenset[int] = frozenset()) -> Any:
    """Return a deeply immutable representation of ``value``.

    Deeply immutable values are returned unchanged. Containers and
    mutable dataclasses are converted recursively; anything without a
    frozen representation (or a cycle) becomes its ``repr``.
    """

    if is_deeply_immutable(value):
        return value

    if _depth > _MAX_FREEZE_DEPTH or id(value) in _seen:
        return repr(value)

    seen = _seen | {id(value)}

    def inner(item: Any) -> Any:
        return freeze(item, _depth + 1, seen)

    if isinstance(value, PurePath):
        return str(value)

    if isinstance(value, Mapping):
        items = {}

        for key, item in value.items():
            frozen_key = inner(key)

            try:
                hash(frozen_key)
            except TypeError:
                frozen_key = repr(key)

            items[frozen_key] = inner(item)

        return FrozenMapping(items)

    if isinstance(value, (list, tuple)):
        return tuple(inner(item) for item in value)

    if isinstance(value, (set, frozenset)):
        frozen_items = [inner(item) for item in value]

        try:
            return frozenset(frozen_items)
        except TypeError:
            return tuple(sorted(frozen_items, key=repr))

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {"__type__": type(value).__name__}

        for field in dataclasses.fields(value):
            fields[field.name] = inner(getattr(value, field.name))

        return FrozenMapping(fields)

    return repr(value)


# ==================================================================
# SNAPSHOT TYPES
# ==================================================================


@dataclass(frozen=True)
class TransitionSnapshot:
    from_state: str
    to_state: str
    reason: str
    timestamp: _datetime.datetime


@dataclass(frozen=True)
class LifecycleSnapshot:
    task: str
    status: str
    is_terminal: bool
    current_iteration: int
    started_at: _datetime.datetime | None
    completed_at: _datetime.datetime | None
    actions: tuple[str, ...]
    errors: tuple[str, ...]
    transitions: tuple[TransitionSnapshot, ...]

    @classmethod
    def capture(cls, lifecycle: TaskLifecycle) -> LifecycleSnapshot:
        return cls(
            task=lifecycle.task,
            status=lifecycle.status,
            is_terminal=lifecycle.is_terminal,
            current_iteration=lifecycle.current_iteration,
            started_at=lifecycle.started_at,
            completed_at=lifecycle.completed_at,
            actions=tuple(lifecycle.actions),
            errors=tuple(lifecycle.errors),
            transitions=tuple(
                TransitionSnapshot(
                    from_state=t.from_state,
                    to_state=t.to_state,
                    reason=t.reason,
                    timestamp=t.timestamp,
                )
                for t in lifecycle.transitions
            ),
        )


@dataclass(frozen=True)
class EvidenceSnapshot:
    intent: str
    status: str
    verified: bool
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
    # The pipeline record the evidence was evaluated with, frozen.
    verification_record: FrozenMapping | None

    @classmethod
    def capture(cls, evidence: CompletionEvidence) -> EvidenceSnapshot:
        return cls(
            intent=evidence.intent,
            status=evidence.status,
            verified=evidence.verified,
            verification_method=evidence.verification_method,
            code_signature=evidence.code_signature,
            verification_signature=evidence.verification_signature,
            tests_run=evidence.tests_run,
            tests_passed=evidence.tests_passed,
            unresolved_errors=tuple(evidence.unresolved_errors),
            human_approval_required=evidence.human_approval_required,
            human_approval_granted=evidence.human_approval_granted,
            criteria_satisfied=evidence.criteria_satisfied,
            reasons=tuple(evidence.reasons),
            verification_record=(
                freeze(evidence.verification)
                if evidence.verification is not None
                else None
            ),
        )


@dataclass(frozen=True)
class StepSnapshot:
    """One executed (or failed) iteration, frozen."""

    iteration: int
    action: str
    success: bool
    message: str
    guardrail_blocked: bool
    data: Any
    evidence: EvidenceSnapshot | None
    component_error: ComponentError | None
    # Frozen observation, recovery, iteration analysis, progress, and
    # confidence recorded for the step.
    details: FrozenMapping

    @property
    def is_component_error(self) -> bool:
        return self.action == COMPONENT_ERROR_ACTION

    @classmethod
    def capture(cls, step: AutonomousStep) -> StepSnapshot:
        result = step.result
        component_error = None

        if result.action == COMPONENT_ERROR_ACTION and isinstance(
            result.data, dict
        ):
            details = result.data.get("component_error") or {}
            component_error = ComponentError(
                type_name=str(details.get("type", "")),
                message=str(details.get("message", "")),
                stage=str(details.get("stage", "")),
                iteration=details.get("iteration"),
                interrupted=bool(details.get("interrupted", False)),
            )

        return cls(
            iteration=step.iteration,
            action=result.action,
            success=bool(result.success),
            message=result.message,
            guardrail_blocked=guardrail_blocked(result),
            data=freeze(result.data),
            evidence=(
                EvidenceSnapshot.capture(step.completion_evidence)
                if step.completion_evidence is not None
                else None
            ),
            component_error=component_error,
            details=FrozenMapping(
                {
                    name: freeze(getattr(step, name))
                    for name in (
                        "observation",
                        "recovery",
                        "iteration_observation",
                        "stall",
                        "iteration_decision",
                        "progress_snapshot",
                        "recovery_assessment",
                        "confidence",
                    )
                }
            ),
        )


@dataclass(frozen=True)
class OrchestrationSnapshot:
    """Deeply immutable, point-in-time view of one ``TaskSession``.

    ``successful`` is only ever ``outcome.successful``, so it cannot
    disagree with the single completion classifier and success gate.
    """

    task_id: str
    task: str
    intent: str | None
    workspace: str | None

    max_iterations: int
    iterations_used: int
    remaining_iterations: int

    started: bool
    running: bool
    closed: bool
    close_reason: str | None

    lifecycle: LifecycleSnapshot | None
    lifecycle_status: str | None
    stop_status: str | None
    stopped_reason: str | None

    outcome: CompletionOutcome | None
    successful: bool
    verified: bool
    evidence: EvidenceSnapshot | None

    progress_history: tuple[Any, ...]
    steps: tuple[StepSnapshot, ...]
    verification_attempts: tuple[VerificationAttempt, ...]
    recovery_attempts: tuple[RecoveryAttempt, ...]

    pending_intervention: InterventionRequest | None
    decisions: tuple[DecisionRecord, ...]

    can_resume: bool
    human_action_required: bool
    next_step: str
    unresolved_issues: tuple[str, ...]
    completed_work: tuple[str, ...]
    remaining_work: tuple[str, ...]

    error: ComponentError | None
    component_errors: tuple[ComponentError, ...]
    diagnostics: tuple[str, ...]
    progress_report: str | None

    def __post_init__(self) -> None:
        if self.successful and not (
            self.outcome is not None
            and self.outcome.successful
            and self.lifecycle_status == TASK_COMPLETED
        ):
            raise ValueError(
                "A snapshot can only be successful for a successful "
                "outcome with a COMPLETED lifecycle."
            )

        if not is_deeply_immutable(self):
            raise TypeError("OrchestrationSnapshot must be deeply immutable")


# ==================================================================
# CONSTRUCTION
# ==================================================================


def _next_step(session: TaskSession, outcome: CompletionOutcome | None) -> str:
    if outcome is not None:
        return outcome.next_step

    if not session.started:
        return "Run the task."

    if session.result is not None and session.result.error is not None:
        return (
            "Review the component error; evidence could not be evaluated, "
            "so no outcome was classified."
        )

    return "Wait for the run to stop."


def build_snapshot(session: TaskSession) -> OrchestrationSnapshot:
    """Capture a deeply immutable snapshot of ``session``.

    Read-only: it never changes the session, its state, or its result.
    """

    result = session.result
    lifecycle = result.lifecycle if result is not None else None
    outcome = result.outcome if result is not None else None
    evidence = result.completion_evidence if result is not None else None
    report = result.progress_report if result is not None else None
    steps = tuple(session.steps)

    workspace = getattr(session.agent, "workspace", None)
    workspace_root = getattr(workspace, "root", None)

    step_snapshots = tuple(StepSnapshot.capture(step) for step in steps)
    component_errors = tuple(
        step.component_error
        for step in step_snapshots
        if step.component_error is not None
    )

    if (
        result is not None
        and result.error is not None
        and result.error not in component_errors
    ):
        component_errors = component_errors + (result.error,)

    recovery_history = result.recovery_history if result is not None else None
    progress_history = result.progress_history if result is not None else None

    return OrchestrationSnapshot(
        task_id=session.task_id,
        task=session.state.task,
        intent=session.state.task_intent,
        workspace=str(workspace_root) if workspace_root is not None else None,
        max_iterations=session.max_iterations,
        iterations_used=session.iterations_used,
        remaining_iterations=session.remaining_iterations,
        started=session.started,
        running=session.running,
        closed=session.closed,
        close_reason=session.close_reason,
        lifecycle=(
            LifecycleSnapshot.capture(lifecycle)
            if lifecycle is not None
            else None
        ),
        lifecycle_status=lifecycle.status if lifecycle is not None else None,
        stop_status=result.status if result is not None else None,
        stopped_reason=result.stopped_reason if result is not None else None,
        outcome=outcome,
        successful=outcome is not None and outcome.successful,
        verified=evidence is not None and evidence.verified,
        evidence=(
            EvidenceSnapshot.capture(evidence)
            if evidence is not None
            else None
        ),
        progress_history=(
            tuple(freeze(snapshot) for snapshot in progress_history)
            if progress_history is not None
            else ()
        ),
        steps=step_snapshots,
        verification_attempts=_verification_attempts(steps),
        recovery_attempts=(
            tuple(recovery_history.attempts)
            if recovery_history is not None
            else ()
        ),
        pending_intervention=freeze(session.intervention_request()),
        decisions=tuple(freeze(record) for record in session.decisions),
        can_resume=outcome is not None and outcome.can_resume,
        human_action_required=(
            outcome.human_action_required
            if outcome is not None
            else session.state.user_input_needed
        ),
        next_step=_next_step(session, outcome),
        unresolved_issues=(
            tuple(outcome.unresolved_issues) if outcome is not None else ()
        ),
        completed_work=(
            tuple(outcome.completed_work) if outcome is not None else ()
        ),
        remaining_work=(
            tuple(outcome.remaining_work) if outcome is not None else ()
        ),
        error=result.error if result is not None else None,
        component_errors=component_errors,
        diagnostics=tuple(session.diagnostics),
        progress_report=report.render() if report is not None else None,
    )
