"""Task progress history for FLY-CODER Phase 10.4.3.

History records immutable snapshots of task-level progress (Phase
10.4.2 metrics) across autonomous iterations. Snapshots are never
rewritten, so a later regression never erases the record of progress
that was reached earlier.

Every history metric (changes, regressions, net progress, peak) is
derived from the recorded snapshots on demand. There are no parallel
mutable counters that could drift out of sync.

Note: ``flycoder.tools.progress.ProgressSnapshot`` is a different,
iteration-level record (did one iteration change state?). The snapshot
here describes task progress through milestones.

History is observational only. It never mutates ``CodingState``,
executes actions, approves repairs, or influences guardrails.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

from flycoder.state import CodingState
from flycoder.tools.iteration import build_state_signature
from flycoder.tools.task_progress import (
    TaskProgressMetrics,
    measure_task_progress,
)


PROGRESS_INCREASED = "INCREASED"
PROGRESS_DECREASED = "DECREASED"
PROGRESS_UNCHANGED = "UNCHANGED"

# Fractions are averages of floats; differences below this are noise.
FRACTION_TOLERANCE = 1e-9


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _milestone_keys(name: str, value: object) -> tuple[str, ...]:
    """Validate and normalize a collection of milestone keys to a tuple."""

    if isinstance(value, (str, bytes)) or not isinstance(
        value,
        (tuple, list, frozenset, set),
    ):
        raise TypeError(f"{name} must be a tuple of milestone keys")

    keys = tuple(value)

    if not all(isinstance(key, str) and key for key in keys):
        raise ValueError(f"{name} must contain non-empty strings")

    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} must not contain duplicates")

    return keys


@dataclass(frozen=True)
class ProgressSnapshot:
    """Immutable record of task progress at one iteration.

    Iteration 0 may be used for a baseline captured before any action.
    """

    iteration: int
    completion_fraction: float
    percent: int
    achieved_milestones: tuple[str, ...]
    applicable_milestones: tuple[str, ...]
    next_milestone: str | None
    state_signature: str

    def __post_init__(self) -> None:
        if not _is_int(self.iteration) or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer")

        fraction = self.completion_fraction

        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction)
            or not 0.0 <= fraction <= 1.0
        ):
            raise ValueError(
                "completion_fraction must be a finite number in [0, 1]"
            )

        if not _is_int(self.percent) or not 0 <= self.percent <= 100:
            raise ValueError("percent must be an integer in [0, 100]")

        achieved = _milestone_keys(
            "achieved_milestones",
            self.achieved_milestones,
        )
        applicable = _milestone_keys(
            "applicable_milestones",
            self.applicable_milestones,
        )

        # Normalize to tuples so recorded snapshots cannot be mutated
        # through a list the caller still holds.
        object.__setattr__(self, "completion_fraction", float(fraction))
        object.__setattr__(self, "achieved_milestones", achieved)
        object.__setattr__(self, "applicable_milestones", applicable)

        if not set(achieved) <= set(applicable):
            raise ValueError(
                "achieved_milestones must be a subset of "
                "applicable_milestones"
            )

        if self.next_milestone is not None:
            if self.next_milestone not in applicable:
                raise ValueError("next_milestone must be applicable")

            if self.next_milestone in achieved:
                raise ValueError("next_milestone must not be achieved")

        all_achieved = bool(applicable) and len(achieved) == len(applicable)

        # Same rule as TaskProgressMetrics.percent: 100 is reserved for
        # every applicable milestone being achieved.
        if (self.percent == 100) != all_achieved:
            raise ValueError(
                "percent is 100 if and only if every applicable "
                "milestone is achieved"
            )

        if not isinstance(self.state_signature, str):
            raise TypeError("state_signature must be a string")

    @classmethod
    def from_metrics(
        cls,
        iteration: int,
        metrics: TaskProgressMetrics,
        state_signature: str,
    ) -> ProgressSnapshot:
        """Build a snapshot from already measured task progress."""

        if not isinstance(metrics, TaskProgressMetrics):
            raise TypeError("metrics must be TaskProgressMetrics")

        next_milestone = metrics.next_milestone

        return cls(
            iteration=iteration,
            completion_fraction=metrics.completion_fraction,
            percent=metrics.percent,
            achieved_milestones=tuple(
                m.key for m in metrics.milestones if m.achieved
            ),
            applicable_milestones=tuple(m.key for m in metrics.applicable),
            next_milestone=(
                next_milestone.key if next_milestone is not None else None
            ),
            state_signature=state_signature,
        )

    @property
    def achieved_count(self) -> int:
        return len(self.achieved_milestones)

    @property
    def applicable_count(self) -> int:
        return len(self.applicable_milestones)

    @property
    def all_milestones_achieved(self) -> bool:
        """Progress statement only; never a verified-completion verdict."""

        return self.percent == 100


def capture_snapshot(
    iteration: int,
    state: CodingState,
    intent: str | None = None,
) -> ProgressSnapshot:
    """Measure current task progress and capture it (read-only)."""

    return ProgressSnapshot.from_metrics(
        iteration,
        measure_task_progress(state, intent),
        build_state_signature(state),
    )


@dataclass(frozen=True)
class ProgressChange:
    """Difference in task progress between two consecutive snapshots."""

    before: ProgressSnapshot
    after: ProgressSnapshot

    @property
    def delta(self) -> float:
        return self.after.completion_fraction - self.before.completion_fraction

    @property
    def direction(self) -> str:
        if self.delta > FRACTION_TOLERANCE:
            return PROGRESS_INCREASED

        if self.delta < -FRACTION_TOLERANCE:
            return PROGRESS_DECREASED

        return PROGRESS_UNCHANGED

    @property
    def increased(self) -> bool:
        return self.direction == PROGRESS_INCREASED

    @property
    def decreased(self) -> bool:
        return self.direction == PROGRESS_DECREASED

    @property
    def gained_milestones(self) -> tuple[str, ...]:
        """Milestones achieved after the change but not before."""

        before = set(self.before.achieved_milestones)

        return tuple(
            key for key in self.after.achieved_milestones if key not in before
        )

    @property
    def lost_milestones(self) -> tuple[str, ...]:
        """Milestones achieved before the change but not after."""

        after = set(self.after.achieved_milestones)

        return tuple(
            key for key in self.before.achieved_milestones if key not in after
        )


class ProgressHistory:
    """Chronological, append-only record of task progress snapshots."""

    def __init__(self) -> None:
        self._snapshots: list[ProgressSnapshot] = []

    def record(self, snapshot: ProgressSnapshot) -> ProgressSnapshot:
        """Append a snapshot after validating chronological order.

        Raises:
            TypeError: if ``snapshot`` is not a ProgressSnapshot.
            ValueError: if its iteration does not follow the latest one.
        """

        if not isinstance(snapshot, ProgressSnapshot):
            raise TypeError("snapshot must be a ProgressSnapshot")

        latest = self.latest

        if latest is not None and snapshot.iteration <= latest.iteration:
            raise ValueError(
                "Snapshots must be recorded in increasing iteration "
                f"order: {snapshot.iteration} after {latest.iteration}"
            )

        self._snapshots.append(snapshot)

        return snapshot

    def capture(
        self,
        iteration: int,
        state: CodingState,
        intent: str | None = None,
    ) -> ProgressSnapshot:
        """Capture current task progress from ``state`` and record it."""

        return self.record(capture_snapshot(iteration, state, intent))

    # --------------------------------------------------------------
    # Read access
    # --------------------------------------------------------------

    @property
    def snapshots(self) -> tuple[ProgressSnapshot, ...]:
        """Return all snapshots in chronological order."""

        return tuple(self._snapshots)

    def __len__(self) -> int:
        return len(self._snapshots)

    def __iter__(self) -> Iterator[ProgressSnapshot]:
        return iter(self.snapshots)

    @property
    def is_empty(self) -> bool:
        return not self._snapshots

    @property
    def first(self) -> ProgressSnapshot | None:
        return self._snapshots[0] if self._snapshots else None

    @property
    def latest(self) -> ProgressSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    # --------------------------------------------------------------
    # Derived metrics
    # --------------------------------------------------------------

    @property
    def changes(self) -> tuple[ProgressChange, ...]:
        """Return the change between each pair of consecutive snapshots."""

        snapshots = self._snapshots

        return tuple(
            ProgressChange(before=before, after=after)
            for before, after in zip(snapshots, snapshots[1:])
        )

    @property
    def increases(self) -> tuple[ProgressChange, ...]:
        return tuple(change for change in self.changes if change.increased)

    @property
    def regressions(self) -> tuple[ProgressChange, ...]:
        return tuple(change for change in self.changes if change.decreased)

    @property
    def increase_count(self) -> int:
        return len(self.increases)

    @property
    def regression_count(self) -> int:
        """Return how many consecutive changes decreased progress."""

        return len(self.regressions)

    @property
    def net_progress(self) -> float:
        """Return latest minus first completion fraction (0.0 if < 2)."""

        if len(self._snapshots) < 2:
            return 0.0

        return (
            self._snapshots[-1].completion_fraction
            - self._snapshots[0].completion_fraction
        )

    @property
    def peak(self) -> ProgressSnapshot | None:
        """Return the earliest snapshot with the highest progress."""

        if not self._snapshots:
            return None

        return max(
            self._snapshots,
            key=lambda snapshot: snapshot.completion_fraction,
        )

    @property
    def below_peak(self) -> bool:
        """Return whether current progress is below the best recorded."""

        peak = self.peak
        latest = self.latest

        if peak is None or latest is None:
            return False

        return (
            peak.completion_fraction - latest.completion_fraction
            > FRACTION_TOLERANCE
        )
