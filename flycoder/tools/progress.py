"""Autonomous progress tracking for FLY-CODER Phase 10.3."""

from __future__ import annotations

from dataclasses import dataclass, field

from flycoder.tools.iteration import IterationObservation


@dataclass
class ProgressSnapshot:
    """Describe progress accumulated through autonomous iterations."""

    iteration: int
    progress: bool
    progress_count: int
    no_progress_count: int
    state_signature: str
    action: str


@dataclass
class ProgressTracker:
    """Track whether autonomous execution is making progress."""

    observations: list[IterationObservation] = field(default_factory=list)
    progress_count: int = 0
    no_progress_count: int = 0
    last_signature: str | None = None

    def record(
        self,
        observation: IterationObservation,
    ) -> ProgressSnapshot:
        """Record one iteration and update progress statistics."""

        self.observations.append(observation)

        if observation.progress:
            self.progress_count += 1
            self.no_progress_count = 0
        else:
            self.no_progress_count += 1

        self.last_signature = observation.state_signature

        return ProgressSnapshot(
            iteration=observation.iteration,
            progress=observation.progress,
            progress_count=self.progress_count,
            no_progress_count=self.no_progress_count,
            state_signature=observation.state_signature,
            action=observation.action,
        )

    @property
    def iterations(self) -> int:
        """Return the number of recorded iterations."""

        return len(self.observations)

    @property
    def has_progress(self) -> bool:
        """Return whether any recorded iteration made progress."""

        return self.progress_count > 0

    @property
    def stalled(self) -> bool:
        """Return whether the latest iteration made no progress."""

        return self.no_progress_count > 0

    def reset(self) -> None:
        """Reset accumulated progress tracking."""

        self.observations.clear()
        self.progress_count = 0
        self.no_progress_count = 0
        self.last_signature = None
