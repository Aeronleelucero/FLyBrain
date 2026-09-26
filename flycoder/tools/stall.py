"""Autonomous no-progress detection for FLY-CODER Phase 10.3."""

from __future__ import annotations

from dataclasses import dataclass

from flycoder.tools.progress import ProgressTracker


STALL_OK = "OK"
STALL_STALLED = "STALLED"
STALL_CRITICAL = "CRITICAL"


@dataclass
class StallAssessment:
    """Describe the current no-progress condition."""

    status: str
    no_progress_count: int
    threshold: int
    message: str

    @property
    def stalled(self) -> bool:
        """Return whether the task is considered stalled."""

        return self.status in {
            STALL_STALLED,
            STALL_CRITICAL,
        }

    @property
    def critical(self) -> bool:
        """Return whether the task is critically stalled."""

        return self.status == STALL_CRITICAL


def assess_stall(
    tracker: ProgressTracker,
    *,
    stalled_threshold: int = 2,
    critical_threshold: int = 3,
) -> StallAssessment:
    """Assess whether autonomous execution has stopped making progress."""

    if stalled_threshold < 1:
        raise ValueError("stalled_threshold must be at least 1")

    if critical_threshold <= stalled_threshold:
        raise ValueError(
            "critical_threshold must be greater than stalled_threshold"
        )

    no_progress_count = tracker.no_progress_count

    if no_progress_count >= critical_threshold:
        return StallAssessment(
            status=STALL_CRITICAL,
            no_progress_count=no_progress_count,
            threshold=critical_threshold,
            message=(
                "Autonomous execution has reached the critical "
                "no-progress threshold."
            ),
        )

    if no_progress_count >= stalled_threshold:
        return StallAssessment(
            status=STALL_STALLED,
            no_progress_count=no_progress_count,
            threshold=stalled_threshold,
            message=(
                "Autonomous execution appears stalled because "
                "no progress was detected across consecutive iterations."
            ),
        )

    return StallAssessment(
        status=STALL_OK,
        no_progress_count=no_progress_count,
        threshold=stalled_threshold,
        message="Autonomous execution is still making progress.",
    )
