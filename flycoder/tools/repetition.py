"""Repeated-action detection for FLY-CODER Phase 10.3."""

from __future__ import annotations

from dataclasses import dataclass, field


REPETITION_NORMAL = "NORMAL"
REPETITION_REPEATED = "REPEATED"
REPETITION_LOOP_RISK = "LOOP_RISK"


@dataclass
class RepetitionAssessment:
    """Describe repeated autonomous action behavior."""

    status: str
    action: str | None
    consecutive_count: int
    threshold: int
    message: str

    @property
    def repeated(self) -> bool:
        """Return whether the action is repeating."""

        return self.status in {
            REPETITION_REPEATED,
            REPETITION_LOOP_RISK,
        }

    @property
    def loop_risk(self) -> bool:
        """Return whether repetition indicates loop risk."""

        return self.status == REPETITION_LOOP_RISK


@dataclass
class ActionRepetitionTracker:
    """Track consecutive repetition of autonomous actions."""

    actions: list[str] = field(default_factory=list)
    current_action: str | None = None
    consecutive_count: int = 0

    def record(self, action: str) -> RepetitionAssessment:
        """Record an action and assess consecutive repetition."""

        if not action:
            self.actions.append(action)
            self.current_action = None
            self.consecutive_count = 0

            return RepetitionAssessment(
                status=REPETITION_NORMAL,
                action=None,
                consecutive_count=0,
                threshold=3,
                message="No action was recorded.",
            )

        self.actions.append(action)

        if action == self.current_action:
            self.consecutive_count += 1
        else:
            self.current_action = action
            self.consecutive_count = 1

        return self.assess()

    def assess(
        self,
        *,
        repeated_threshold: int = 2,
        loop_risk_threshold: int = 3,
    ) -> RepetitionAssessment:
        """Assess the current consecutive action count."""

        if repeated_threshold < 1:
            raise ValueError(
                "repeated_threshold must be at least 1"
            )

        if loop_risk_threshold <= repeated_threshold:
            raise ValueError(
                "loop_risk_threshold must be greater than "
                "repeated_threshold"
            )

        count = self.consecutive_count
        action = self.current_action

        if count >= loop_risk_threshold:
            return RepetitionAssessment(
                status=REPETITION_LOOP_RISK,
                action=action,
                consecutive_count=count,
                threshold=loop_risk_threshold,
                message=(
                    "The same autonomous action has been repeated "
                    "enough times to indicate loop risk."
                ),
            )

        if count >= repeated_threshold:
            return RepetitionAssessment(
                status=REPETITION_REPEATED,
                action=action,
                consecutive_count=count,
                threshold=repeated_threshold,
                message=(
                    "The same autonomous action has been repeated "
                    "across consecutive iterations."
                ),
            )

        return RepetitionAssessment(
            status=REPETITION_NORMAL,
            action=action,
            consecutive_count=count,
            threshold=repeated_threshold,
            message="No problematic action repetition was detected.",
        )

    def reset(self) -> None:
        """Reset the repetition tracker."""

        self.actions.clear()
        self.current_action = None
        self.consecutive_count = 0
