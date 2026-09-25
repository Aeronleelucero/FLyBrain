"""Learning validation tools for FLY-CODER Phase 9.7.5."""

from __future__ import annotations

from dataclasses import dataclass, field

from flycoder.tools.learning import (
    LearningContext,
    LearningMemory,
)
from flycoder.tools.memory import (
    validate_experience,
)


@dataclass
class ValidatedLearning:
    """A learning memory together with its validation assessment."""

    memory: LearningMemory
    valid: bool
    trust_level: str
    reasons: list[str] = field(
        default_factory=list
    )
    warnings: list[str] = field(
        default_factory=list
    )


@dataclass
class LearningValidationResult:
    """Validation result for retrieved learning."""

    task: str
    valid_memories: list[ValidatedLearning] = field(
        default_factory=list
    )
    rejected_memories: list[ValidatedLearning] = field(
        default_factory=list
    )
    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def trusted_memories(
        self,
    ) -> list[ValidatedLearning]:
        """Return memories considered sufficiently trustworthy."""

        return [
            memory
            for memory in self.valid_memories
            if memory.trust_level in {
                "HIGH",
                "MEDIUM",
            }
        ]

    @property
    def high_trust_memories(
        self,
    ) -> list[ValidatedLearning]:
        """Return high-trust learning memories."""

        return [
            memory
            for memory in self.valid_memories
            if memory.trust_level == "HIGH"
        ]

    @property
    def low_trust_memories(
        self,
    ) -> list[ValidatedLearning]:
        """Return valid but low-trust memories."""

        return [
            memory
            for memory in self.valid_memories
            if memory.trust_level == "LOW"
        ]

    @property
    def has_reliable_learning(
        self,
    ) -> bool:
        """Return whether reliable learning evidence exists."""

        return bool(
            self.trusted_memories
        )


def _validate_memory(
    memory: LearningMemory,
) -> ValidatedLearning:
    """Validate one retrieved learning memory."""

    experience = memory.experience

    validation = validate_experience(
        experience
    )

    if not validation.valid:
        return ValidatedLearning(
            memory=memory,
            valid=False,
            trust_level="REJECTED",
            reasons=list(
                validation.errors
            ),
            warnings=[
                "Invalid experience was excluded from learning decisions."
            ],
        )

    reasons: list[str] = []
    warnings: list[str] = []

    # --------------------------------------------------------------
    # Failed experiences are useful as warnings but are not positive
    # solution evidence.
    # --------------------------------------------------------------

    if not experience.success:
        reasons.append(
            "The experience records a failed previous attempt."
        )

        warnings.append(
            "This experience should be treated as cautionary evidence."
        )

        return ValidatedLearning(
            memory=memory,
            valid=True,
            trust_level="LOW",
            reasons=reasons,
            warnings=warnings,
        )

    # --------------------------------------------------------------
    # Verified successful experience
    # --------------------------------------------------------------

    if experience.verified:
        reasons.append(
            "The experience records a successful verified outcome."
        )

        if memory.decay_score >= 0.75:
            reasons.append(
                "The experience is still relatively fresh."
            )

            return ValidatedLearning(
                memory=memory,
                valid=True,
                trust_level="HIGH",
                reasons=reasons,
                warnings=warnings,
            )

        if memory.decay_score >= 0.40:
            reasons.append(
                "The experience has moderate freshness."
            )

            warnings.append(
                "The evidence has decayed and should be "
                "considered with some caution."
            )

            return ValidatedLearning(
                memory=memory,
                valid=True,
                trust_level="MEDIUM",
                reasons=reasons,
                warnings=warnings,
            )

        reasons.append(
            "The experience is substantially decayed."
        )

        warnings.append(
            "The old evidence should not be treated as strong "
            "current guidance."
        )

        return ValidatedLearning(
            memory=memory,
            valid=True,
            trust_level="LOW",
            reasons=reasons,
            warnings=warnings,
        )

    # --------------------------------------------------------------
    # Successful but unverified experience
    # --------------------------------------------------------------

    reasons.append(
        "The experience records success without verification."
    )

    if memory.decay_score >= 0.75:
        warnings.append(
            "The result was not independently verified."
        )

        return ValidatedLearning(
            memory=memory,
            valid=True,
            trust_level="MEDIUM",
            reasons=reasons,
            warnings=warnings,
        )

    warnings.append(
        "The result was not verified and has also decayed."
    )

    return ValidatedLearning(
        memory=memory,
        valid=True,
        trust_level="LOW",
        reasons=reasons,
        warnings=warnings,
    )


def validate_learning_context(
    context: LearningContext,
) -> LearningValidationResult:
    """
    Validate retrieved learning without modifying the source context.

    Validation is advisory only. It does not execute actions, modify
    files, mutate MemoryStore, grant approval, or change CodingState.
    """

    result = LearningValidationResult(
        task=context.task
    )

    for memory in context.memories:
        validated = _validate_memory(
            memory
        )

        if validated.valid:
            result.valid_memories.append(
                validated
            )
        else:
            result.rejected_memories.append(
                validated
            )

        result.warnings.extend(
            validated.warnings
        )

    return result


def build_validation_guidance(
    result: LearningValidationResult,
) -> list[str]:
    """Build deterministic guidance from learning validation."""

    guidance: list[str] = []

    for memory in result.high_trust_memories:
        experience = memory.memory.experience

        guidance.append(
            "High-trust learning evidence: "
            f"{experience.action}."
        )

    for memory in result.valid_memories:
        if memory.trust_level == "MEDIUM":
            experience = memory.memory.experience

            guidance.append(
                "Moderate-trust learning evidence: "
                f"{experience.action}."
            )

    for memory in result.low_trust_memories:
        experience = memory.memory.experience

        guidance.append(
            "Low-trust learning evidence: "
            f"{experience.action}."
        )

    for memory in result.rejected_memories:
        for reason in memory.reasons:
            guidance.append(
                "Rejected learning evidence: "
                f"{reason}."
            )

    return guidance
