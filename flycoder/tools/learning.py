"""Learning integration tools for FLY-CODER."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from flycoder.tools.memory import (
    Experience,
    MemoryStore,
    calculate_decay_score,
    rank_experiences,
    validate_experience,
)


@dataclass
class LearningMemory:
    """A memory prepared for use by the planning system."""

    experience: Experience
    similarity_score: float
    outcome_score: float
    ranking_score: float
    decay_score: float


@dataclass
class LearningContext:
    """Knowledge retrieved from previous coding experiences."""

    task: str
    memories: list[LearningMemory] = field(default_factory=list)

    @property
    def successful_memories(self) -> list[LearningMemory]:
        """Return memories representing successful experiences."""
        return [
            memory
            for memory in self.memories
            if memory.experience.success
        ]

    @property
    def failed_memories(self) -> list[LearningMemory]:
        """Return memories representing failed experiences."""
        return [
            memory
            for memory in self.memories
            if not memory.experience.success
        ]

    @property
    def verified_memories(self) -> list[LearningMemory]:
        """Return memories whose solutions were verified."""
        return [
            memory
            for memory in self.memories
            if memory.experience.verified
        ]


def retrieve_learning_context(
    store: MemoryStore,
    task: str,
    minimum_score: float = 0.0,
    now: datetime | None = None,
    half_life_days: float = 30.0,
) -> LearningContext:
    """Retrieve validated memories relevant to a task.

    Existing memory ranking determines relevance and outcome quality.
    This layer adds validation and freshness without modifying the
    underlying MemoryStore.
    """
    normalized_task = " ".join(task.strip().split())

    if not normalized_task:
        return LearningContext(task="", memories=[])

    ranked = rank_experiences(
        store,
        normalized_task,
        minimum_score=minimum_score,
    )

    memories: list[LearningMemory] = []

    for result in ranked:
        validation = validate_experience(
            result.experience
        )

        if not validation.valid:
            continue

        decay_score = calculate_decay_score(
            result.experience.recorded_at,
            now=now,
            half_life_days=half_life_days,
        )

        memories.append(
            LearningMemory(
                experience=result.experience,
                similarity_score=result.similarity_score,
                outcome_score=result.outcome_score,
                ranking_score=result.ranking_score,
                decay_score=decay_score,
            )
        )

    memories.sort(
        key=lambda memory: (
            memory.ranking_score,
            memory.decay_score,
        ),
        reverse=True,
    )

    return LearningContext(
        task=normalized_task,
        memories=memories,
    )


def build_learning_guidance(
    context: LearningContext,
) -> list[str]:
    """Build deterministic guidance from retrieved experiences."""
    guidance: list[str] = []

    for memory in context.successful_memories:
        experience = memory.experience

        if experience.verified:
            guidance.append(
                "A similar verified solution previously succeeded: "
                f"{experience.action}."
            )
        else:
            guidance.append(
                "A similar previous solution succeeded: "
                f"{experience.action}."
            )

    for memory in context.failed_memories:
        experience = memory.experience

        if experience.errors:
            error_text = "; ".join(experience.errors)
            guidance.append(
                "A similar previous attempt failed with: "
                f"{error_text}."
            )
        else:
            guidance.append(
                "A similar previous attempt failed: "
                f"{experience.action}."
            )

    return guidance
