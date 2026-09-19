"""Learning and memory tools for FLY-CODER."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import re


@dataclass
class Outcome:
    """Structured result of a coding action."""

    status: str
    summary: str
    tests_passed: int = 0
    tests_failed: int = 0
    errors: list[str] = field(default_factory=list)
    verified: bool = False


@dataclass
class Experience:
    """A single recorded coding experience."""

    task: str
    action: str
    outcome: str
    success: bool
    files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    verified: bool = False
    recorded_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class MemoryStore:
    """In-memory collection of coding experiences."""

    experiences: list[Experience] = field(default_factory=list)

    def record_experience(self, experience: Experience) -> Experience:
        """Record an experience and return it."""
        self.experiences.append(experience)
        return experience

    def find_experiences(self, task: str) -> list[Experience]:
        """Find experiences whose task contains the requested text."""
        query = task.strip().lower()

        if not query:
            return []

        return [
            experience
            for experience in self.experiences
            if query in experience.task.lower()
        ]


@dataclass
class SimilarityResult:
    """A memory experience together with its similarity information."""

    experience: Experience
    score: float
    matched_terms: list[str] = field(default_factory=list)


@dataclass
class RankedMemory:
    """A retrieved memory together with ranking information."""

    experience: Experience
    similarity_score: float
    outcome_score: float
    ranking_score: float


@dataclass
class MemoryValidationResult:
    """Validation result for a stored experience."""

    valid: bool
    errors: list[str] = field(default_factory=list)


_STOP_WORDS = {
    "a",
    "an",
    "add",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "change",
    "create",
    "for",
    "fix",
    "from",
    "implement",
    "improve",
    "in",
    "is",
    "it",
    "modify",
    "of",
    "on",
    "or",
    "refactor",
    "remove",
    "the",
    "to",
    "update",
    "with",
}


def _tokenize(text: str) -> list[str]:
    """Normalize text into deterministic unique terms."""
    words = re.findall(r"[a-z0-9]+", text.lower())

    return list(
        dict.fromkeys(
            word
            for word in words
            if word not in _STOP_WORDS
        )
    )


def similarity_score(first: str, second: str) -> tuple[float, list[str]]:
    """Calculate lexical similarity between two pieces of text.

    The score is the Jaccard similarity of the normalized term sets.
    """
    first_terms = set(_tokenize(first))
    second_terms = set(_tokenize(second))

    if not first_terms or not second_terms:
        return 0.0, []

    matched_terms = sorted(first_terms & second_terms)
    union = first_terms | second_terms

    score = len(matched_terms) / len(union)

    return score, matched_terms


def find_similar_experiences(
    store: MemoryStore,
    task: str,
    minimum_score: float = 0.0,
) -> list[SimilarityResult]:
    """Find experiences similar to a task.

    Results are sorted by descending similarity score. Ties preserve
    the original memory insertion order.
    """
    results: list[SimilarityResult] = []

    for experience in store.experiences:
        score, matched_terms = similarity_score(
            task,
            experience.task,
        )

        if score >= minimum_score and score > 0.0:
            results.append(
                SimilarityResult(
                    experience=experience,
                    score=score,
                    matched_terms=matched_terms,
                )
            )

    results.sort(key=lambda result: result.score, reverse=True)

    return results


def calculate_outcome_score(experience: Experience) -> float:
    """Calculate the usefulness score of an experience outcome."""
    if not experience.success:
        return 0.0

    if experience.verified:
        return 1.0

    return 0.5


def calculate_ranking_score(
    similarity: float,
    outcome: float,
) -> float:
    """Combine similarity and outcome quality into a ranking score."""
    return (similarity * 0.7) + (outcome * 0.3)


def rank_experiences(
    store: MemoryStore,
    task: str,
    minimum_score: float = 0.0,
) -> list[RankedMemory]:
    """Find and rank memories for a task."""
    similar = find_similar_experiences(
        store,
        task,
        minimum_score=minimum_score,
    )

    ranked = [
        RankedMemory(
            experience=result.experience,
            similarity_score=result.score,
            outcome_score=calculate_outcome_score(
                result.experience,
            ),
            ranking_score=calculate_ranking_score(
                result.score,
                calculate_outcome_score(result.experience),
            ),
        )
        for result in similar
    ]

    ranked.sort(
        key=lambda result: result.ranking_score,
        reverse=True,
    )

    return ranked


def validate_experience(
    experience: Experience,
) -> MemoryValidationResult:
    """Validate the structural integrity of an experience."""
    errors: list[str] = []

    if not isinstance(experience.task, str) or not experience.task.strip():
        errors.append("task must be a non-empty string")

    if not isinstance(experience.action, str) or not experience.action.strip():
        errors.append("action must be a non-empty string")

    if not isinstance(experience.outcome, str) or not experience.outcome.strip():
        errors.append("outcome must be a non-empty string")

    if not isinstance(experience.success, bool):
        errors.append("success must be a boolean")

    if not isinstance(experience.verified, bool):
        errors.append("verified must be a boolean")

    if not isinstance(experience.files, list):
        errors.append("files must be a list")

    if not isinstance(experience.errors, list):
        errors.append("errors must be a list")

    if not isinstance(experience.notes, list):
        errors.append("notes must be a list")

    if not isinstance(experience.recorded_at, datetime):
        errors.append("recorded_at must be a datetime")

    return MemoryValidationResult(
        valid=not errors,
        errors=errors,
    )


def validate_memory_store(
    store: MemoryStore,
) -> list[MemoryValidationResult]:
    """Validate every experience in a memory store."""
    return [
        validate_experience(experience)
        for experience in store.experiences
    ]


def calculate_decay_score(
    recorded_at: datetime,
    now: datetime | None = None,
    half_life_days: float = 30.0,
) -> float:
    """Calculate freshness using exponential half-life decay.

    A memory at age zero has a score of 1.0. Every half-life period
    reduces the score by half. Future timestamps are treated as current.
    """
    if half_life_days <= 0:
        raise ValueError("half_life_days must be greater than zero")

    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=timezone.utc)

    current_time = now or datetime.now(timezone.utc)

    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    age_seconds = max(
        0.0,
        (current_time - recorded_at).total_seconds(),
    )
    age_days = age_seconds / 86400.0

    decay = math.pow(
        0.5,
        age_days / half_life_days,
    )

    return decay


def record_solution(
    store: MemoryStore,
    task: str,
    action: str,
    outcome: str,
    files: list[str] | None = None,
    notes: list[str] | None = None,
    verified: bool = False,
) -> Experience:
    """Record a successful solution."""
    experience = Experience(
        task=task,
        action=action,
        outcome=outcome,
        success=True,
        files=list(files or []),
        notes=list(notes or []),
        verified=verified,
    )

    return store.record_experience(experience)


def record_failure(
    store: MemoryStore,
    task: str,
    action: str,
    outcome: str,
    error: str | None = None,
    files: list[str] | None = None,
    notes: list[str] | None = None,
) -> Experience:
    """Record a failed solution attempt."""
    errors = [error] if error else []

    experience = Experience(
        task=task,
        action=action,
        outcome=outcome,
        success=False,
        files=list(files or []),
        errors=errors,
        notes=list(notes or []),
        verified=False,
    )

    return store.record_experience(experience)


def record_outcome(
    store: MemoryStore,
    task: str,
    action: str,
    outcome: Outcome,
    files: list[str] | None = None,
    notes: list[str] | None = None,
) -> Experience:
    """Record a structured outcome as an experience."""
    success = (
        outcome.status.lower() == "success"
        and outcome.verified
        and outcome.tests_failed == 0
    )

    experience = Experience(
        task=task,
        action=action,
        outcome=outcome.summary,
        success=success,
        files=list(files or []),
        errors=list(outcome.errors),
        notes=list(notes or []),
        verified=outcome.verified,
    )

    return store.record_experience(experience)


def build_memory_report(store: MemoryStore) -> dict:
    """Build a deterministic summary of stored experiences."""
    successful = sum(
        1 for experience in store.experiences if experience.success
    )
    failed = len(store.experiences) - successful

    return {
        "experience_count": len(store.experiences),
        "successful_count": successful,
        "failed_count": failed,
        "experiences": [
            {
                "task": experience.task,
                "action": experience.action,
                "outcome": experience.outcome,
                "success": experience.success,
                "files": list(experience.files),
                "errors": list(experience.errors),
                "notes": list(experience.notes),
            }
            for experience in store.experiences
        ],
    }