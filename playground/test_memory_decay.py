from datetime import datetime, timedelta, timezone

import pytest

from flycoder.tools.memory import calculate_decay_score


BASE_TIME = datetime(
    2026,
    1,
    1,
    tzinfo=timezone.utc,
)


def test_current_memory_has_full_decay_score():
    score = calculate_decay_score(
        recorded_at=BASE_TIME,
        now=BASE_TIME,
    )

    assert score == pytest.approx(1.0)


def test_half_life_reduces_score_to_half():
    score = calculate_decay_score(
        recorded_at=BASE_TIME,
        now=BASE_TIME + timedelta(days=30),
        half_life_days=30,
    )

    assert score == pytest.approx(0.5)


def test_two_half_lives_reduce_score_to_quarter():
    score = calculate_decay_score(
        recorded_at=BASE_TIME,
        now=BASE_TIME + timedelta(days=60),
        half_life_days=30,
    )

    assert score == pytest.approx(0.25)


def test_decay_is_monotonically_decreasing():
    recent = calculate_decay_score(
        recorded_at=BASE_TIME,
        now=BASE_TIME + timedelta(days=10),
    )
    older = calculate_decay_score(
        recorded_at=BASE_TIME,
        now=BASE_TIME + timedelta(days=40),
    )

    assert recent > older


def test_future_timestamp_is_not_penalized():
    score = calculate_decay_score(
        recorded_at=BASE_TIME + timedelta(days=10),
        now=BASE_TIME,
    )

    assert score == pytest.approx(1.0)


def test_custom_half_life_is_supported():
    score = calculate_decay_score(
        recorded_at=BASE_TIME,
        now=BASE_TIME + timedelta(days=10),
        half_life_days=10,
    )

    assert score == pytest.approx(0.5)


def test_zero_half_life_is_rejected():
    with pytest.raises(ValueError, match="half_life_days"):
        calculate_decay_score(
            recorded_at=BASE_TIME,
            now=BASE_TIME,
            half_life_days=0,
        )


def test_negative_half_life_is_rejected():
    with pytest.raises(ValueError, match="half_life_days"):
        calculate_decay_score(
            recorded_at=BASE_TIME,
            now=BASE_TIME,
            half_life_days=-10,
        )
