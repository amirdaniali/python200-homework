# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 1

"""Tests for weatherkit.summarize."""

import pytest

from weatherkit import DailyAggregator, HourlyReading


def make_day(
    date: str, hours: int, temp_start: float = 0.0, precip: float = 0.0
) -> list[HourlyReading]:
    """Build `hours` readings for one date; temperature rises 1 degree per hour."""
    return [
        HourlyReading(f"{date}T{h:02d}:00", temp_start + h, precip)
        for h in range(hours)
    ]


@pytest.fixture
def two_full_days() -> list[HourlyReading]:
    return make_day("2026-04-08", 24, temp_start=0.0, precip=0.1) + make_day(
        "2026-04-09", 24, temp_start=5.0, precip=0.0
    )


@pytest.fixture
def full_plus_partial() -> list[HourlyReading]:
    """One complete day plus a day with only 6 hours."""
    return make_day("2026-04-08", 24) + make_day("2026-04-09", 6)


def test_grouping_produces_one_summary_per_date(two_full_days):
    summaries = DailyAggregator().summarize(two_full_days)
    assert [s.date for s in summaries] == ["2026-04-08", "2026-04-09"]


def test_output_sorted_by_date(two_full_days):
    shuffled = list(reversed(two_full_days))
    summaries = DailyAggregator().summarize(shuffled)
    assert [s.date for s in summaries] == ["2026-04-08", "2026-04-09"]


def test_max_and_min_for_known_input(two_full_days):
    first, second = DailyAggregator().summarize(two_full_days)
    assert (first.temp_max, first.temp_min) == (23.0, 0.0)
    assert (second.temp_max, second.temp_min) == (28.0, 5.0)
    assert first.temp_range() == 23.0
    assert first.hours_observed == 24


@pytest.mark.parametrize(
    "temps, expected_max, expected_min",
    [
        ([5.0], 5.0, 5.0),
        ([1.0, 3.0, 2.0], 3.0, 1.0),
        ([-4.0, -1.0, -9.5], -1.0, -9.5),
    ],
)
def test_max_min_parametrized(temps, expected_max, expected_min):
    readings = [
        HourlyReading(f"2026-04-08T{h:02d}:00", t, 0.0) for h, t in enumerate(temps)
    ]
    (summary,) = DailyAggregator(min_hours=1).summarize(readings)
    assert summary.temp_max == expected_max
    assert summary.temp_min == expected_min


def test_precipitation_sum(two_full_days):
    first, second = DailyAggregator().summarize(two_full_days)
    assert first.precipitation_sum == pytest.approx(2.4)  # 24 * 0.1
    assert second.precipitation_sum == pytest.approx(0.0)


def test_partial_day_dropped_and_reported(full_plus_partial):
    agg = DailyAggregator()
    summaries = agg.summarize(full_plus_partial)
    assert [s.date for s in summaries] == ["2026-04-08"]
    assert agg.incomplete_days(full_plus_partial) == ["2026-04-09"]


def test_lowering_min_hours_keeps_partial_day(full_plus_partial):
    agg = DailyAggregator(min_hours=6)
    summaries = agg.summarize(full_plus_partial)
    assert [s.date for s in summaries] == ["2026-04-08", "2026-04-09"]
    assert summaries[1].hours_observed == 6
    assert agg.incomplete_days(full_plus_partial) == []
