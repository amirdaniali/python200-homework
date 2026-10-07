# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 1

"""Aggregate hourly readings into daily summaries."""

from collections import defaultdict
from dataclasses import dataclass

from weatherkit.records import HourlyReading


@dataclass
class DailySummary:
    """Summary of one calendar day.

    Attributes:
        date: Calendar date, "YYYY-MM-DD".
        temp_max: Highest hourly temperature, in degrees Celsius.
        temp_min: Lowest hourly temperature, in degrees Celsius.
        precipitation_sum: Total precipitation, in millimetres.
        hours_observed: Number of hourly readings that contributed.
    """

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the spread between the daily high and low, in degrees Celsius."""
        return self.temp_max - self.temp_min

    # I swap temp_max and temp_min in this function and tested the results with running pytest
    # The tests\test_summarize.py file broke
    # Result: FAILED tests/test_summarize.py::test_max_and_min_for_known_input - AssertionError: assert -23.0 == 23.0


class DailyAggregator:
    """Group hourly readings by calendar date and summarize each day.

    Days with fewer than ``min_hours`` observations are not reported,
    because a max or min from partial data cannot be trusted.
    """

    def __init__(self, min_hours: int = 24) -> None:
        """Create an aggregator.

        Args:
            min_hours: Minimum number of hourly observations a day needs
                before it is reported.
        """
        self.min_hours = min_hours

    @staticmethod
    def _group_by_date(
        readings: list[HourlyReading],
    ) -> dict[str, list[HourlyReading]]:
        """Group readings by the date portion (first 10 chars) of the timestamp."""
        groups: dict[str, list[HourlyReading]] = defaultdict(list)
        for reading in readings:
            groups[reading.timestamp[:10]].append(reading)
        return groups

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Compute daily summaries, dropping days with too few observations.

        Args:
            readings: Hourly readings, in any order.

        Returns:
            DailySummary objects for each complete day, sorted by date.
        """
        summaries: list[DailySummary] = []
        for date, day in self._group_by_date(readings).items():
            if len(day) < self.min_hours:
                continue
            temps = [r.temperature_c for r in day]
            summaries.append(
                DailySummary(
                    date=date,
                    temp_max=max(temps),
                    temp_min=min(temps),
                    precipitation_sum=sum(r.precipitation_mm for r in day),
                    hours_observed=len(day),
                )
            )
        return sorted(summaries, key=lambda s: s.date)

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """List the dates that summarize() would drop.

        Args:
            readings: Hourly readings, in any order.

        Returns:
            Sorted dates with fewer than ``min_hours`` observations.
        """
        return sorted(
            date
            for date, day in self._group_by_date(readings).items()
            if len(day) < self.min_hours
        )
