# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 1


"""
Data structures and conversion functions for processed weather data.
"""

"""Plain in-memory records used inside the pipeline (after validation)."""

from dataclasses import dataclass

from weatherkit.schemas import WeatherResponse

# Why a dataclass and not a Pydantic model?
# The boundary is the point where untrusted data (a JSON payload from an API)
# enters our program. WeatherResponse sits ON that boundary, so it must parse,
# coerce, and validate: types, lat/lon ranges, equal list lengths, nulls.
# By the time we build an HourlyReading, WeatherResponse has already validated
# everything. We construct readings ourselves from trusted values, so
# re-validating the objects would be wasted work.
# A dataclass is a lightweight, typed container with free __init__, __repr__,
# and __eq__, which is all we need inside the boundary.


@dataclass
class HourlyReading:
    """One hourly weather observation.

    Attributes:
        timestamp: Local time as an ISO-like string, e.g. "2026-04-08T13:00".
        temperature_c: Air temperature at 2 m, in degrees Celsius.
        precipitation_mm: Precipitation over the hour, in millimetres.
    """

    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert the columnar hourly block into one reading per hour.

    Args:
        response: A validated WeatherResponse whose hourly lists are
            guaranteed to be the same length.

    Returns:
        A list of HourlyReading objects in the same order as the input
        lists: element i combines index i of time, temperature_2m, and
        precipitation.
    """
    hourly = response.hourly
    return [
        HourlyReading(
            timestamp=ts,
            temperature_c=temp,
            precipitation_mm=precip,
        )
        for ts, temp, precip in zip(
            hourly.time, hourly.temperature_2m, hourly.precipitation
        )
    ]
