# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 1

"""Tests for weatherkit.records."""

import pytest

from weatherkit import HourlyReading, WeatherResponse, to_readings


@pytest.fixture
def response() -> WeatherResponse:
    return WeatherResponse.model_validate(
        {
            "latitude": 35.2,
            "longitude": -80.8,
            "timezone": "America/New_York",
            "elevation": 220.0,
            "hourly": {
                "time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
                "temperature_2m": [10.0, 9.5, 9.0],
                "precipitation": [0.0, 0.4, 1.2],
            },
        }
    )


def test_one_reading_per_hour_in_order(response):
    readings = to_readings(response)
    assert len(readings) == 3
    assert readings[0].timestamp == "2026-04-08T00:00"
    assert readings[-1].timestamp == "2026-04-08T02:00"


def test_values_match_input_indices(response):
    readings = to_readings(response)
    hourly = response.hourly
    for i, reading in enumerate(readings):
        assert reading.timestamp == hourly.time[i]
        assert reading.temperature_c == hourly.temperature_2m[i]
        assert reading.precipitation_mm == hourly.precipitation[i]


def test_equal_readings_compare_equal():
    a = HourlyReading("2026-04-08T00:00", 10.0, 0.0)
    b = HourlyReading("2026-04-08T00:00", 10.0, 0.0)
    assert a == b
    assert a != HourlyReading("2026-04-08T01:00", 10.0, 0.0)
