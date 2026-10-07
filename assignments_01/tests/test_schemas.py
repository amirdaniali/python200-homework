# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 1

"""Tests for the boundary models in weatherkit.schemas."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit import WeatherResponse

# A plain relative path like "weather_raw.json" is resolved against the
# *current working directory*, which depends on where pytest was launched
# from (the repo root, assignments_01/, an IDE's choice...). Anchoring on
# __file__ makes the path relative to this test file, so it works anywhere.
DATA_PATH = Path(__file__).parent.parent / "weather_raw.json"


def make_payload(**overrides) -> dict:
    """Build a small valid payload, optionally overriding top-level keys."""
    payload = {
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "America/New_York",
        "elevation": 220.0,
        "hourly": {
            "time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
            "temperature_2m": [10.0, 9.5, 9.0],
            "precipitation": [0.0, 0.1, 0.0],
        },
    }
    payload.update(overrides)
    return payload


def test_real_file_validates():
    raw = json.loads(DATA_PATH.read_text())
    response = WeatherResponse.model_validate(raw)
    assert len(response.hourly.time) == 168


@pytest.mark.parametrize("bad_lat", [200.0, -90.1, 91.0])
def test_out_of_range_latitude_rejected(bad_lat):
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(make_payload(latitude=bad_lat))


def test_mismatched_lengths_rejected():
    payload = make_payload()
    payload["hourly"]["temperature_2m"] = [10.0, 9.5]  # one short
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(payload)


def test_null_temperature_rejected():
    payload = make_payload()
    payload["hourly"]["temperature_2m"] = [10.0, None, 9.0]
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(payload)


def test_out_of_range_longitude_rejected():
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(make_payload(longitude=-181.0))
