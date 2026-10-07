# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 1

"""WeatherKit - A package for processing weather API responses."""

from weatherkit.records import HourlyReading, to_readings
from weatherkit.schemas import HourlyBlock, WeatherResponse
from weatherkit.summarize import DailyAggregator, DailySummary

__all__ = [
    "HourlyBlock",
    "WeatherResponse",
    "HourlyReading",
    "to_readings",
    "DailySummary",
    "DailyAggregator",
]
