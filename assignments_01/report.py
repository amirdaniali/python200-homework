# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 1


"""Load weather_raw.json and print a daily weather summary table."""

import json
from pathlib import Path

from weatherkit import DailyAggregator, WeatherResponse, to_readings

DATA_PATH = Path(__file__).parent / "weather_raw.json"


def load_response(path: Path = DATA_PATH) -> WeatherResponse:
    """Read the JSON file and validate it into a WeatherResponse."""
    raw = json.loads(path.read_text())
    return WeatherResponse.model_validate(raw)


def print_table(summaries) -> None:
    """Print one row per day: date, high, low, total precipitation, range."""
    header = (
        f"{'Date':<12}{'High (C)':>10}{'Low (C)':>10}"
        f"{'Precip (mm)':>14}{'Range (C)':>11}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.date:<12}{s.temp_max:>10.1f}{s.temp_min:>10.1f}"
            f"{s.precipitation_sum:>14.1f}{s.temp_range():>11.1f}"
        )


def main() -> None:
    response = load_response()
    readings = to_readings(response)

    aggregator = DailyAggregator()  # default min_hours=24
    summaries = aggregator.summarize(readings)
    dropped = aggregator.incomplete_days(readings)

    print(f"Location: {response.latitude}, {response.longitude} ({response.timezone})")
    print()
    print_table(summaries)
    if dropped:
        print()
        print(
            f"WARNING: dropped incomplete days (< {aggregator.min_hours} hours): "
            f"{', '.join(dropped)}"
        )


# Without this guard, every top-level statement would run at import time. If
# another module did `from report import print_table`, Python would execute
# the whole file first, which means loading the JSON, running the pipeline,
# and printing the table as a side effect of the import (and failing outright
# if weather_raw.json were missing). With the guard, main() runs only when the
# file is executed directly (`python report.py`), so importing is safe.
if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Reflection
#
# 1. Rejecting the whole file on a single null temperature.
#    Where it is right: when the data feeds something that assumes completeness
#    and silent gaps are dangerous, such as a model trained on this data, or a
#    pipeline that publishes daily highs/lows. One missing hour could hide
#    the real daily max, so failing loudly beats publishing a wrong number.
#    Where I'd tolerate it: a live dashboard or an ingestion job that runs
#    every hour. One bad sensor reading shouldn't block 167 good ones, and I
#    would rather store the partial data and flag the gap.
#    Schema change: make the lists nullable, i.e.
#    `temperature_2m: list[float | None]` (and the same for precipitation),
#    then have to_readings() skip or flag None entries so the aggregator
#    counts only real observations. min_hours then enforces completeness.
#
# 2. min_hours at noon.
#    If the pipeline runs at noon, today has only ~12 hours of data. Without
#    min_hours, we would report a "daily" high and low computed from the
#    morning alone, and the true afternoon max would be missing. The number
#    would look plausible but be wrong. With min_hours=24, the partial day is
#    dropped, and incomplete_days() returns today's date so the drop is
#    visible rather than silent. Callers can log it, retry later, or display
#    "today (in progress)".
#
# 3. Package vs. one file.
#    In Week 10, the pipeline can simply `from weatherkit import
#    WeatherResponse, DailyAggregator` (or install it as a dependency). With a
#    single script I'd have to copy-paste code or import from a file that runs
#    the whole report on import. Separate modules also let tests and other
#    pipeline steps import only the piece they need.
# ---------------------------------------------------------------------------
