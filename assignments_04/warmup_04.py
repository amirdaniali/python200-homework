# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 4


import json
from dataclasses import dataclass, Field
from typing import Dict


import pytest

from pathlib import Path

# Paths are resolved relative to this file, so the script works from any directory.
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# Warmup Question 1
class ModelInfo:

    def __init__(self, path):
        self.data = json.load(open(path))

    def feature_count(self) -> int:
        return len(self.data["features"])


def make_features_json():
    with open(BASE_DIR.joinpath("features.json"), "w") as f:
        json.dump({"features": [1, 2, 3, 4, 5, 6, 7, 8, 7, 5, 2]}, f)


make_features_json()
info = ModelInfo(BASE_DIR / "features.json")

print("-" * 60)
print("Q1 - Feature count:", info.feature_count())

# Its better to initialize the data in the init because it is only done once
# when making the instance. Otherwise it would be called per call to feature_count.
# This saves memory and processing.

# Warmup Question 2


@dataclass()
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        return "".join(
            [
                "Day ",
                self.date,
                ": high ",
                str(self.high_c),
                ", rain expected" if self.will_rain else ", rain not expected",
            ]
        )


day1 = DayForecast("2023-06-15", 23, False)
day2 = DayForecast("2023-02-15", 3, True)
for day in [day1, day2]:
    print(day.summary())

# The decorator created the __repr__ , __init__, and __eq__ methods to represent data in the class
# and allow equality comparison between separate instances
# Otherwise equality will fall back on memory

# Warmup Question 3


def parse_temperature(text: str) -> float:
    try:
        return float(text.split(" ")[0])
    except ValueError:
        raise ValueError("temperature must be float")


@pytest.mark.parametrize(
    "values,expected",
    [
        ("24.1", 24.1),
        ("21.1", 21.1),
        ("1.2 ", 1.2),
    ],
)
def test_parse_temperature(values, expected):
    assert parse_temperature(values) == pytest.approx(expected)


def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# Result: ============================================================== test session starts ===============================================================
# platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0 -- M:\Make\CodeTheDream\python200-homework\.venv\Scripts\python.exe
# cachedir: .pytest_cache
# rootdir: M:\Make\CodeTheDream\python200-homework
# plugins: platformdirs-4.12.3
# collected 4 items
#
# assignments_04/warmup_04.py::test_parse_temperature[24.1-24.1] PASSED                                                                       [ 25%]
# assignments_04/warmup_04.py::test_parse_temperature[21.1-21.1] PASSED                                                                       [ 50%]
# assignments_04/warmup_04.py::test_parse_temperature[1.2 -1.2] PASSED                                                                        [ 75%]
# assignments_04/warmup_04.py::test_parse_temperature_bad_input_raises PASSED                                                                 [100%]
#
# =============================================================== 4 passed in 0.08s ================================================================
# (.venv) PS M:\Make\CodeTheDream\python200-homework>

# without matching the str "temperature" other errors could also be caught which is not our intention.
#  each test should test one small thing.
