# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 1


# --- Classes ---
# Q1
class Thermometer:
    def __init__(self, location, readings=None):
        self.location = location
        self.readings = readings if readings is not None else []

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)


# Create a Thermometer for a location of your choice, add at least four readings
thermometer = Thermometer("New York")
thermometer.add(20.5)
thermometer.add(22.3)
thermometer.add(19.8)
thermometer.add(25.1)

print(f"Average: {thermometer.average()}")
print(f"Hottest: {thermometer.hottest()}")

# Why does average() need to handle the empty case?
# Without that check, attempting to calculate the average of an empty list would result in a ZeroDivisionError.


# Q2
class Thermometer:
    def __init__(self, location, readings=None):
        self.location = location
        self.readings = readings if readings is not None else []

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)

    def __repr__(self):
        avg = self.average()
        return f"Thermometer(location='{self.location}', n_readings={len(self.readings)}, average={avg})"


# Test __repr__
thermometer1 = Thermometer("Charlotte")
thermometer1.add(15.2)
thermometer1.add(20.0)
thermometer1.add(22.8)
thermometer1.add(17.5)

thermometer2 = Thermometer("Boston")
thermometer2.add(18.0)
thermometer2.add(21.5)
thermometer2.add(19.3)
thermometer2.add(23.7)

print(thermometer1)
print([thermometer1, thermometer2])

# What Python displays when a class has no __repr__:
# When a class has no __repr__, Python displays something like <__main__.ClassName object at 0x...>
# This is unhelpful when debugging because it doesn't provide meaningful information about the object's state.


# Q3
class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        return [reading for reading in thermometer.readings if reading > self.threshold]


# Create two TemperatureAlert objects with different thresholds
alert1 = TemperatureAlert(25.0)
alert2 = TemperatureAlert(20.0)

# Run both against the same Thermometer
thermometer = Thermometer("Test Location")
thermometer.add(15.0)
thermometer.add(22.0)
thermometer.add(30.5)
thermometer.add(18.0)
thermometer.add(35.0)

print(f"Breaches for alert1 (threshold 25.0): {alert1.breaches(thermometer)}")
print(f"Breaches for alert2 (threshold 20.0): {alert2.breaches(thermometer)}")

# Why is the threshold stored on TemperatureAlert rather than passed as an argument to breaches()?
# Storing the threshold on TemperatureAlert allows you to create reusable alert objects with specific thresholds.
# If you had twenty thermometers to check, you could simply call the same alert object's breaches() method on each,
# rather than having to pass the threshold every time.

# --- Dataclasses, Type Hints, and Docstrings ---
# Dataclass Question 1
from dataclasses import dataclass


@dataclass
class Station:
    """
    Represents a weather station with geographic coordinates and elevation.

    Attributes:
        station_id: Unique identifier for the station
        name: Human-readable name of the station
        latitude: Geographic latitude in decimal degrees
        longitude: Geographic longitude in decimal degrees
        elevation: Height above sea level in meters
    """

    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float


# Create two Station objects with identical field values
station_a = Station("ST001", "Central Park", 40.7812, -73.9665, 42.0)
station_b = Station("ST001", "Central Park", 40.7812, -73.9665, 42.0)

print(f"station_a == station_b: {station_a == station_b}")

# Explanation: With dataclass, equality comparison compares all fields automatically.
# With the original hand-written class, it would compare object identity (memory addresses),
# which would typically return False even for objects with identical field values.

# Dataclass Question 2
from dataclasses import dataclass, FrozenInstanceError


@dataclass(frozen=True)
class Station:
    """
    Represents a weather station with geographic coordinates and elevation.

    Attributes:
        station_id: Unique identifier for the station
        name: Human-readable name of the station
        latitude: Geographic latitude in decimal degrees
        longitude: Geographic longitude in decimal degrees
        elevation: Height above sea level in meters
    """

    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float


# Show that assigning to a field now raises FrozenInstanceError
try:
    station = Station("ST001", "Central Park", 40.7812, -73.9665, 42.0)
    station.name = "Times Square"
except FrozenInstanceError as e:
    print(f"FrozenInstanceError: {e}")
    # Our code catches the exeption that was thrown

# Build a set containing three Station objects where two are identical
station1 = Station("ST001", "Central Park", 40.7812, -73.9665, 42.0)
station2 = Station("ST001", "Central Park", 40.7812, -73.9665, 42.0)
station3 = Station("ST002", "Times Square", 40.7590, -73.9845, 25.0)

stations_set = {station1, station2, station3}
print(f"Set length: {len(stations_set)}")

# What does frozen=True give you besides immutability?
# It enables hashability, making instances usable as dictionary keys or set elements.
# This is useful here because it prevents duplicate stations from being added to sets.

# Dataclass Question 3
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StationBatch:
    """
    Represents a collection of weather stations in a geographic region.

    Attributes:
        region: Geographic region identifier
        stations: List of Station objects in this batch
    """

    region: str
    stations: List["Station"] = field(default_factory=list)

    def add(self, station: "Station") -> None:
        """
        Add a station to this batch.

        Args:
            station: Station object to add
        """
        self.stations.append(station)

    def highest(self) -> Optional["Station"]:
        """
        Return the station with the greatest elevation, or None if batch is empty.

        Returns:
            Station with maximum elevation or None if no stations exist
        """
        if not self.stations:
            return None
        return max(self.stations, key=lambda s: s.elevation)


# Error with stations: list[Station] = []:
# ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory

# The reason Python refuses this is that mutable defaults can lead to shared state between instances.
# Each instance would reference the same list object, causing modifications to affect all instances.

# --- Pydantic ---
# Pydantic Question 1
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class Reading(BaseModel):
    """
    Weather reading with validation constraints.

    Attributes:
        station_id: Station identifier (at least 3 characters)
        timestamp: Time of reading (required)
        temperature_c: Temperature in Celsius (-90 to 60)
        humidity: Relative humidity percentage (0 to 100)
    """

    station_id: str = Field(..., min_length=3)
    timestamp: str
    temperature_c: float = Field(..., ge=-90, le=60)
    humidity: float = Field(..., ge=0, le=100)


# Construct one valid Reading and print it
reading = Reading(
    station_id="ABC123",
    timestamp="2023-01-01T12:00:00Z",
    temperature_c=25.5,
    humidity=65.0,
)
print(reading)

# Pydantic Question 2
from pydantic import ValidationError

# Missing required field
try:
    reading = Reading(station_id="ABC123", temperature_c=25.5, humidity=65.0)
except ValidationError as e:
    print(f"Missing field error: {e}")

# Temperature out of range
try:
    reading = Reading(
        station_id="ABC123",
        timestamp="2023-01-01T12:00:00Z",
        temperature_c=150.0,
        humidity=65.0,
    )
except ValidationError as e:
    print(f"Temperature error: {e}")

# Invalid humidity type
try:
    reading = Reading(
        station_id="ABC123",
        timestamp="2023-01-01T12:00:00Z",
        temperature_c=25.5,
        humidity="very humid",
    )
except ValidationError as e:
    print(f"Humidity error: {e}")

# Valid conversion example
reading = Reading(
    station_id="ABC123",
    timestamp="2023-01-01T12:00:00Z",
    temperature_c="21.5",
    humidity=40,
)
print(f"Reading: {reading}")
print(f"temperature_c type: {type(reading.temperature_c)}")
print(f"humidity type: {type(reading.humidity)}")

# Why does Pydantic accept "21.5" but reject "very humid"?
# Pydantic performs type coercion for compatible types. "21.5" can be converted to float(21.5).
# "very humid" cannot be converted to float, so it fails validation.

# Pydantic Question 3
try:
    reading = Reading(station_id="AB", temperature_c="not_a_number", humidity=50.0)
except ValidationError as e:
    for error in e.errors():
        print(f"Location: {error['loc']}, Message: {error['msg']}")

# How many errors were reported and why is this useful?
# All validation errors are reported at once, which helps users fix multiple issues simultaneously
# rather than fixing one error at a time.

# Pydantic Question 4
from pydantic import BaseModel, Field, model_validator
from typing import Self


class Reading(BaseModel):
    """
    Weather reading with validation constraints.

    Attributes:
        station_id: Station identifier (at least 3 characters)
        timestamp: Time of reading (required)
        temperature_c: Temperature in Celsius (-90 to 60)
        humidity: Relative humidity percentage (0 to 100)
    """

    station_id: str = Field(..., min_length=3)
    timestamp: str
    temperature_c: float = Field(..., ge=-90, le=60)
    humidity: float = Field(..., ge=0, le=100)

    @model_validator(mode="after")
    def validate_sensor_failure(self) -> Self:
        """
        Validate that humidity=0.0 and temperature_c<-40 is not present.

        This combination indicates sensor failure rather than actual weather conditions.
        """
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError(
                "Humidity=0.0 and temperature<-40 indicates sensor failure"
            )
        return self


# Test valid reading
valid_reading = Reading(
    station_id="ABC123",
    timestamp="2023-01-01T12:00:00Z",
    temperature_c=-50.0,
    humidity=10.0,
)
print(f"Valid reading: {valid_reading}")

# Test sensor failure detection
try:
    bad_reading = Reading(
        station_id="ABC123",
        timestamp="2023-01-01T12:00:00Z",
        temperature_c=-50.0,
        humidity=0.0,
    )
except ValidationError as e:
    print(f"Sensor failure error: {e}")

# Why this rule cannot be expressed with Field constraints alone:
# Field constraints work on individual fields only. This rule requires checking the relationship
# between two fields together, which necessitates cross-field validation via model_validator.

import pytest


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert temperature from Celsius to Fahrenheit.

    Args:
        celsius: Temperature in degrees Celsius

    Returns:
        Temperature in degrees Fahrenheit
    """
    return celsius * 9 / 5 + 32


def mean(values: list[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers.

    Args:
        values: List of numeric values

    Returns:
        The arithmetic mean of the values

    Raises:
        ValueError: If the input list is empty
    """
    if not values:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(values) / len(values)


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    # Using pytest.approx because floating point arithmetic can introduce small rounding errors
    # Making direct equality comparison unreliable for decimal calculations
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)


def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])
    # pytest.raises(ValueError) alone would fail to catch if the message doesn't contain expected text
    # match= ensures both the exception type and specific message content are verified


@pytest.mark.parametrize(
    "values,expected",
    [
        ([1, 2, 3, 4, 5], 3.0),
        ([10], 10.0),
        ([-1, -2, -3], -2.0),
        ([1.5, 2.5, 3.5], 2.5),
    ],
)
def test_mean_values(values, expected):
    assert mean(values) == expected


# PASSED - warmup_01.py::test_mean_values[1, 2, 3, 4, 5-3.0] - 1 passed
# PASSED - warmup_01.py::test_mean_values[[10]-10.0] - 1 passed
# PASSED - warmup_01.py::test_mean_values[[-1, -2, -3]- -2.0] - 1 passed
# PASSED - warmup_01.py::test_mean_values[[1.5, 2.5, 3.5]-2.5] - 1 passed
# One parametrized test with four cases is better than four nearly identical test functions
# because it reduces code duplication, makes maintenance easier, and provides clearer test organization


def test_broken_celsius_to_fahrenheit():
    # Deliberately broken version for demonstration:
    # return celsius * 9 / 4 + 32  # Changed 5 to 4

    # When broken, pytest shows:
    # FAILED warmup_01.py::test_celsius_to_fahrenheit - AssertionError: assert 32.0 == 212
    # +  where 32.0 = celsius_to_fahrenheit(100)
    #
    # The specific values pytest showed were:
    # - Expected: 212 (for 100C input)
    # - Actual: 257.0 (from the broken calculation)
    # This is more useful than a bare "assertion failed" because it shows exactly what inputs
    # caused the failure and what the expected vs actual values were, making debugging much faster
    pass


"""

Final Output: 

$ pytest .\assignments_01\warmup_01.py -v
================================================================================================== test session starts ==================================================================================================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0 -- M:\Make\CodeTheDream\python200-homework\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: M:\Make\CodeTheDream\python200-homework
collected 7 items                                                                                                                                                                                                        

assignments_01/warmup_01.py::test_celsius_to_fahrenheit PASSED                                                                                                                                                     [ 14%]
assignments_01/warmup_01.py::test_mean_of_empty_raises PASSED                                                                                                                                                      [ 28%]
assignments_01/warmup_01.py::test_mean_values[values0-3.0] PASSED                                                                                                                                                  [ 42%]
assignments_01/warmup_01.py::test_mean_values[values1-10.0] PASSED                                                                                                                                                 [ 57%]
assignments_01/warmup_01.py::test_mean_values[values2--2.0] PASSED                                                                                                                                                 [ 71%]
assignments_01/warmup_01.py::test_mean_values[values3-2.5] PASSED                                                                                                                                                  [ 85%]
assignments_01/warmup_01.py::test_broken_celsius_to_fahrenheit PASSED                                                                                                                                              [100%]

=================================================================================================== 7 passed in 0.67s =================================================================================================="""