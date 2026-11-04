# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 4

"""Classify a few hypothetical days using the saved weather model."""

from pathlib import Path

from weather_model import Prediction, WeatherClassifier

# The model trained and saved in Week 3, copied into assignments_04/models/.
MODEL_PATH = Path(__file__).parent / "models" / "weather_classifier.pkl"

SAMPLE_DAYS = [
    # Clearly good: mild, dry, calm.
    {
        "temperature_2m_max": 18,
        "temperature_2m_min": 9,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 10,
    },
    # Clearly bad: cold, rainy, windy.
    {
        "temperature_2m_max": 3,
        "temperature_2m_min": -5,
        "precipitation_sum": 12.0,
        "wind_speed_10m_max": 40,
    },
    # Clearly bad: very hot.
    {
        "temperature_2m_max": 36,
        "temperature_2m_min": 24,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 8,
    },
    # Borderline: warm, but wind just under the 30 km/h limit and light rain.
    {
        "temperature_2m_max": 25,
        "temperature_2m_min": 15,
        "precipitation_sum": 2.5,
        "wind_speed_10m_max": 28,
    },
    # Borderline: cool day near the 7 degree lower limit.
    {
        "temperature_2m_max": 8,
        "temperature_2m_min": 1,
        "precipitation_sum": 0.5,
        "wind_speed_10m_max": 25,
    },
]


def describe_day(day: dict, result: Prediction) -> str:
    """Format one day's inputs and its prediction as a readable line."""
    return (
        f"{day['temperature_2m_max']:.0f}C high, {day['temperature_2m_min']:.0f}C low, "
        f"{day['precipitation_sum']:.1f}mm rain, {day['wind_speed_10m_max']:.0f} km/h wind "
        f"-> {result.label} ({result.probability:.2f})"
    )


def main() -> None:
    """Load the component and print a prediction for each sample day."""
    classifier = WeatherClassifier(MODEL_PATH)
    results = classifier.predict(SAMPLE_DAYS)
    for day, result in zip(SAMPLE_DAYS, results):
        print(describe_day(day, result))


#     result:
# 18C high, 9C low, 0.0mm rain, 10 km/h wind -> good (0.54)
# 3C high, -5C low, 12.0mm rain, 40 km/h wind -> skip (0.23)
# 36C high, 24C low, 0.0mm rain, 8 km/h wind -> skip (0.12)
# 25C high, 15C low, 2.5mm rain, 28 km/h wind -> skip (0.35)
# 8C high, 1C low, 0.5mm rain, 25 km/h wind -> good (0.87)

# The guard below means main() only runs when this file is executed directly
# (python predict_weather.py). Without it, any other script that did
# `from predict_weather import describe_day` just to reuse the helper would trigger
# main() at import time: the model would be loaded, predictions made, and the whole
# report printed before the importing script did anything of its own. If the model
# file were missing, the import itself would crash with FileNotFoundError.
if __name__ == "__main__":
    main()


# =====================================================================
# Task 6: Reflection
# =====================================================================
"""
Q1. Why select the four features in a fixed order?
A caller might build a day dictionary with the keys in a different order, for example
wind_speed_10m_max first. If predict() passed the columns straight through in the
caller's order, a model that matches columns by position would apply the
temperature_2m_max scaling and coefficient to the wind speed values (and vice versa).
A 33 km/h wind would be treated like a 33 degree high, and the model would return
confident but wrong predictions without raising any error. Selecting self.features in
the training order prevents that, whatever order the caller used.

Q2. Why do the tests train their own tiny model instead of loading the real file?
The tests become self-contained, fast, and deterministic. They do not fail just because
models/weather_classifier.pkl is missing, was retrained, or was saved by a different
scikit-learn version on another machine. The tiny model is trained on obvious good and
bad days, so the expected labels never change, whereas the real model's predictions on
borderline days can shift every time it is retrained. The tests check the component's
behavior (labels, ordering, errors), not the quality of one particular trained model.

Q3. Week 10 pipeline.
Ready: the model is loaded once in __init__ and reused by predict(), the input is plain
dictionaries (easy to build from database rows), predict() processes a whole batch in
one call, and bad input produces clear errors (ValueError, FileNotFoundError) rather
than obscure failures.
To add before production: validation of the values themselves, such as rejecting NaN or
None readings, negative precipitation, or impossible temperatures, since right now a
missing measurement in a database row would reach the model unchecked. Logging of each
batch and of the model version would also help, so wrong predictions can be traced.
"""
