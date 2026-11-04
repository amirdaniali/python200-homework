# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 4

"""The WeatherClassifier component: loads a saved pipeline and predicts running weather."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

# The four input features, in the exact order the model was trained on.
FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]


# --- Task 1: The Prediction dataclass ---
@dataclass
class Prediction:
    """The model's verdict for a single day.

    Attributes:
        label: Either "good" or "skip".
        probability: The probability, from 0 to 1, that the day is good for running.
    """

    label: str
    probability: float


# --- Task 2: The WeatherClassifier class ---
class WeatherClassifier:
    """Loads a saved weather pipeline once and predicts on many days."""

    def __init__(self, model_path: str | Path) -> None:
        """Load the saved pipeline from disk.

        Args:
            model_path: Path to a .pkl file saved with joblib.

        Raises:
            FileNotFoundError: If no file exists at model_path.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model file at {model_path}. Train and save a model first "
                "(run train_classifier.py from Week 3, then copy the .pkl here)."
            )
        self.model = joblib.load(model_path)
        self.features = FEATURES

    def predict(self, days: list[dict]) -> list[Prediction]:
        """Predict whether each day is good for running.

        Args:
            days: A list of dictionaries, each with the keys temperature_2m_max,
                temperature_2m_min, precipitation_sum, and wind_speed_10m_max.

        Returns:
            One Prediction per input day, in the same order. An empty list if
            days is empty.

        Raises:
            ValueError: If any required feature is missing from the input days.
        """
        if not days:
            return []

        frame = pd.DataFrame(days)
        missing = [f for f in self.features if f not in frame.columns]
        if missing:
            raise ValueError(f"missing required features: {missing}")

        # Select the features in the fixed training order, whatever order the caller used.
        X = frame[self.features]
        labels = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        # lets try to break this function by switching good and skip
        # and seeing which tests fail: you can see the broken
        # return expression  in the comment below.

        # return [
        #     Prediction("skip" if int(label) == 1 else "good", float(prob))
        #     for label, prob in zip(labels, probabilities)
        # ]

        # result: collected 10 items
        #
        # assignments_04/tests/test_classifier.py::test_clear_days_get_expected_label[day0-good] FAILED                                               [ 10%]
        # assignments_04/tests/test_classifier.py::test_clear_days_get_expected_label[day1-good] FAILED                                               [ 20%]
        # assignments_04/tests/test_classifier.py::test_clear_days_get_expected_label[day2-skip] FAILED                                               [ 30%]
        # assignments_04/tests/test_classifier.py::test_clear_days_get_expected_label[day3-skip] FAILED                                               [ 40%]
        # assignments_04/tests/test_classifier.py::test_probability_is_between_zero_and_one PASSED                                                    [ 50%]
        # assignments_04/tests/test_classifier.py::test_three_days_return_three_predictions_in_order FAILED                                           [ 60%]
        # assignments_04/tests/test_classifier.py::test_empty_list_returns_empty_list PASSED                                                          [ 70%]
        # assignments_04/tests/test_classifier.py::test_missing_feature_raises_value_error PASSED                                                     [ 80%]
        # assignments_04/tests/test_classifier.py::test_missing_model_file_raises_file_not_found PASSED                                               [ 90%]
        # assignments_04/tests/test_classifier.py::test_key_order_does_not_change_result PASSED                                                       [100%

        return [
            Prediction("good" if int(label) == 1 else "skip", float(prob))
            for label, prob in zip(labels, probabilities)
        ]
