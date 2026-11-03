# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 4

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from weather_model import FEATURES, Prediction, WeatherClassifier

# Reusable example days.
GOOD_DAY = {
    "temperature_2m_max": 18,
    "temperature_2m_min": 9,
    "precipitation_sum": 0.0,
    "wind_speed_10m_max": 10,
}
BAD_DAY = {
    "temperature_2m_max": 3,
    "temperature_2m_min": -5,
    "precipitation_sum": 10.0,
    "wind_speed_10m_max": 38,
}


@pytest.fixture
def model_path(tmp_path):
    """Train a tiny model on obvious good and bad days and save it to a temp file."""
    good = [
        {
            "temperature_2m_max": 16 + i % 5,
            "temperature_2m_min": 7 + i % 4,
            "precipitation_sum": (i % 3) * 0.2,
            "wind_speed_10m_max": 8 + i % 6,
        }
        for i in range(15)
    ]
    bad = [
        {
            "temperature_2m_max": 1 + i % 4,
            "temperature_2m_min": -7 + i % 4,
            "precipitation_sum": 10 + i % 5,
            "wind_speed_10m_max": 36 + i % 6,
        }
        for i in range(15)
    ]
    X = pd.DataFrame(good + bad)[FEATURES]
    y = [1] * 15 + [0] * 15

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(X, y)

    path = tmp_path / "weather_classifier.pkl"
    joblib.dump(pipe, path)
    return path


@pytest.fixture
def clf(model_path):
    """A WeatherClassifier loaded from the saved tiny model."""
    return WeatherClassifier(model_path)


# --- Behaviors 1 and 2: clearly good days are "good", clearly bad days are "skip" ---
@pytest.mark.parametrize(
    "day, expected_label",
    [
        (GOOD_DAY, "good"),
        (
            {
                "temperature_2m_max": 20,
                "temperature_2m_min": 10,
                "precipitation_sum": 0.0,
                "wind_speed_10m_max": 12,
            },
            "good",
        ),
        (BAD_DAY, "skip"),
        (
            {
                "temperature_2m_max": 1,
                "temperature_2m_min": -8,
                "precipitation_sum": 14.0,
                "wind_speed_10m_max": 40,
            },
            "skip",
        ),
    ],
)
def test_clear_days_get_expected_label(clf, day, expected_label):
    result = clf.predict([day])
    assert result[0].label == expected_label


# --- Behavior 3: probability is between 0 and 1 ---
def test_probability_is_between_zero_and_one(clf):
    day = {
        "temperature_2m_max": 15,
        "temperature_2m_min": 8,
        "precipitation_sum": 1.0,
        "wind_speed_10m_max": 12,
    }
    result = clf.predict([day])
    assert 0.0 <= result[0].probability <= 1.0


# --- Behavior 4: three days in, three Prediction objects out, in order ---
def test_three_days_return_three_predictions_in_order(clf):
    results = clf.predict([GOOD_DAY, BAD_DAY, GOOD_DAY])
    assert len(results) == 3
    assert all(isinstance(r, Prediction) for r in results)
    assert [r.label for r in results] == ["good", "skip", "good"]


# --- Behavior 5: empty list in, empty list out ---
def test_empty_list_returns_empty_list(clf):
    assert clf.predict([]) == []


# --- Behavior 6: a missing feature raises ValueError ---
def test_missing_feature_raises_value_error(clf):
    incomplete_day = {"temperature_2m_max": 18, "temperature_2m_min": 9}
    with pytest.raises(ValueError, match="missing required features"):
        clf.predict([incomplete_day])


# --- Behavior 7: a missing model file raises FileNotFoundError ---
def test_missing_model_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model file"):
        WeatherClassifier(tmp_path / "does_not_exist.pkl")


# --- Extra: key order in the input dictionaries must not change the result ---
def test_key_order_does_not_change_result(clf):
    reordered = dict(reversed(list(GOOD_DAY.items())))
    assert clf.predict([reordered])[0] == clf.predict([GOOD_DAY])[0]


# Result:

# (.venv) PS M:\Make\CodeTheDream\python200-homework> pytest .\assignments_04\ -v
# ============================================================== test session starts ===============================================================
# platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0 -- M:\Make\CodeTheDream\python200-homework\.venv\Scripts\python.exe
# cachedir: .pytest_cache
# rootdir: M:\Make\CodeTheDream\python200-homework
# plugins: platformdirs-4.12.3
# collected 10 items
#
# assignments_04/tests/test_classifier.py::test_clear_days_get_expected_label[day0-good] PASSED                                               [ 10%]
# assignments_04/tests/test_classifier.py::test_clear_days_get_expected_label[day1-good] PASSED                                               [ 20%]
# assignments_04/tests/test_classifier.py::test_clear_days_get_expected_label[day2-skip] PASSED                                               [ 30%]
# assignments_04/tests/test_classifier.py::test_clear_days_get_expected_label[day3-skip] PASSED                                               [ 40%]
# assignments_04/tests/test_classifier.py::test_probability_is_between_zero_and_one PASSED                                                    [ 50%]
# assignments_04/tests/test_classifier.py::test_three_days_return_three_predictions_in_order PASSED                                           [ 60%]
# assignments_04/tests/test_classifier.py::test_empty_list_returns_empty_list PASSED                                                          [ 70%]
# assignments_04/tests/test_classifier.py::test_missing_feature_raises_value_error PASSED                                                     [ 80%]
# assignments_04/tests/test_classifier.py::test_missing_model_file_raises_file_not_found PASSED                                               [ 90%]
# assignments_04/tests/test_classifier.py::test_key_order_does_not_change_result PASSED                                                       [100%]
#
# ============================================================== 10 passed in 18.22s ==============================================================


# lets try to break the predict function by switching good and skip
# return values and seeing which tests fail: you can see the broken
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
