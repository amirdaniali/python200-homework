# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 3

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "weather_classifier.pkl"
METADATA_PATH = BASE_DIR / "models" / "weather_classifier_metadata.json"


# =====================================================================
# Task 7: Load and Predict
# =====================================================================

# Stop with a clear message if the training script has not been run yet.
missing = [p.name for p in (MODEL_PATH, METADATA_PATH) if not p.exists()]
if missing:
    sys.exit(
        f"Could not find {', '.join(missing)} in {MODEL_PATH.parent}.\n"
        "These files are created by train_classifier.py. "
        "Run `python train_classifier.py` first, then run this script again."
    )

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text())

print("Model metadata:")
print("  City:          ", metadata["city"])
print("  Model type:    ", metadata["model_type"])
print("  Features:      ", metadata["features"])
print("  Test accuracy: ", round(metadata["test_accuracy"], 4))
print("  Test F1:       ", round(metadata["test_f1"], 4))

# Model metadata:
#   City:           Charlotte, NC
#   Model type:     LogisticRegression
#   Features:       ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'wind_speed_10m_max']
#   Test accuracy:  0.8767
#   Test F1:        0.8421

# Hypothetical days. Column order comes from the metadata so it matches training.
features = metadata["features"]
days = pd.DataFrame(
    [
        # label,                   high, low, precip, wind
        ("Clearly good day", 18.0, 8.0, 0.0, 10.0),
        ("Clearly bad day (storm)", 3.0, -5.0, 12.0, 40.0),
        ("Clearly bad day (heat)", 36.0, 24.0, 0.0, 8.0),
        ("Borderline (near limits)", 26.0, 1.0, 2.8, 28.0),
        ("Borderline (cold edge)", 7.5, 0.5, 2.5, 25.0),
    ],
    columns=["description"] + features,
)

X_new = days[features]
predictions = model.predict(X_new)
good_index = list(model.classes_).index(1)
prob_good = model.predict_proba(X_new)[:, good_index]

print("\nPredictions:")
for i, row in days.iterrows():
    label = "good" if predictions[i] == 1 else "skip"
    print(f"\n{row['description']}")
    for feat in features:
        print(f"  {feat}: {row[feat]}")
    print(f"  Predicted label: {label}")
    print(f"  Confidence (probability of good): {prob_good[i]:.2f}")

# Results:
# Predictions:
#
# Clearly good day
#   temperature_2m_max: 18.0
#   temperature_2m_min: 8.0
#   precipitation_sum: 0.0
#   wind_speed_10m_max: 10.0
#   Predicted label: good
#   Confidence (probability of good): 0.53
#
# Clearly bad day (storm)
#   temperature_2m_max: 3.0
#   temperature_2m_min: -5.0
#   precipitation_sum: 12.0
#   wind_speed_10m_max: 40.0
#   Predicted label: skip
#   Confidence (probability of good): 0.23
#
# Clearly bad day (heat)
#   temperature_2m_max: 36.0
#   temperature_2m_min: 24.0
#   precipitation_sum: 0.0
#   wind_speed_10m_max: 8.0
#   Predicted label: skip
#   Confidence (probability of good): 0.12
#
# Borderline (near limits)
#   temperature_2m_max: 26.0
#   temperature_2m_min: 1.0
#   precipitation_sum: 2.8
#   wind_speed_10m_max: 28.0
#   Predicted label: skip
#   Confidence (probability of good): 0.25
#
# Borderline (cold edge)
#   temperature_2m_max: 7.5
#   temperature_2m_min: 0.5
#   precipitation_sum: 2.5
#   wind_speed_10m_max: 25.0
#   Predicted label: good
#   Confidence (probability of good): 0.76

# =====================================================================
# Task 8: Reflect
# =====================================================================

"""
Q1. Borderline day.

Probability of good for Borderline (near limits) is 0.25 and Probability of good for Borderline (cold edge)
is 0.76. Since these probabilities are almost halfway between confident (1 and 0) and uncertain (0.5), I call the model uncertain.

Reasoning: for a day where the model predicts 0.52, I would not
treat it as a real "go" or "skip" decision. 0.52 means the model is nearly on the
fence, so the app should show it as "uncertain / could go either way" and let the
runner decide, rather than showing a confident "good day" label. I would also look at
the raw conditions (for example, a high near 26 or wind near 30) to explain why.

Q2. Running predict.py before train_classifier.py.
Without the saved files, joblib.load and open() would raise a FileNotFoundError
pointing at models/weather_classifier.pkl, which does not tell a new user what to do.
This script already checks for the missing files first and exits with a message that
names the missing files, says train_classifier.py creates them, and gives the exact
command to run. It could be improved further by also checking that the saved
scikit-learn version matches the one installed (the metadata records it), since a
mismatch can break or silently change a loaded model.

Q3. What would need to change for the Week 10 cloud pipeline.
The input days are hard-coded in the script. In a cloud pipeline each day's weather
would need to come in automatically (for example fetched from the weather API or read
from a data store, or pulled from cloud storage, or 3rd party integrations)
and the predictions would need to be written out (to a database,
file, or message queue) instead of only being printed. The file paths would also need
to be configurable rather than relative to this script's folder.
"""
