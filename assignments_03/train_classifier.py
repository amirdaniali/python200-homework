# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 3

import json
import platform
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]


# =====================================================================
# Task 1: Fetch the Data
# =====================================================================

# City chosen: Charlotte, NC (latitude 35.23, longitude -80.84), year 2023.
CITY = "Charlotte, NC"

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 35.23,
    "longitude": -80.84,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": FEATURES,
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

# Drop any day with a missing measurement so the model never sees NaN values.
df = df.dropna(subset=FEATURES).reset_index(drop=True)

print("Task 1: Data shape:", df.shape)
print(df.head())

# Result Task 1: Data shape: (365, 5)
#    temperature_2m_max  temperature_2m_min  precipitation_sum  wind_speed_10m_max       date
# 0                15.3                 7.8                0.2                12.7 2023-01-01
# 1                18.5                 6.8                0.0                12.2 2023-01-02
# 2                20.4                11.7               14.8                24.5 2023-01-03
# 3                17.9                14.9               33.4                22.3 2023-01-04
# 4                17.2                 5.6                0.0                22.7 2023-01-05

# =====================================================================
# Task 2: Engineer the Label
# =====================================================================

# Thresholds are used exactly as given in the assignment (no changes):
#   high between 7 and 26 C, low >= 0 C, precipitation < 3 mm, max wind < 30 km/h.
LABEL_RULE = (
    "good_for_running = 1 if 7 <= temperature_2m_max <= 26 and temperature_2m_min >= 0 "
    "and precipitation_sum < 3 and wind_speed_10m_max < 30, else 0 "
    "(units: degrees C, mm, km/h)"
)

df["good_for_running"] = (
    (df["temperature_2m_max"] >= 7)
    & (df["temperature_2m_max"] <= 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)

class_counts = df["good_for_running"].value_counts().sort_index()
print("\nTask 2: Class balance (0 = skip, 1 = good):")
print(class_counts)
print("Task 2: Fraction of good days:", round(df["good_for_running"].mean(), 3))

# Result Task 2: Class balance (0 = skip, 1 = good):
# good_for_running
# 0    226
# 1    139
# Name: count, dtype: int64
# Task 2: Fraction of good days: 0.381

# The fraction seems reasonable for Charlotte NC. The split is not very
# lopsided (for example under 20% or over 80% would be lopsided)

# =====================================================================
# Task 3: Build and Evaluate a KNN Classifier
# =====================================================================

X = df[FEATURES]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv_scores = {}
print("\nTask 3: Mean 5-fold CV accuracy by k:")
for k in k_values:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    cv_scores[k] = scores.mean()
    print(f"  k={k}: {scores.mean():.4f}")

# Task 3: Mean 5-fold CV accuracy by k:
#   k=1: 0.9109
#   k=3: 0.9109
#   k=5: 0.8937
#   k=7: 0.8972
#   k=9: 0.8733
#   k=11: 0.8629
#   k=15: 0.8630
#   k=21: 0.8527

best_k = max(cv_scores, key=cv_scores.get)
print("Task 3: Chosen k:", best_k)

# Save a plot of CV accuracy against k.
plt.figure()
plt.plot(list(cv_scores.keys()), list(cv_scores.values()), marker="o")
plt.xlabel("k (number of neighbors)")
plt.ylabel("Mean 5-fold CV accuracy")
plt.title("KNN: choosing k with cross-validation")
plt.savefig(OUTPUTS_DIR / "knn_cv_scores.png", bbox_inches="tight")
plt.close()

knn_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=best_k)),
    ]
)
knn_pipeline.fit(X_train, y_train)
knn_preds = knn_pipeline.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_preds)

print("Task 3: KNN test accuracy:", round(knn_accuracy, 4))
print("Task 3: KNN classification report:")
print(classification_report(y_test, knn_preds, target_names=["skip", "good"]))

# Results:
# Task 3: Chosen k: 3
# Task 3: KNN test accuracy: 0.9726
# Task 3: KNN classification report:
#               precision    recall  f1-score   support
#
#         skip       1.00      0.96      0.98        45
#         good       0.93      1.00      0.97        28
#
#     accuracy                           0.97        73
#    macro avg       0.97      0.98      0.97        73
# weighted avg       0.97      0.97      0.97        73


# =====================================================================
# Task 4: Build and Evaluate a Logistic Regression Classifier
# =====================================================================

logreg_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ]
)
logreg_pipeline.fit(X_train, y_train)
lr_preds = logreg_pipeline.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_preds)
lr_precision = precision_score(y_test, lr_preds)
lr_recall = recall_score(y_test, lr_preds)
lr_f1 = f1_score(y_test, lr_preds)

print("\nTask 4: LogReg test accuracy: ", round(lr_accuracy, 4))
print("Task 4: LogReg precision:     ", round(lr_precision, 4))
print("Task 4: LogReg recall:        ", round(lr_recall, 4))
print("Task 4: LogReg F1:            ", round(lr_f1, 4))

# Result:
# Task 4: LogReg test accuracy:  0.8767
# Task 4: LogReg precision:      0.8276
# Task 4: LogReg recall:         0.8571
# Task 4: LogReg F1:             0.8421

print("Task 4: LogReg classification report:")
print(classification_report(y_test, lr_preds, target_names=["skip", "good"]))

#               precision    recall  f1-score   support
#
#         skip       0.91      0.89      0.90        45
#         good       0.83      0.86      0.84        28
#
#     accuracy                           0.88        73
#    macro avg       0.87      0.87      0.87        73
# weighted avg       0.88      0.88      0.88        73

coefs = logreg_pipeline.named_steps["logreg"].coef_[0]
print("Task 4: Coefficients (on standardized features):")
for name, coef in zip(FEATURES, coefs):
    print(f"  {name}: {coef:.4f}")

# Task 4: Coefficients (on standardized features):
#   temperature_2m_max: -0.9343
#   temperature_2m_min: 0.1186
#   precipitation_sum: -2.4916
#   wind_speed_10m_max: 0.3198

strongest_skip = FEATURES[int(coefs.argmin())]
print("Task 4: Most negative coefficient (strongest push toward skip):", strongest_skip)

# Results: Task 4: Most negative coefficient (strongest push toward skip): precipitation_sum

# precipitation_sum most strongly pushes a day toward "skip" because it is the most negative.
# Reasoning : negative coefficients push toward "skip" and positive
# ones toward "good". The label rule treats the high temperature as a two-sided band
# (7 to 26), which a single linear coefficient cannot represent, so the
# temperature_2m_max coefficient will understate how much extreme heat or cold matters.

# Save a confusion matrix plot for the logistic regression model.
cm = confusion_matrix(y_test, lr_preds)
ConfusionMatrixDisplay(cm, display_labels=["skip", "good"]).plot()
plt.title("Logistic Regression Confusion Matrix")
plt.savefig(OUTPUTS_DIR / "weather_logreg_confusion_matrix.png", bbox_inches="tight")
plt.close()


# =====================================================================
# Task 5: Compare and Choose
# =====================================================================

higher = "KNN" if knn_accuracy > lr_accuracy else "Logistic Regression"
if knn_accuracy == lr_accuracy:
    higher = "Tie"
print(
    "\nTask 5: KNN accuracy:",
    round(knn_accuracy, 4),
    "| LogReg accuracy:",
    round(lr_accuracy, 4),
)
print("Task 5: Higher test accuracy:", higher)

# Result:
# Task 5: KNN accuracy: 0.9726 | LogReg accuracy: 0.8767
# Task 5: Higher test accuracy: KNN

# Metric choice: for a running app, accuracy alone is not what I care
# about most. A false positive means sending someone out on a bad day (rain, storm,
# freezing), which is the costly error, so precision for the "good" class matters most.
# A false negative just means a missed good day, which is a smaller cost, so recall
# matters but less.

# Deployment choice (already complete): I will deploy Logistic Regression.
#   - Size: it stores only a handful of coefficients, while KNN stores the entire
#     training set and keeps it in the saved file.
#   - Speed: prediction is one dot product, while KNN must compute distances to every
#     stored training day for each new prediction.
#   - Interpretability: the coefficients show how each weather variable pushes a day
#     toward good or skip, which I can explain to app users.
#   - Probabilities: logistic regression gives smooth probabilities of "good", while
#     KNN's probabilities are coarse fractions of k neighbors.
# If the accuracy gap printed above turns out to be large in KNN's favor, revisit this
# choice, since a big accuracy difference could outweigh these practical advantages.

chosen_name = "LogisticRegression"
chosen_pipeline = logreg_pipeline
chosen_preds = lr_preds


# =====================================================================
# Task 6: Save the Model
# =====================================================================

model_path = MODELS_DIR / "weather_classifier.pkl"
metadata_path = MODELS_DIR / "weather_classifier_metadata.json"

joblib.dump(chosen_pipeline, model_path)

metadata = {
    "model_type": chosen_name,
    "python_version": platform.python_version(),
    "sklearn_version": sklearn.__version__,
    "features": FEATURES,
    "label_rule": LABEL_RULE,
    "city": CITY,
    "test_accuracy": float(accuracy_score(y_test, chosen_preds)),
    "test_f1": float(f1_score(y_test, chosen_preds)),
}
metadata_path.write_text(json.dumps(metadata, indent=2))

print(f"\nTask 6: Saved model to {model_path}")
print(f"Task 6: Saved metadata to {metadata_path}")
