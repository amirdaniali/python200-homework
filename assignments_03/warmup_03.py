# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import joblib
from pathlib import Path

# Paths are resolved relative to this file, so the script works from any directory.
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(BASE_DIR / "weather_classification.csv")
numeric = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# NOTE: this code assumes good_for_running is encoded 0 = "skip", 1 = "good".
# If yours uses strings, check df["good_for_running"].unique() and pass pos_label="good"
# to precision_score / recall_score / f1_score.


# =====================================================================
# --- Preprocessing ---
# =====================================================================

# --- Preprocessing Q1 ---
# Q1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train only, then transform
X_test_scaled = scaler.transform(X_test)  # transform only, no refitting

print("Q1: Mean of each scaled training column:")
for col, mean in zip(numeric, X_train_scaled.mean(axis=0)):
    print(f"  {col}: {mean:.6f}")

# We fit the scaler on X_train only so that no information from the test set
# (its mean and standard deviation) leaks into training, which keeps the test
# score an honest estimate of performance on unseen data.


# --- Preprocessing Q2 ---
# Q2
encoder = OneHotEncoder(sparse_output=False)
season_encoded = encoder.fit_transform(df[["season"]])

print("\nQ2: One-hot encoded season shape:", season_encoded.shape)
print("Q2: Category names:", encoder.get_feature_names_out())

# Result:
# Q2: One-hot encoded season shape: (600, 4)
# Q2: Category names: ['season_fall' 'season_spring' 'season_summer' 'season_winter']

# season became 4 columns.
# Reasoning one-hot encoding makes one binary column per unique
# season value, so the count equals the number of distinct seasons in the data. Each
# row has a 1 in the column for its own season and 0 in the rest, so the model does
# not treat the categories as ordered numbers.


# --- Preprocessing Q3 ---
# Q3
features_with_season = numeric + ["season"]
X_full = df[features_with_season]
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(), ["season"]),
    ]
)

season_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)
season_pipeline.fit(X_train_full, y_train_full)
print(
    "\nQ3: Pipeline test accuracy:",
    round(season_pipeline.score(X_test_full, y_test_full), 4),
)

# The pipeline automatically fits the scaler and encoder on the training data only,
# applies the same transformations to any new data (including at predict time), and
# feeds the combined result into the model, so I don't have to scale, encode,
# stack the columns, and repeat all of that by hand for the test set.


# =====================================================================
# --- KNN ---
# =====================================================================

# --- KNN Q1 ---
# Q1
knn_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)
knn_pipeline.fit(X_train, y_train)
knn_preds = knn_pipeline.predict(X_test)

print("\nKNN Q1: Test accuracy:", round(accuracy_score(y_test, knn_preds), 4))
print("KNN Q1: Classification report:")
print(classification_report(y_test, knn_preds))


# --- KNN Q2 ---
# Q2
print("KNN Q2: Mean 5-fold CV accuracy by k:")
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv_results = {}
for k in k_values:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    cv_results[k] = scores.mean()
    print(f"  k={k}: {scores.mean():.4f}")

best_k = max(cv_results, key=cv_results.get)
print("KNN Q2: Best k by CV accuracy:", best_k)

# Result: KNN Q2: Best k by CV accuracy: 11

# I would choose k = 11
# Reasoning: I choose the k with the highest mean CV accuracy, and
# if several values are nearly tied I take the larger one, since a bigger k gives a
# smoother decision boundary that is less sensitive to noise than k=1.
# In this case, K=11 is the best value.


# --- KNN Q3 ---
# Q3
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
raw_acc = knn_raw.score(X_test, y_test)

knn_scaled = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)
knn_scaled.fit(X_train, y_train)
scaled_acc = knn_scaled.score(X_test, y_test)

print("\nKNN Q3: Test accuracy without scaling:", round(raw_acc, 4))
print("KNN Q3: Test accuracy with scaling:   ", round(scaled_acc, 4))

# Result:
# KNN Q3: Test accuracy without scaling: 0.925
# KNN Q3: Test accuracy with scaling:    0.9583

# Scaling improved the test accuracy by roughly 0.033.
# Reasoning: scaling matters for a distance-based model because KNN
# measures raw Euclidean distance, so a feature with a large numeric range (like
# temperature) dominates the distance while a feature with a small range (like
# precipitation) is nearly ignored, even if it is more informative.


# =====================================================================
# --- Logistic Regression ---
# =====================================================================

# --- Logistic Regression Q1 ---
# Q1
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ]
)
pipeline.fit(X_train, y_train)
lr_preds = pipeline.predict(X_test)

print("\nLogReg Q1: Test accuracy: ", round(accuracy_score(y_test, lr_preds), 4))
print("LogReg Q1: Precision:     ", round(precision_score(y_test, lr_preds), 4))
print("LogReg Q1: Recall:        ", round(recall_score(y_test, lr_preds), 4))
print("LogReg Q1: F1:            ", round(f1_score(y_test, lr_preds), 4))


# --- Logistic Regression Q2 ---
# Q2
coefs = pipeline.named_steps["logreg"].coef_[0]
print("\nLogReg Q2: Coefficients (on standardized features):")
for name, coef in zip(numeric, coefs):
    print(f"  {name}: {coef:.4f}")

# Results:
# LogReg Q2: Coefficients (on standardized features):
#   temperature_2m_max: -0.1212
#   temperature_2m_min: 0.2563
#   precipitation_sum: -1.8891
#   wind_speed_10m_max: -2.4438

# precipitation_sum and wind_speed_10m_max push most strongly toward skip, this matches
# my expectation since few people like rainy or windy days for their outdoor activity.
# Reasoning : negative coefficients push the prediction toward "skip"
# (class 0) and positive ones toward "good" (class 1). Temperature is tricky for a
# linear model, since very hot days are bad and mild days are good, so one coefficient
# cannot capture both ends.


# --- Logistic Regression Q3 ---
# Q3
probs = pipeline.predict_proba(X_test)[:, 1]  # column 1 = probability of "good"
print("\nLogReg Q3: P(good) for first five test days:")
print(np.round(probs[:5], 2))

# A probability near 0.5 means the model is basically on the fence: the day's features
# sit close to the decision boundary, so the model has low confidence and the prediction
# could easily flip with a small change in the weather.


# =====================================================================
# --- Evaluation ---
# =====================================================================

# --- Evaluation Q1 ---
# Q1
cm = confusion_matrix(y_test, lr_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["skip", "good"])
disp.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.savefig(OUTPUTS_DIR / "logreg_confusion_matrix.png", bbox_inches="tight")
plt.close()

tn, fp, fn, tp = cm.ravel()
print("\nEval Q1: Confusion matrix:\n", cm)
print(f"Eval Q1: False positives: {fp}, False negatives: {fn}")

# Results:
# Eval Q1: Confusion matrix:
#  [[50  9]
#  [ 7 54]]
# Eval Q1: False positives: 9, False negatives: 7

# for a running app, false positives (telling someone it's
# a good day when it's actually bad) probably matter more, since a runner could get
# caught in a storm or extreme weather; a false negative just means a missed chance to
# run on a decent day. A recommendation that makes a user miss other good experiences
# and forces them to run in poor weather will lead to user churn.


# =====================================================================
# --- joblib ---
# =====================================================================

# --- joblib Q1 ---
# Q1
warmup_model_path = MODELS_DIR / "warmup_model.pkl"
joblib.dump(pipeline, warmup_model_path)
loaded = joblib.load(warmup_model_path)
assert (loaded.predict(X_test) == pipeline.predict(X_test)).all()
print("\njoblib Q1: Predictions match.")

# If I had saved only the LogisticRegression step without the scaler, then calling
# .predict() on raw data would feed unscaled values into a model trained on standardized
# features. The coefficients would be applied to the wrong scale, producing wrong and
# unreliable predictions without raising an error.
