# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 2

# Week 2 Warmup Exercises
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Save outputs next to this script, no matter where it is run from
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# --- scikit-learn API ---

# Q1
temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

model = LinearRegression()  # create
model.fit(temp_min, temp_max)  # fit
preds = model.predict([[6], [18]])  # predict (2D input: 2 rows, 1 feature)

print("Q1 slope:", model.coef_[0])
print("Q1 intercept:", model.intercept_)
print("Q1 predicted high for low of 6:", preds[0])
print("Q1 predicted high for low of 18:", preds[1])

# Q2
x = np.array([10, 20, 30, 40, 50])
print("\nQ2 original shape:", x.shape)
x_2d = x.reshape(-1, 1)
print("Q2 reshaped shape:", x_2d.shape)
# scikit-learn expects X to be shaped (n_samples, n_features): rows are
# observations and columns are features. A 1D array is ambiguous (is it
# 5 samples with 1 feature, or 1 sample with 5 features?), so we reshape to
# make it explicit: 5 rows, 1 column.

# --- Linear Regression ---

rng = np.random.default_rng(42)
num_days = 120
temp_min = rng.uniform(-5, 22, num_days)
is_rainy = rng.integers(0, 2, num_days).astype(float)
temp_max = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

# Q1
plt.figure(figsize=(7, 5))
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")
plt.xlabel("Daily Low Temperature")
plt.ylabel("Daily High Temperature")
plt.savefig(OUT_DIR / "high_vs_low.png", dpi=150, bbox_inches="tight")
plt.close()
# The points form a clear upward linear trend, and there are two visible,
# parallel bands: rainy days (one color) sit consistently below dry days
# (the other color) at the same low temperature. That suggests is_rainy
# shifts the high down by a roughly constant amount, so it should be a
# useful second feature.

# Q2
X = temp_min.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, temp_max, test_size=0.2, random_state=42
)
print("\nLR Q2 X_train shape:", X_train.shape)
print("LR Q2 X_test shape:", X_test.shape)
print("LR Q2 y_train shape:", y_train.shape)
print("LR Q2 y_test shape:", y_test.shape)

# Q3
model = LinearRegression()
model.fit(X_train, y_train)
print("\nLR Q3 slope:", model.coef_[0])
print("LR Q3 intercept:", model.intercept_)
y_pred = model.predict(X_test)
rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred))
r2_simple = model.score(X_test, y_test)
print("LR Q3 RMSE:", rmse_simple)
print("LR Q3 test R²:", r2_simple)
# The slope is about 1: each 1-degree increase in the daily low goes with
# roughly a 1-degree increase in the predicted daily high. Warmer nights mean
# warmer days.

# Q4
X_full = np.column_stack([temp_min, is_rainy])
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_full, temp_max, test_size=0.2, random_state=42
)
model_full = LinearRegression()
model_full.fit(Xf_train, yf_train)
r2_full = model_full.score(Xf_test, yf_test)
print("\nLR Q4 test R² (two features):", r2_full)
print("LR Q4 test R² (Q3, temp_min only):", r2_simple)
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# LR Q4 test R² (two features): 0.945061816962249
# LR Q4 test R² (Q3, temp_min only): 0.8944151516634623
# temp_min coefficient: 0.9764368687661398
# is_rainy coefficient: -2.9190219583163612

# Adding the rain flag raises R² noticeably, because it explains the gap
# between the two bands we saw in the scatter plot. The is_rainy coefficient
# (about -2.91 here; the data was built with -4, and the gap is sampling noise
# from a small test set) is the average change in the daily high on a rainy
# day versus a dry day, holding the low temperature fixed: rainy days are
# roughly 2.91 degrees cooler.

# Q5
y_pred_full = model_full.predict(Xf_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_pred_full, yf_test)
lims = [min(y_pred_full.min(), yf_test.min()), max(y_pred_full.max(), yf_test.max())]
plt.plot(lims, lims, "k--", label="Perfect prediction")
plt.title("Predicted vs Actual")
plt.xlabel("Predicted High")
plt.ylabel("Actual High")
plt.legend()
plt.savefig(OUT_DIR / "predicted_vs_actual_warmup.png", dpi=150, bbox_inches="tight")
plt.close()
# A point above the diagonal means the actual high was greater than the
# prediction (the model under-predicted). A point below the diagonal means the
# actual high was lower than predicted (the model over-predicted).
