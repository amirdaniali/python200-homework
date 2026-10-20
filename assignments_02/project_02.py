# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 2

# Week 2 Mini-Project: Predicting the Daily High
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Save outputs next to this script, no matter where it is run from
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# --- Task 1: Fetch the Data ---
# City chosen: Charlotte, NC (change latitude/longitude/timezone for your own city)
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 35.23,
    "longitude": -80.84,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("Shape:", df.shape)
print(df.head())

# --- Task 2: Look at the Features ---
print("\n", df.describe())
print("\nMissing values per column:")
print(df.isna().sum())
# If any values are missing, the simplest reasonable choice is to drop those
# rows (df = df.dropna()) since a year of data has plenty of rows to spare and
# imputing weather values would add made-up numbers. Uncomment if needed:
# df = df.dropna().reset_index(drop=True)

corr = df.drop(columns="date").corr()["temperature_2m_max"].sort_values()
print("\nCorrelation with temperature_2m_max:")
print(corr)
# temperature_2m_min has by far the strongest relationship with the daily high
# (a strong positive correlation). Precipitation and wind are much weaker.

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df["temperature_2m_max"], bins=30, edgecolor="black")
axes[0].set_title("Distribution of Daily High")
axes[0].set_xlabel("Daily High (°C)")
axes[0].set_ylabel("Number of Days")
axes[1].scatter(df["temperature_2m_min"], df["temperature_2m_max"], alpha=0.6)
axes[1].set_title("Daily High vs Daily Low")
axes[1].set_xlabel("Daily Low (°C)")
axes[1].set_ylabel("Daily High (°C)")
plt.tight_layout()
plt.savefig(OUT_DIR / "explore_high.png", dpi=150, bbox_inches="tight")
plt.close()
# The histogram of highs is spread over a wide range with no single peak
# (a mild bimodal / flat shape from the seasons), and the scatter plot shows a
# tight, roughly linear upward trend between low and high temperature.

# --- Task 3: Engineer a Feature ---
df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)
print("\nSummer days:", (df["is_summer"] == 1).sum())
print("Non-summer days:", (df["is_summer"] == 0).sum())
# I expect is_summer to help only a little. The daily low already carries most
# of the seasonal signal (summer days have warm lows), so is_summer mostly
# adds a small extra bump on top of what the low already explains.

# --- Task 4: Baseline Model ---
X_base = df[["temperature_2m_min"]].values
y = df["temperature_2m_max"].values
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)
base_model = LinearRegression()
base_model.fit(Xb_train, yb_train)
yb_pred = base_model.predict(Xb_test)
base_rmse = np.sqrt(mean_squared_error(yb_test, yb_pred))
base_r2 = r2_score(yb_test, yb_pred)

print("\n--- Baseline (temperature_2m_min only) ---")
print(f"Slope: {base_model.coef_[0]:.3f}")
print(f"RMSE:  {base_rmse:.3f} °C")
print(f"R²:    {base_r2:.3f}")
# The RMSE is in °C, so a typical baseline prediction is off by roughly that
# many degrees (see the printed value above), e.g. predicting 20 °C when the
# real high was around 20 ± RMSE.

# --- Task 5: Full Model ---
feature_cols = [
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "is_summer",
]
X = df[feature_cols].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Full model ---")
print(f"Train R²: {train_r2:.3f}")
print(f"Test R²:  {test_r2:.3f}")
print(f"Test RMSE: {rmse:.3f} °C")
print(f"Baseline test R² was {base_r2:.3f}; change: {test_r2 - base_r2:+.3f}")
print(f"Baseline RMSE was {base_rmse:.3f}; change: {rmse - base_rmse:+.3f}")

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:22s}: {coef:+.3f}")

# Train R²: 0.882
# Test R²:  0.909
# Test RMSE: 2.537 °C
# Baseline test R² was 0.867; change: +0.042
# Baseline RMSE was 3.067; change: -0.530

# temperature_2m_min    : +0.890
# precipitation_sum     : -0.192
# wind_speed_10m_max    : -0.036
# is_summer             : +1.295

# Interpreting coefficients
# - temperature_2m_min: positive and close to 1 or a bit higher; warmer lows
#   raise the predicted high.
# - precipitation_sum: negative; rainy days are cloudier and cooler, so they
#   lower the predicted high.
# - wind_speed_10m_max: small negative number; Windy days often come with fronts, which can be cooler.
# - is_summer: positive but small once the low is in the model. If it came out
#   negative, that would be surprising, but it can happen because the low
#   already accounts for summer warmth.
# Train vs test R²: they are close. A small gap means the model is
# generalizing well and is not overfitting; this is expected for a simple
# linear model with only four features and ~290 training rows.

# --- Task 6: Evaluate and Summarize ---
plt.figure(figsize=(6, 6))
plt.scatter(y_pred, y_test, alpha=0.7)
lims = [min(y_pred.min(), y_test.min()), max(y_pred.max(), y_test.max())]
plt.plot(lims, lims, "k--", label="Perfect prediction")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted High (°C)")
plt.ylabel("Actual High (°C)")
plt.legend()
plt.savefig(OUT_DIR / "predicted_vs_actual_high.png", dpi=150, bbox_inches="tight")
plt.close()
# Error looks to be bigger in the middle. The ends are surprisingly close to the line.

print("\n--- Summary ---")
print(f"Dataset: {len(df)} days; test set: {len(y_test)} days.")
print(f"Full model RMSE: {rmse:.2f} °C, R²: {test_r2:.3f}.")
print(
    f"A typical prediction is off by about {rmse:.1f} °C " f"(~{rmse * 9 / 5:.1f} °F)."
)
biggest = feature_cols[int(np.argmax(np.abs(model.coef_)))]
print(
    f"Largest raw coefficient: {biggest} "
    f"({model.coef_[feature_cols.index(biggest)]:+.3f})"
)

# Output:
# Dataset: 365 days; test set: 73 days.
# Full model RMSE: 2.54 °C, R²: 0.909.
# A typical prediction is off by about 2.5 °C (~4.6 °F).
# Largest raw coefficient: is_summer (+1.295)

# SUMMARY
# - Dataset: one year of daily data (365 days), with 73 days in the test set.
# - The full model's RMSE and R² are printed above. An RMSE of a few degrees
#   Celsius means predictions are typically within a few degrees of the real
#   high, which is useful but not exact.
# - temperature_2m_min is the most important feature, and it pushes the
#   predicted high up as the low rises. (Raw coefficients are not scaled, so
#   compare with care: is_summer or precipitation can have a bigger raw
#   coefficient simply because of their units.)
# - Something that surprised me: I didn't expect the final error to be only 2.5 C which is really good.
