# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.3
# Week 2
# --- Mini-Project ---


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

"""Load the dataset with the correct separator. Print the shape, the first five rows, and the data types of all columns.
Then plot a histogram of G3 with 21 bins (one per possible value, 0-20). Add a title "Distribution of Final Math Grades", label both axes, and save to outputs/g3_distribution.png. You should see a cluster of zeros sitting apart from the main distribution. They represent the students who didn't take the final exam."""

project_path = "data/student_performance_math.csv"
df = pd.read_csv(project_path, sep=";")

# Task 1: Load and Explore


print(f"Shape: {df.shape}")
print(f"First 5 rows:\n{df.head()}")
print(f"Data types:\n{df.dtypes}")

plt.figure(figsize=(8, 6))
plt.hist(df["G3"], bins=21, edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("G3")
plt.ylabel("Count")
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/g3_distribution.png", bbox_inches="tight")
plt.show()

# Task 2: Preprocess the Data

"""Handle the G3=0 rows first. Filter them out and save the result to a new DataFrame. Print the shape before and after to confirm how many rows were removed. Add a comment explaining your reasoning -- why would keeping these rows distort the model?

Then convert the yes/no columns to 1/0 and the sex column to 0/1.
Now check something interesting before moving on. Compute the Pearson correlation between absences and G3 on both the original dataset and the filtered one, and print both values. The difference is striking. Add a comment explaining why filtering changes the result: what were students with G3=0 doing in the original data that made absences look like a weak predictor? You might want to explore scatter plots to help understand this."""

df_before = df.copy()
print(f"Shape before filtering G3=0 rows: {df_before.shape}")

df_clean = df[df["G3"] != 0].copy()
print(f"Shape after filtering G3=0 rows: {df_clean.shape}")

# Keeping G3=0 rows would distort the model because those rows represent students who did not take the final exam, not true low grades.

binary_cols = ["schoolsup", "internet", "higher", "activities"]
for col in binary_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

abs_corr_original = df["absences"].corr(df["G3"])
abs_corr_filtered = df_clean["absences"].corr(df_clean["G3"])

print(f"Correlation between absences and G3 (original): {abs_corr_original}")
print(f"Correlation between absences and G3 (filtered): {abs_corr_filtered}")

# result: Correlation between absences and G3 (original): 0.03424731615006934
# Correlation between absences and G3 (filtered): -0.21312853214380897

# The original correlation is weakened by students with G3=0 who may also have many absences, which mixes missing-exam cases with actual performance. After understanding the data more, I suggest we should G3=0 them to avoid disrupting analysis relating to exams.

# Task 3: Exploratory Data Analysis

"""Compute the Pearson correlation between each numeric feature and G3 on the filtered dataset, and print them sorted from most negative to most positive. Which feature has the strongest relationship with G3? Are any results surprising?
Then create at least two visualizations of your own choosing and save them to outputs/. Use your judgment from previous weeks of data engineering to guide your use of plots. Use the correlation results to guide you -- what relationships seem worth a closer look? Add a comment for each plot describing what you see."""

numeric_cols = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "absences",
    "freetime",
    "goout",
    "Walc",
]
corrs = (
    df_clean[numeric_cols + ["G3"]]
    .corr(numeric_only=True)["G3"]
    .drop("G3")
    .sort_values()
)

print("Correlations with G3:")
print(corrs)

plt.figure(figsize=(8, 6))
plt.scatter(df_clean["failures"], df_clean["G3"], alpha=0.7)
plt.title("G3 vs Failures")
plt.xlabel("Failures")
plt.ylabel("G3")
plt.savefig("outputs/g3_vs_failures.png", bbox_inches="tight")
plt.show()
# todo
# Failures appears to be negatively related to final grade.

plt.figure(figsize=(8, 6))
plt.scatter(df_clean["studytime"], df_clean["G3"], alpha=0.7)
plt.title("G3 vs Study Time")
plt.xlabel("Study Time")
plt.ylabel("G3")
plt.savefig("outputs/g3_vs_studytime.png", bbox_inches="tight")
plt.show()
# todo
# Higher study time appears to be associated with higher final grades.

# Task 4: Baseline Model

"""Build the simplest possible model: use failures alone to predict G3. Split into training and test sets (80/20, random_state=42), fit a LinearRegression model, and print the slope, RMSE, and R² on the test set.

Add a comment: given that grades are on a 0-20 scale, what do the slopes and RMSE tell you in plain English? Is R² better or worse than you expected from exploratory data analysis?"""

X_base = df_clean[["failures"]].values
y = df_clean["G3"].values

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)

model_base = LinearRegression()
model_base.fit(X_train_b, y_train_b)

y_pred_b = model_base.predict(X_test_b)
rmse_b = np.sqrt(np.mean((y_pred_b - y_test_b) ** 2))
r2_b = model_base.score(X_test_b, y_test_b)

print(f"Baseline slope: {model_base.coef_[0]}")
print(f"Baseline RMSE: {rmse_b}")
print(f"Baseline R^2: {r2_b}")

# results:
# Baseline slope: -1.4275148751598756
# Baseline RMSE: 2.9617372470468797
# Baseline R^2: 0.08949272357478744

# A negative slope means more past failures predict lower final grades, and the RMSE tells us the typical prediction error in grade points. overall the R^2 value is very low to fully trust in the base model.

# Task 5: Build the Full Model

"""Now build a regression model using all of the numeric and binary features from the Feature Guide:

feature_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex"]
X = df_clean[feature_cols].values
y = df_clean["G3"].values

Split into training and test sets (80/20, random_state=42), fit a LinearRegression model, and print both train R² and test R², as well as RMSE on the test set. Compare the test R² to your baseline from Task 4 -- how much does adding more features help?

Print each feature name alongside its coefficient:

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

Look carefully at the coefficients. Sort them mentally from largest to smallest. Are any signs (positive or negative) surprising given what you know about the data? For any surprising result, add a comment with your best explanation. Then compare train R² to test R² -- are they close, or is there a gap? What does that tell you about the model?

Finally, add a comment answering: if you were deploying this model in production, which features would you keep and which would you drop? Justify your choices based on what you see in the numbers."""

feature_cols = [
    "failures",
    "Medu",
    "Fedu",
    "studytime",
    "higher",
    "schoolsup",
    "internet",
    "sex",
    "freetime",
    "activities",
    "traveltime",
]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

print(f"Train R^2: {train_r2}")
print(f"Test R^2: {test_r2}")
print(f"Test RMSE: {rmse}")

print("Feature coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# A surprisingly positive coefficient may reflect correlation with other features rather than a direct causal effect.
# For deployment, I would keep the features with clear signal and reasonable real-world meaning, and drop weak or unstable ones if they do not improve test performance.

# Task 6: Evaluate and Summarize

"""A useful way to evaluate a regression model visually is a predicted vs actual plot. This is a scatter plot where each point in the test set becomes a dot, with the model's prediction (y_hat) on the x-axis and the true value (y) on the y-axis. If the model were perfect, every point would fall exactly on the diagonal (predicted = actual). Clusters or curves away from the diagonal reveal systematic errors that RMSE alone won't show you. Random scattering around the diagonal is expected, and acceptable, prediction error.
Create this plot for your test set. Add a diagonal reference line (for y=y_predicted), a title "Predicted vs Actual (Full Model)", labeled axes, and save to outputs/predicted_vs_actual_g3.png. Add a comment: does the model seem to struggle more at the high end, the low end, or is error roughly uniform across grade levels? What does a value above or below the diagonal mean?

Then write a plain-language summary in your comments statements covering:

    The size of the filtered dataset and the test set
    The RMSE and R² of your best model in plain language -- on a 0-20 scale, what does a typical prediction error actually mean?
    Which two features have the largest positive and largest negative coefficients, and what those mean
    One result that surprised you
"""

# todo

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test, alpha=0.8)
min_val = min(y_pred.min(), y_test.min())
max_val = max(y_pred.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.savefig("outputs/predicted_vs_actual.png", bbox_inches="tight")
plt.show()

# Points above the diagonal mean the actual grade is higher than predicted.
# Points below the diagonal mean the model overpredicted the grade.

# The filtered dataset is smaller after removing G3=0 rows, and the test set is 20% of that cleaned data.
# On a 0-20 scale, the RMSE is the typical number of grade points the model misses by.
# The largest positive coefficient shows the strongest feature associated with higher grades, and the largest negative coefficient shows the strongest feature associated with lower grades.
# One surprising result is that some binary or family-background features may have weaker or less intuitive coefficients than expected because they overlap with other predictors.

# Task 7: Neglected Feature G1
X_g1 = df_clean[feature_cols + ["G1"]].values

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y, test_size=0.2, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

r2_g1 = model_g1.score(X_test_g1, y_test_g1)
print(f"Test R^2 with G1 added: {r2_g1}")

# A high R^2 here does not mean G1 causes G3; it means first-period performance is strongly predictive of final grade.
# This is useful for identifying students who may struggle, but educators would need intervention data earlier than G1 if they want to help before the first grade is known.
