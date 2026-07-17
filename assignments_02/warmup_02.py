# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.3
# Week 2

# --- The scikit-learn API ---

# Sklearn Q1

"""The core pattern in scikit-learn is create → fit → predict. Practice it here with a simple dataset: years of work experience versus annual salary.

import numpy as np
from sklearn.linear_model import LinearRegression

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

Create a LinearRegression model, fit it to this data, and then predict the salary for someone with 4 years of experience and someone with 8 years. Print the slope (model.coef_[0]), the intercept (model.intercept_), and the two predictions. Label each printed value.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)
print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"Salary for someone with 4 years of experience: {model.predict([[4]])}")
print(f"Salary for someone with 8 years of experience: {model.predict([[8]])}")

# Sklearn Q2

"""scikit-learn requires the feature array X to be 2D even when you only have one feature. Start with this 1D array:

x = np.array([10, 20, 30, 40, 50])

Print its shape. Use .reshape() to convert it to a 2D array and print the new shape. Add a comment explaining, in your own words, why scikit-learn needs X to be 2D."""

x = np.array([10, 20, 30, 40, 50])

x_modified = x.reshape(-1, 1)
print(f"Original Shape: {x.shape}")
print(f"Modified Shape: {x_modified.shape}")

"""
Scikit-learn needs X to be 2D to distinguish between features and samples. The x[0] is the first feature, and x[1] is the second feature and so on. Each feature has multiple samples. Scikit needs to know if we have 5 features total with 1 sample each or if we have 1 feature with 5 samples."""

# Sklearn Q3

"""K-Means is an unsupervised algorithm that follows the same create → fit → predict pattern as everything else in scikit-learn. Use the code below to generate a synthetic dataset with three natural clusters:

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

Create a KMeans model with n_clusters=3 and random_state=42, fit it to X_clusters, and predict a cluster label for each point. Print the cluster centers (kmeans.cluster_centers_) and how many points fell into each cluster using np.bincount(labels).
Then create a scatter plot coloring each point by its cluster label, plot the cluster centers as black X's, add a title and axis labels. Save the figure to outputs/kmeans_clusters.png."""


X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

# plot: what K-Means discovers
kmeans = KMeans(n_clusters=3, random_state=42)  # 1. Create the model
kmeans.fit(X_clusters)  # 2. Fit -- find cluster centers
labels = kmeans.predict(X_clusters)  # 3. Predict a label for each point

for i, center in enumerate(kmeans.cluster_centers_):
    plt.plot(center[0], center[1], "kx")
    print(f"Cluster Center No {i}: {center}")
    print(f"Number of points in cluster {i}: {np.bincount(labels)[i]}")

plt.scatter(
    X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap="viridis", s=60, alpha=0.7
)

plt.title("Clusters Found by K-Means")
plt.xlabel("Data Points")


plt.tight_layout()
plt.show()

# Linear Regression

# Linear Regression Q1

"""Linear Regression

The questions below all use the same synthetic medical costs dataset: 100 patients, each with an age (20 to 65), a smoker flag (0 = non-smoker, 1 = smoker), and an annual medical cost as the target. Generate it once and reuse the variables throughout.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

Linear Regression Question 1
Before fitting anything, look at the data. Create a scatter plot of age on the x-axis and cost on the y-axis. Color the points by smoker status by passing c=smoker and cmap="coolwarm" to plt.scatter(). Add a title "Medical Cost vs Age", label both axes, and save to outputs/cost_vs_age.png.
Add a comment describing what you see. Are there two distinct groups visible? What does that suggest about the smoker variable?"""


np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

plt.figure(figsize=(8, 6))
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.legend(["Smoker", "Cost"])
plt.ylabel("Annual Medical Cost")
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/cost_vs_age.png", bbox_inches="tight")
plt.show()

# The plot should show two visiblly distinct groups, which suggests smoker status has a strong effect on medical cost.

# Linear Regression Q2

"""Split the data into training and test sets using age as the only feature, an 80/20 split, and random_state=42. Reshape age to a 2D array before using it as X. Print the shapes of all four arrays."""

X_age = age.reshape(-1, 1)
y_cost = cost

X_train, X_test, y_train, y_test = train_test_split(
    X_age, y_cost, test_size=0.2, random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Linear Regression Q3

"""Fit a LinearRegression model to your training data from Question 2. Print the slope and intercept. Then predict on the test set and print:

    RMSE: np.sqrt(np.mean((y_pred - y_test) ** 2))
    R² on the test set: model.score(X_test, y_test)

Add a comment interpreting the slope in plain English -- what does it mean for medical costs?"""

model_age = LinearRegression()
model_age.fit(X_train, y_train)

y_pred = model_age.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2_test = model_age.score(X_test, y_test)

print(f"Slope: {model_age.coef_[0]}")
print(f"Intercept: {model_age.intercept_}")
print(f"RMSE: {rmse}")
print(f"Test R^2: {r2_test}")

# The slope and intercept are both positive. R^2 is roughly 0.069 percent which is very low. The plot roughly means that cost increases with age. Although the R^2 is very low in my opinion.

# Linear Regression Q4

"""Now add smoker as a second feature and fit a new model.

X_full = np.column_stack([age, smoker])

Split, fit, and print the test R². Compare it to the R² from Question 3 -- does adding the smoker flag help? Print both coefficients:

print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

Add a comment interpreting the smoker coefficient: what does it represent in practical terms?"""

X_full = np.column_stack([age, smoker])

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_cost, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

y_pred_full = model_full.predict(X_test_full)
r2_full = model_full.score(X_test_full, y_test_full)

print(f"Age coefficient:    {model_full.coef_[0]}")
print(f"Smoker coefficient: {model_full.coef_[1]}")
print(f"Full model test R^2: {r2_full}")
print(f"Age-only model test R^2: {r2_test}")

# The smoker coefficient represents the expected extra annual cost for smokers compared with non-smokers, holding age constant. It increases the R^2 significantly to 0.77 which is a vast improvement in explaining the correlation between the data.

# Linear Regression Q5

"""A predicted vs actual plot is a standard tool for evaluating regression models. Each test observation becomes a dot: the model's prediction goes on the x-axis, the true value goes on the y-axis. A perfect model would place every point on the diagonal line where predicted equals actual.
Using the two-feature model from Linear Regression Question 4, create this plot for the test set. Add a diagonal reference line, a title "Predicted vs Actual", labeled axes, and save to outputs/predicted_vs_actual_cost.png.

Add a comment: what does it mean when a point falls above the diagonal? What about below?"""

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_full, y_test_full, alpha=0.8)
min_val = min(y_pred_full.min(), y_test_full.min())
max_val = max(y_pred_full.max(), y_test_full.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
plt.title("Predicted vs Actual")
plt.xlabel("Predicted")
plt.ylabel("Actual")
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/predicted_vs_actual.png", bbox_inches="tight")
plt.show()

# A point above the diagonal means the actual value is higher than the prediction.
# A point below the diagonal means the model predicted a value that was too high.
